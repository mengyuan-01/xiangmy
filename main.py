import os
import re
import sys
import json
import uuid
import sqlite3
import threading
import functools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote, parse_qs

# 隐藏 Python 控制台黑窗口：启动后桌面只剩应用窗口，不再有命令行走出来遮挡内容
try:
    import ctypes
    _console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if _console_hwnd:
        ctypes.windll.user32.ShowWindow(_console_hwnd, 0)
except Exception:
    pass

import webview
from ai_chai_p import AIService


def _plain_preview(html, limit=80):
    """把笔记 HTML 正文转成纯文本摘要，避免列表里显示成一堆标签。"""
    if not html:
        return ''
    text = re.sub(r'<[^>]+>', ' ', str(html))
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:limit]



# ============================================================
# 路径
# ============================================================
def get_root():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


ROOT = get_root()
DATA_DIR = os.path.join(ROOT, 'shuj', 'swdt')
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, 'swdt.db')

HTML_DIR = os.path.join(ROOT, 'html')
if not os.path.isdir(HTML_DIR) and hasattr(sys, '_MEIPASS'):
    HTML_DIR = os.path.join(sys._MEIPASS, 'html')


# ============================================================
# 数据层
# ============================================================
class Api:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        self._init_tables()
        self._init_ai_tables()

        # 已删除的导图墓碑：多个前端实例（pywebview 窗口 / 浏览器）各自持有旧快照时，
        # 旧实例的全量保存会把刚删掉的导图写回来。记下删除的 id，保存时跳过它。
        self._deleted_mindmap_ids = set()

        # AI 服务：RAG 懒加载，不阻塞启动
        # 知识来源 = 『笔记』页面里的笔记（SQLite notes 表），不再是空的 bij 目录
        self.ai = AIService(
            notes_dir=os.path.join(ROOT, 'bij'),
            enable_rag=True,
            notes_provider=self.load_notes_full,
            rag_kwargs={
                "chroma_path": os.path.join(ROOT, 'chroma_notes_db'),
                "cache_file":  os.path.join(ROOT, 'nodes_bm25_cache.pkl'),
                "models_dir":  os.path.join(ROOT, 'models'),
            },
        )
        env_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
        if env_key:
            self.ai.set_api_key(env_key)

        # 读取本地 AI 配置（用户"写入"的 Key/平台/模型持久化，供炼炉出题等后端默认调用）
        self._ai_config_path = os.path.join(ROOT, 'shuj', 'ai_config.json')
        try:
            if os.path.isfile(self._ai_config_path):
                with open(self._ai_config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f) or {}
                if cfg.get('platform'):
                    self.ai.platform = cfg['platform']
                if cfg.get('model'):
                    self.ai.model = cfg['model']
                file_key = (cfg.get('api_key') or '').strip()
                if not self.ai.api_key and file_key:
                    self.ai.set_api_key(file_key)
                print(f"⚙️ 已加载 AI 配置: {self.ai.platform} / {self.ai.model or '(默认)'} / Key长度 {len(self.ai.api_key or '')}")
        except Exception as e:
            print(f'⚠️ 读取 AI 配置失败: {e}')

    def save_ai_config(self, config):
        """把平台/Key/模型写入本地配置文件并立即生效"""
        try:
            config = config or {}
            cfg = {
                'platform': config.get('platform') or 'guiji',
                'api_key':  config.get('api_key') or '',
                'model':    config.get('model') or '',
            }
            os.makedirs(os.path.dirname(self._ai_config_path), exist_ok=True)
            with open(self._ai_config_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            self.ai.platform = cfg['platform']
            if cfg['model']:
                self.ai.model = cfg['model']
            if cfg['api_key']:
                self.ai.set_api_key(cfg['api_key'])
            self.ai._llm = None  # 让下次调用重建客户端
            print(f"💾 AI 配置已保存: {cfg['platform']} / {cfg['model']}")
            return {"ok": True}
        except Exception as e:
            print(f'❌ save_ai_config: {e}')
            return {"ok": False, "error": str(e)}

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS mindmaps (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                color       TEXT NOT NULL DEFAULT '#409eff',
                data        TEXT NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT UNIQUE NOT NULL,
                title       TEXT NOT NULL,
                content     TEXT,
                remark      TEXT DEFAULT '',
                status      TEXT NOT NULL DEFAULT 'normal',
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cols = [r[1] for r in cur.execute("PRAGMA table_info(notes)").fetchall()]
        if 'remark' not in cols:
            try:
                cur.execute("ALTER TABLE notes ADD COLUMN remark TEXT DEFAULT ''")
                print('🔧 已为 notes 表补充 remark 列')
            except Exception as e:
                print(f'⚠️ 补充 remark 列失败: {e}')
        cur.execute("CREATE INDEX IF NOT EXISTS idx_notes_status ON notes(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_notes_updated ON notes(updated_at)")
        # 知识炼炉：出题项目
        cur.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id          INTEGER PRIMARY KEY,
                name        TEXT NOT NULL,
                questions   TEXT NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _init_ai_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id          TEXT PRIMARY KEY,
                title       TEXT,
                timestamp   INTEGER,
                messages    TEXT,
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_ts ON chats(timestamp)")
        self.conn.commit()

    # ========================================================
    # 思维导图
    # ========================================================
    def load_mindmaps(self):
        try:
            rows = self.conn.execute(
                "SELECT id, name, color, data FROM mindmaps ORDER BY created_at"
            ).fetchall()
            return [{
                'id':    r['id'],
                'name':  r['name'],
                'color': r['color'] or '#409eff',
                'data':  json.loads(r['data']),
            } for r in rows]
        except Exception as e:
            print(f'❌ load_mindmaps: {e}')
            return []

    def save_mindmaps(self, mindmaps):
        try:
            if not isinstance(mindmaps, list):
                return {'ok': False, 'error': 'expected array'}

            cur = self.conn.cursor()
            existing = {r[0] for r in cur.execute("SELECT id FROM mindmaps")}
            incoming = {str(mm['id']) for mm in mindmaps if mm.get('id')}

            # 空数组 + 数据库非空 → 拒绝（防误清空）
            if len(incoming) == 0 and len(existing) > 0:
                return {'ok': False, 'error': '拒绝用空数组清空导图'}

            to_del = existing - incoming
            if to_del:
                cur.executemany("DELETE FROM mindmaps WHERE id=?",
                                [(i,) for i in to_del])

            for mm in mindmaps:
                mid = mm.get('id')
                if not mid:
                    continue
                # 跳过已被删除的导图（防止其它窗口的旧快照把它复活）
                if str(mid) in self._deleted_mindmap_ids:
                    continue
                cur.execute("""
                    INSERT INTO mindmaps (id, name, color, data, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        color = excluded.color,
                        data = excluded.data,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    str(mid),
                    str(mm.get('name') or '未命名导图'),
                    str(mm.get('color') or '#409eff'),
                    json.dumps(mm.get('data') or {}, ensure_ascii=False),
                ))
            self.conn.commit()
            print(f'💾 保存 {len(mindmaps)} 张导图')
            return {'ok': True, 'count': len(mindmaps)}
        except Exception as e:
            print(f'❌ save_mindmaps: {e}')
            return {'ok': False, 'error': str(e)}

    def delete_mindmap(self, mid):
        """删除单张导图并登记墓碑，避免其它前端实例用旧快照把它写回来。"""
        try:
            mid = str(mid)
            self._deleted_mindmap_ids.add(mid)
            if len(self._deleted_mindmap_ids) > 200:
                self._deleted_mindmap_ids = set(list(self._deleted_mindmap_ids)[-100:])
            self.conn.execute("DELETE FROM mindmaps WHERE id = ?", (mid,))
            self.conn.commit()
            print(f'🗑️ 已删除导图 {mid}')
            return {'ok': True, 'id': mid}
        except Exception as e:
            print(f'❌ delete_mindmap: {e}')
            return {'ok': False, 'error': str(e)}

    # ========================================================
    # 笔记 CRUD
    # ========================================================
    def load_notes(self):
        try:
            rows = self.conn.execute("""
                SELECT id, filename, title,
                       substr(content, 1, 1000) AS preview,
                       remark, status, updated_at
                FROM notes
                WHERE status='normal'
                ORDER BY updated_at DESC
            """).fetchall()
            return [{
                'id':         r['id'],
                'filename':   r['filename'],
                'title':      r['title'] or '无标题',
                'preview':    _plain_preview(r['preview']),
                'remark':     r['remark'] or '',
                'status':     r['status'],
                'updated_at': r['updated_at'],
            } for r in rows]
        except Exception as e:
            print(f'❌ load_notes: {e}')
            return []

    def load_note_content(self, filename):
        try:
            row = self.conn.execute(
                "SELECT content FROM notes WHERE filename=?", (filename,)
            ).fetchone()
            return {'content': (row['content'] if row else '') or ''}
        except Exception as e:
            print(f'❌ load_note_content: {e}')
            return {'content': ''}

    def create_note(self, title, content=''):
        try:
            filename = f"{uuid.uuid4().hex}.html"
            self.conn.execute(
                """INSERT INTO notes (filename, title, content, status, updated_at)
                   VALUES (?, ?, ?, 'normal', CURRENT_TIMESTAMP)""",
                (filename, title or '未命名', content or '')
            )
            self.conn.commit()
            print(f'🆕 新建笔记: {filename} / {title}')
            self._invalidate_rag()
            return {'ok': True, 'filename': filename}
        except Exception as e:
            print(f'❌ create_note: {e}')
            return {'ok': False, 'error': str(e)}

    def update_note(self, filename, title=None, content=None, remark=None):
        try:
            sets, params = [], []
            if title is not None:
                sets.append("title=?");    params.append(title)
            if content is not None:
                sets.append("content=?");  params.append(content)
            if remark is not None:
                sets.append("remark=?");   params.append(remark)

            if not sets:
                return {'ok': True, 'note': 'nothing to update'}

            sets.append("updated_at=CURRENT_TIMESTAMP")
            params.append(filename)

            self.conn.execute(
                f"UPDATE notes SET {', '.join(sets)} WHERE filename=?",
                params
            )
            self.conn.commit()
            self._invalidate_rag()
            return {'ok': True}
        except Exception as e:
            print(f'❌ update_note: {e}')
            return {'ok': False, 'error': str(e)}

    def delete_note(self, filename):
        try:
            self.conn.execute("DELETE FROM notes WHERE filename=?", (filename,))
            self.conn.commit()
            print(f'🗑️ 删除笔记: {filename}')
            self._invalidate_rag()
            return {'ok': True}
        except Exception as e:
            print(f'❌ delete_note: {e}')
            return {'ok': False, 'error': str(e)}

    def archive_note(self, filename):
        try:
            self.conn.execute(
                "UPDATE notes SET status='archived', updated_at=CURRENT_TIMESTAMP "
                "WHERE filename=?", (filename,)
            )
            self.conn.commit()
            print(f'📦 归档笔记: {filename}')
            self._invalidate_rag()
            return {'ok': True}
        except Exception as e:
            print(f'❌ archive_note: {e}')
            return {'ok': False, 'error': str(e)}

    def unarchive_note(self, filename):
        """取消归档：把归档箱里的笔记放回正常列表"""
        try:
            row = self.conn.execute(
                "SELECT filename FROM notes WHERE filename=?", (filename,)
            ).fetchone()
            if not row:
                return {'ok': False, 'error': '笔记不存在'}
            self.conn.execute(
                "UPDATE notes SET status='normal', updated_at=CURRENT_TIMESTAMP "
                "WHERE filename=?", (filename,)
            )
            self.conn.commit()
            print(f'♻️ 取消归档: {filename}')
            self._invalidate_rag()
            return {'ok': True}
        except Exception as e:
            print(f'❌ unarchive_note: {e}')
            return {'ok': False, 'error': str(e)}

    def _invalidate_rag(self):
        """笔记变化后让 RAG 索引失效，下次问答自动用最新笔记重建"""
        try:
            self.ai.invalidate_rag()
        except Exception as e:
            print(f'⚠️ 标记 RAG 索引失效失败: {e}')

    # ========================================================
    # 知识炼炉：项目持久化
    # ========================================================
    def load_notes_full(self, status='normal'):
        """HTTP 用：返回完整笔记（含 content），供炼炉/降级列表使用
        status: normal 正常笔记；archived 归档箱；all 全部
        """
        try:
            if status not in ('normal', 'archived', 'all'):
                status = 'normal'
            sql = ("SELECT id, filename, title, content, remark, status, updated_at "
                   "FROM notes")
            if status == 'all':
                sql += " ORDER BY updated_at DESC"
                rows = self.conn.execute(sql).fetchall()
            else:
                sql += " WHERE status=? ORDER BY updated_at DESC"
                rows = self.conn.execute(sql, (status,)).fetchall()
            return [{
                'id':         r['id'],
                'filename':   r['filename'],
                'title':      r['title'] or '无标题',
                'content':    r['content'] or '',
                'preview':    _plain_preview(r['content']),
                'remark':     r['remark'] or '',
                'status':     r['status'],
                'updated_at': r['updated_at'],
            } for r in rows]
        except Exception as e:
            print(f'❌ load_notes_full: {e}')
            return []

    def load_projects(self):
        try:
            rows = self.conn.execute(
                "SELECT id, name, questions, created_at FROM projects ORDER BY id DESC"
            ).fetchall()
            out = []
            for r in rows:
                try:
                    questions = json.loads(r['questions'] or '{}')
                except json.JSONDecodeError:
                    questions = {}
                out.append({
                    'id':        r['id'],
                    'name':      r['name'] or '未命名项目',
                    'questions': questions,
                    'createdAt': r['created_at'] or '',
                })
            return out
        except Exception as e:
            print(f'❌ load_projects: {e}')
            return []

    def save_projects(self, projects):
        try:
            if not isinstance(projects, list):
                return {'ok': False, 'error': 'expected array'}

            cur = self.conn.cursor()
            existing = {r[0] for r in cur.execute("SELECT id FROM projects")}
            incoming = set()
            for p in projects:
                try:
                    pid = int(p.get('id'))
                except (TypeError, ValueError):
                    continue
                incoming.add(pid)
                cur.execute("""
                    INSERT INTO projects (id, name, questions, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET
                        name=excluded.name,
                        questions=excluded.questions,
                        updated_at=CURRENT_TIMESTAMP
                """, (
                    pid,
                    str(p.get('name') or '未命名项目'),
                    json.dumps(p.get('questions') or {}, ensure_ascii=False),
                ))

            to_del = existing - incoming
            if to_del:
                cur.executemany("DELETE FROM projects WHERE id=?",
                                [(i,) for i in to_del])
            self.conn.commit()
            print(f'💾 保存 {len(incoming)} 个炼炉项目')
            return {'ok': True, 'count': len(incoming)}
        except Exception as e:
            print(f'❌ save_projects: {e}')
            return {'ok': False, 'error': str(e)}

    # ========================================================
    # 知识炼炉：AI 出题
    # ========================================================
    def generate_questions(self, filename):
        """
        根据笔记全文，让 LLM 生成 5 类题目。
        返回 {'success': bool, 'questions': {...}, 'error': str}
        """
        try:
            row = self.conn.execute(
                "SELECT content, title FROM notes WHERE filename=?", (filename,)
            ).fetchone()
            if not row:
                return {'success': False, 'error': '笔记不存在', 'questions': EMPTY_QUESTIONS()}
            content = (row['content'] or '').strip()
            if not content:
                return {'success': False, 'error': '笔记内容为空', 'questions': EMPTY_QUESTIONS()}

            if not self.ai.api_key:
                return {'success': False, 'error': '未配置 API Key，请设置环境变量 SILICONFLOW_API_KEY',
                        'questions': EMPTY_QUESTIONS()}

            # 去 HTML 标签，压缩空白
            text = re.sub(r'<[^>]+>', ' ', content)
            text = re.sub(r'\s+', ' ', text).strip()[:8000]

            schema = (
                '{"choice": [{"question": "题干", "options": ["A. ", "B. ", "C. ", "D. "], "answer": "A"}], '
                '"multi": [{"question": "题干", "options": ["A. ", "B. ", "C. ", "D. "], "answer": "AB"}], '
                '"fill": [{"question": "题干（用_____表示填空）", "answer": "参考答案"}], '
                '"dictation": [{"title": "默写标题", "hint": "提示词"}], '
                '"reading": [{"passage": "短文", "questions": [{"question": "小题"}]}]}'
            )
            prompt = (
                "你是一位资深出题专家。请根据下面提供的笔记内容，生成一套练习题，"
                "必须严格输出 JSON（不要输出任何解释或 Markdown 代码块之外的文字），格式如下：\n"
                f"{schema}\n\n"
                "数量要求：choice 3 题、multi 2 题、fill 2 题、dictation 1 题、reading 1 篇（含 2 个小题）。"
                "所有题目必须基于笔记内容，不得编造。\n\n"
                f"笔记内容：\n{text}"
            )
            raw = self.ai.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=4096,
            )
            questions = parse_questions_json(raw)
            total = sum(len(v) for v in questions.values())
            if total == 0:
                return {'success': False, 'error': 'AI 返回的题目无法解析',
                        'questions': EMPTY_QUESTIONS()}
            return {'success': True, 'questions': questions}
        except Exception as e:
            print(f'❌ generate_questions: {e}')
            return {'success': False, 'error': str(e), 'questions': EMPTY_QUESTIONS()}

    # ========================================================
    # AI 接口
    # ========================================================
    def set_ai_key(self, key):
        try:
            self.ai.set_api_key(key or "")
            print(f"🔑 AI Key 已更新（长度 {len(self.ai.api_key)}）")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def ai_health(self):
        try:
            return self.ai.health()
        except Exception as e:
            return {"error": str(e)}

    def teach(self, question, config=None, use_classifier=True):
        """RAG 问答；config 可选（页面级平台/Key/模型）；use_classifier 控制 A/B/C 三档判断"""
        try:
            return self.ai.teach(question, use_classifier=use_classifier, config=config)
        except Exception as e:
            print(f'❌ teach: {e}')
            return {"answer": f"❌ 后端异常: {e}", "branch": "C",
                    "sources": [], "entity": ""}

    def ai_chat(self, text):
        """纯聊天"""
        try:
            return {"answer": self.ai.chat(text)}
        except Exception as e:
            return {"answer": f"❌ {e}"}

    def ai_clear_history(self):
        try:
            self.ai.clear_history()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ========================================================
    # 会话持久化
    # ========================================================
    def load_chats(self):
        try:
            rows = self.conn.execute(
                "SELECT id, title, timestamp, messages FROM chats "
                "ORDER BY timestamp DESC"
            ).fetchall()
            out = []
            for r in rows:
                try:
                    msgs = json.loads(r["messages"] or "[]")
                except json.JSONDecodeError:
                    msgs = []
                out.append({
                    "id":        r["id"],
                    "title":     r["title"] or "新会话",
                    "timestamp": r["timestamp"] or 0,
                    "messages":  msgs,
                })
            return out
        except Exception as e:
            print(f'❌ load_chats: {e}')
            return []

    def save_chats(self, chats, allow_empty=False):
        try:
            if not isinstance(chats, list):
                return {"ok": False, "error": "expected array"}

            # 空数组 + 数据库非空 → 默认拒绝（防误清空）；
            # 用户主动删掉最后一个会话时前端会带 ?force=1，此时放行
            if len(chats) == 0:
                n = self.conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
                if n > 0 and not allow_empty:
                    return {"ok": False, "error": "拒绝用空数组清空会话（需 force=1）"}

            cur = self.conn.cursor()
            incoming = set()
            for c in chats:
                cid = str(c.get("id") or "")
                if not cid:
                    continue
                incoming.add(cid)
                cur.execute("""
                    INSERT INTO chats (id, title, timestamp, messages, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET
                        title=excluded.title,
                        timestamp=excluded.timestamp,
                        messages=excluded.messages,
                        updated_at=CURRENT_TIMESTAMP
                """, (
                    cid,
                    str(c.get("title") or "新会话"),
                    int(c.get("timestamp") or 0),
                    json.dumps(c.get("messages") or [], ensure_ascii=False),
                ))

            existing = {r[0] for r in cur.execute("SELECT id FROM chats")}
            to_del = existing - incoming
            if to_del:
                cur.executemany("DELETE FROM chats WHERE id=?",
                                [(i,) for i in to_del])
            self.conn.commit()
            return {"ok": True, "count": len(incoming)}
        except Exception as e:
            print(f'❌ save_chats: {e}')
            return {"ok": False, "error": str(e)}

    def delete_chat(self, chat_id):
        """精确删除单个会话（删最后一个也用得上，不受空数组保护影响）"""
        try:
            cid = str(chat_id or "").strip()
            if not cid:
                return {"ok": False, "error": "missing chat id"}
            cur = self.conn.cursor()
            n = cur.execute("DELETE FROM chats WHERE id=?", (cid,)).rowcount
            self.conn.commit()
            print(f'🗑️ 删除会话 {cid}（{n} 行）')
            return {"ok": True, "deleted": n, "id": cid}
        except Exception as e:
            print(f'❌ delete_chat: {e}')
            return {"ok": False, "error": str(e)}

    def delete_project(self, project_id):
        """精确删除单个炼炉项目"""
        try:
            pid = int(str(project_id).strip())
            cur = self.conn.cursor()
            n = cur.execute("DELETE FROM projects WHERE id=?", (pid,)).rowcount
            self.conn.commit()
            print(f'🗑️ 删除项目 {pid}（{n} 行）')
            return {"ok": True, "deleted": n, "id": pid}
        except Exception as e:
            print(f'❌ delete_project: {e}')
            return {"ok": False, "error": str(e)}

    # ========================================================
    # 通用工具
    # ========================================================
    def list_methods(self):
        return [m for m in dir(self) if not m.startswith('_')]


# ============================================================
# 出题 JSON 解析
# ============================================================
def EMPTY_QUESTIONS():
    return {"choice": [], "multi": [], "fill": [], "dictation": [], "reading": []}


def parse_questions_json(raw):
    """容错解析 LLM 返回的题目 JSON，任何失败都回退为空五类"""
    questions = EMPTY_QUESTIONS()
    if not raw:
        return questions
    text = raw.strip()
    # 去掉 markdown 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        # 没有大括号：走宽松逐类型提取
        return _extract_questions_loose(text, questions)
    body = text[s:e + 1]
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return _extract_questions_loose(body, questions)
    if isinstance(data, dict) and any(k in data for k in questions):
        for t in questions:
            items = data.get(t)
            if isinstance(items, list):
                questions[t] = [it for it in items if isinstance(it, dict)]
        return questions
    # 解析成功但结构不对（例如截到了单个题目对象）→ 全文宽松提取
    return _extract_questions_loose(text, questions)


def _extract_questions_loose(body, questions):
    """宽松模式：按 '"类型": [' 逐段提取 JSON 数组"""
    for t in questions:
        seg = re.search(r'"' + t + r'"\s*:\s*\[', body)
        if not seg:
            continue
        start = seg.end() - 1
        depth = 0
        for i in range(start, len(body)):
            if body[i] == '[':
                depth += 1
            elif body[i] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        items = json.loads(body[start:i + 1])
                        if isinstance(items, list):
                            questions[t] = [it for it in items if isinstance(it, dict)]
                    except json.JSONDecodeError:
                        pass
                    break
    return questions


# ============================================================
# 内置 HTTP 服务（供 ai_chat / knowledge_furnace / 降级 fetch）
# ============================================================
MIME_MAP = {
    ".html": "text/html; charset=utf-8",
    ".htm":  "text/html; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".mjs":  "application/javascript; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".svg":  "image/svg+xml",
    ".ico":  "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf":  "font/ttf",
    ".map":  "application/json",
}


def make_handler(api, html_dir):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        _api = api
        _html_dir = html_dir

        def log_message(self, fmt, *args):
            pass  # 静默，避免每次请求刷屏

        # ---------- helpers ----------
        def _send_json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            # 允许浏览器用 file:// 直接打开 html 时跨源调用本地接口
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send_error_json(self, status, msg):
            self._send_json({"error": msg}, status)

        def _read_json(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0:
                return None
            try:
                raw = self.rfile.read(length)
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return None

        def _serve_static(self, relpath):
            base = os.path.realpath(self._html_dir)
            target = os.path.realpath(os.path.join(base, relpath.lstrip("/\\")))
            if not (target == base or target.startswith(base + os.sep)):
                self._send_error_json(404, "not found")
                return
            if not os.path.isfile(target):
                self._send_error_json(404, "not found")
                return
            ext = os.path.splitext(target)[1].lower()
            ctype = MIME_MAP.get(ext, "application/octet-stream")
            with open(target, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        # ---------- routes ----------
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/swdt":
                return self._send_json(self._api.load_mindmaps())
            if path == "/api/notes":
                qs = parse_qs(urlparse(self.path).query)
                status = (qs.get("status", ["normal"])[0] or "normal").lower()
                return self._send_json(self._api.load_notes_full(status))
            if path.startswith("/api/notes/") and not path.startswith("/api/notes/archive/"):
                fn = unquote(path[len("/api/notes/"):])
                return self._send_json(self._api.load_note_content(fn))
            if path == "/api/chats":
                return self._send_json(self._api.load_chats())
            if path == "/api/projects":
                return self._send_json(self._api.load_projects())
            if path == "/api/health":
                return self._send_json({"ok": True, "ai": self._api.ai_health()})
            if path in ("/", ""):
                path = "/swdt.html"
            return self._serve_static(path)

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/swdt":
                data = self._read_json()
                if not isinstance(data, list):
                    return self._send_error_json(400, "expected array")
                return self._send_json(self._api.save_mindmaps(data))
            if path == "/api/notes":
                data = self._read_json() or {}
                return self._send_json(self._api.create_note(
                    data.get("title", "未命名"), data.get("content", "")))
            if path.startswith("/api/notes/archive/"):
                fn = unquote(path[len("/api/notes/archive/"):])
                return self._send_json(self._api.archive_note(fn))
            if path == "/api/chats":
                data = self._read_json()
                if not isinstance(data, list):
                    return self._send_error_json(400, "expected array")
                qs = parse_qs(urlparse(self.path).query)
                force = (qs.get("force", ["0"])[0] or "").lower() in ("1", "true", "yes")
                return self._send_json(self._api.save_chats(data, allow_empty=force))
            if path == "/api/projects":
                data = self._read_json()
                if not isinstance(data, list):
                    return self._send_error_json(400, "expected array")
                return self._send_json(self._api.save_projects(data))
            if path == "/api/teach":
                data = self._read_json() or {}
                question = data.get("question", "")
                if not question:
                    return self._send_error_json(400, "question is required")
                return self._send_json(self._api.teach(
                    question, data.get("config"),
                    data.get("use_classifier", True)))
            if path == "/api/generate_questions":
                data = self._read_json() or {}
                filename = data.get("filename", "")
                if not filename:
                    return self._send_error_json(400, "filename is required")
                return self._send_json(self._api.generate_questions(filename))
            self._send_error_json(404, "unknown api")

        def do_PUT(self):
            path = urlparse(self.path).path
            if path.startswith("/api/notes/"):
                fn = unquote(path[len("/api/notes/"):])
                data = self._read_json() or {}
                return self._send_json(self._api.update_note(
                    fn, data.get("title"), data.get("content"), data.get("remark")))
            self._send_error_json(405, "method not allowed")

        def do_DELETE(self):
            path = urlparse(self.path).path
            if path.startswith("/api/swdt/"):
                mid = unquote(path[len("/api/swdt/"):])
                return self._send_json(self._api.delete_mindmap(mid))
            # /api/notes/archive/<fn> = 取消归档（放回正常列表），不能当成删笔记
            if path.startswith("/api/notes/archive/"):
                fn = unquote(path[len("/api/notes/archive/"):])
                return self._send_json(self._api.unarchive_note(fn))
            if path.startswith("/api/notes/"):
                fn = unquote(path[len("/api/notes/"):])
                return self._send_json(self._api.delete_note(fn))
            if path.startswith("/api/chats/"):
                cid = unquote(path[len("/api/chats/"):])
                return self._send_json(self._api.delete_chat(cid))
            if path.startswith("/api/projects/"):
                pid = unquote(path[len("/api/projects/"):])
                return self._send_json(self._api.delete_project(pid))
            self._send_error_json(405, "method not allowed")

    return Handler


def start_server(api, html_dir, host="127.0.0.1", port=5000):
    """启动内置 HTTP 服务；端口被占用时向后顺延，返回 (server, port)"""
    while port < 5100:
        try:
            handler = make_handler(api, html_dir)
            server = ThreadingHTTPServer((host, port), handler)
            server.daemon_threads = True
            return server, port
        except OSError:
            port += 1
    raise RuntimeError("未找到可用端口（5000-5099 均被占用）")


# ============================================================
# 启动
# ============================================================
def _setup_dpi():
    """声明进程 DPI 感知：否则 Windows 会按系统缩放比例把整个界面位图放大，
    页面实际按 CSS 像素布局却被拉伸，出现字发虚、内容偏大甚至右侧被裁。"""
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def main():
    _setup_dpi()
    api = Api()

    # 后台预热 RAG（不阻塞 UI）
    def _warm():
        try:
            rag = api.ai.rag
            if rag:
                ok = rag.ensure_ready()
                print(f"🔥 RAG 预热{'成功' if ok else '失败: ' + rag.error}")
        except Exception as e:
            print(f"🔥 RAG 预热异常: {e}")

    threading.Thread(target=_warm, daemon=True).start()

    target = 'swdt.html'
    title = '思维地图'
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ('note', 'notes', '笔记'):
            target, title = 'note_fixed.html', '智能笔记'
        elif arg in ('chat', 'ai', '聊天'):
            target, title = 'ai_chat.html', 'AI 助手'
        elif arg in ('furnace', '炼炉', '出题'):
            target, title = 'knowledge_furnace.html', '知识炼炉'

    url = None
    server = None
    try:
        # 启动内置 HTTP 服务（ai_chat / knowledge_furnace / 降级 fetch 依赖 /api/*）
        server, port = start_server(api, HTML_DIR)
        threading.Thread(target=server.serve_forever, daemon=True).start()

        url = f"http://127.0.0.1:{port}/{target}"
        print(f'📂 数据库: {DB_PATH}')
        print(f'📂 HTML:   {HTML_DIR}')
        print(f'🌐 打开页面: {url}')

        # 窗口大小按屏幕自适应：保持 1400x900 的逻辑尺寸，按 DPI 换算成物理像素
        try:
            import ctypes as _ct
            _u = _ct.windll.user32
            _sw = _u.GetSystemMetrics(0)
            _sh = _u.GetSystemMetrics(1)
            try:
                _dpi = _u.GetDpiForSystem()
            except Exception:
                _dpi = 96
        except Exception:
            _sw, _sh, _dpi = 1400, 900, 96
        _scale = max(1.0, float(_dpi) / 96.0)
        _logic_w, _logic_h = _sw / _scale, _sh / _scale
        win_w = int(min(1400, max(760, _logic_w - 80)) * _scale)
        win_h = int(min(900, max(520, _logic_h - 100)) * _scale)
        win_x = max(0, int((_sw - win_w) / 2))
        win_y = max(0, int((_sh - win_h) / 2))
        min_size = (int(760 * _scale), int(520 * _scale))
        print(f'🖥️ 屏幕 {_sw}x{_sh} dpi={_dpi} scale={_scale:.2f} 窗口 {win_w}x{win_h}')

        webview.create_window(
            title=title,
            url=url,
            js_api=api,
            width=win_w,
            height=win_h,
            x=win_x,
            y=win_y,
            min_size=min_size,
            confirm_close=False,
            text_select=True,
        )
        webview.start()
    finally:
        if server is not None:
            server.shutdown()


if __name__ == '__main__':
    main()