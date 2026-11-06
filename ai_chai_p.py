import os
import re
import json
import pickle
import hashlib
import html as html_lib
import threading
import requests
from typing import List, Dict, Optional

# ---------- RAG 可选依赖：缺失时自动降级为纯聊天 ----------
# 说明：llama_index / chromadb / modelscope 这些重型库导入一次要 10 秒左右，
# 放在模块顶层会让每次启动都白等这么久；改为首次真正用到 RAG 时才加载
# （应用启动时由后台线程预热调用，不阻塞窗口显示）。
RAG_AVAILABLE = False
RAG_DEPS_LOADED = False
RAG_IMPORT_ERROR = ""
_rag_import_lock = threading.Lock()


def _load_rag_deps() -> bool:
    """延迟加载 RAG 重型依赖，成功返回 True。线程安全，结果缓存。"""
    global RAG_AVAILABLE, RAG_DEPS_LOADED, RAG_IMPORT_ERROR
    if RAG_DEPS_LOADED:
        return RAG_AVAILABLE
    with _rag_import_lock:
        if RAG_DEPS_LOADED:
            return RAG_AVAILABLE
        try:
            from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
            from llama_index.vector_stores.chroma import ChromaVectorStore
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.retrievers.bm25 import BM25Retriever
            from llama_index.core.node_parser import SentenceSplitter
            from modelscope import snapshot_download
            import chromadb
            globals().update({
                "VectorStoreIndex": VectorStoreIndex,
                "StorageContext": StorageContext,
                "SimpleDirectoryReader": SimpleDirectoryReader,
                "ChromaVectorStore": ChromaVectorStore,
                "HuggingFaceEmbedding": HuggingFaceEmbedding,
                "BM25Retriever": BM25Retriever,
                "SentenceSplitter": SentenceSplitter,
                "snapshot_download": snapshot_download,
                "chromadb": chromadb,
            })
            RAG_AVAILABLE = True
            RAG_IMPORT_ERROR = ""
        except Exception as _e:
            RAG_AVAILABLE = False
            RAG_IMPORT_ERROR = str(_e)
        finally:
            RAG_DEPS_LOADED = True
    return RAG_AVAILABLE


# ============================================================
# 提示词（保留 mota.py 原文）
# ============================================================
BRANCH_A_PROMPT = """用户问题：{question}
相关笔记片段：
{context}
请严格按以下格式输出：
1. 核心结论：[精准回答用户的问题]
2. 原文依据：[引用核心原文]
3. 知识延伸：[补充相关知识点]
4. 知识缺口：[指出没写透彻的地方]
5. 灵魂反问：[引发思考的问题]

（输出要求：方括号里只是写作要求，请直接写出你自己的内容，绝对不要原样输出方括号或其中“精准回答用户的问题”“引用核心原文”这类模板文字。）"""

BRANCH_B_PROMPT = """用户问题：{question}
相关笔记片段：
{context}
笔记只能部分回答该问题。请严格按以下格式输出：
1. 笔记中已有的信息：[引用相关内容，必须明确告诉用户哪些文件里有答案]
2. 笔记缺什么细节：[指出缺了什么]
3. 补全方向：[去搜什么]
4. 一个通俗类比：[生活化解释]
5. 深入引导：[引导问题]

（输出要求：方括号里只是写作要求，请直接写出你自己的内容，绝对不要原样输出方括号或其中“指出缺了什么”“去搜什么”这类模板文字。）"""

BRANCH_C_PROMPT = """用户问题：{question}
用户的笔记中完全没有这个知识点。请直接输出4行学习路径：
搜索：[具体搜什么]
记录：[记录什么]
实践：[去哪练]
复盘：[怎么复盘]

（输出要求：方括号里只是写作要求，请直接写出你自己的内容，绝对不要原样输出方括号或其中“具体搜什么”“记录什么”这类模板文字，也不要额外增加解释段落。）"""

CLASSIFY_PROMPT = """你是一个严格的知识库分类助手。
判断【笔记片段】能否回答【用户问题】：
- A：能完整回答
- B：只能部分回答
- C：完全不能回答
只输出 A、B 或 C。"""


# ============================================================
# 1. LLM 客户端（合并 ai.py 的 call_ai / 流式 / pandn）
# ============================================================
class LLMClient:
    """统一 LLM 调用层。返回字符串；出错抛异常，由上层决定降级。"""

    def __init__(self, api_key: str, platform: str = "guiji",
                 model: str = None, timeout: int = 120):
        self.api_key = (api_key or "").strip()
        self.platform = platform
        self.timeout = timeout
        if platform == "guiji":
            self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
            self.model = model or "Qwen/Qwen2.5-7B-Instruct"
        elif platform == "zhupu":
            self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            self.model = model or "glm-4-flash"
        else:
            raise ValueError("platform 只能是 guiji 或 zhupu")
        if not self.api_key:
            raise ValueError("API Key 为空")

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages, temperature, max_tokens, stream):
        p = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        # Qwen3 系列关闭 thinking，避免短输出被思考吃光
        if "qwen3" in (self.model or "").lower():
            p["enable_thinking"] = False
        return p

    def chat(self, messages, temperature=0.01, max_tokens=2048,
             stream=False) -> str:
        if stream:
            return "".join(self._chat_stream(messages, temperature, max_tokens))

        payload = self._payload(messages, temperature, max_tokens, False)
        try:
            r = requests.post(self.base_url, headers=self._headers(),
                              data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                              timeout=self.timeout)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                j = r.json()
                detail = (j.get("error") or {}).get("message") or str(j.get("error")) or ""
            except Exception:
                detail = (r.text or "")[:200]
            if not detail:
                detail = e
            raise RuntimeError(f"HTTP {r.status_code}: {detail}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败: {e}") from e
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"返回无 choices: {data}")
        return ((choices[0].get("message") or {}).get("content") or "").strip()

    def _chat_stream(self, messages, temperature, max_tokens):
        payload = self._payload(messages, temperature, max_tokens, True)
        with requests.post(self.base_url, headers=self._headers(),
                           json=payload, stream=True,
                           timeout=self.timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    yield content

    def yes_no(self, question: str,
               system: str = "只回答「是」或「否」，不要解释。") -> bool:
        """让模型只回答是/否（修正 ai.py 的 '不是' 误判）"""
        try:
            raw = self.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": question}],
                temperature=0.1, max_tokens=16,
            )
        except Exception:
            return False
        s = re.sub(r"[\s。.!！?？]", "", raw or "")
        if not s:
            return False
        if s.startswith("不是") or s.startswith("否"):
            return False
        return s.startswith("是")


# ============================================================
# 2. RAG 引擎（合并 mota.py，懒加载 + 线程安全 + 可降级）
# ============================================================
class RAGEngine:
    def __init__(self, notes_dir: str = "bij",
                 chroma_path: str = "./chroma_notes_db",
                 cache_file: str = "./nodes_bm25_cache.pkl",
                 models_dir: str = "./models",
                 embed_model_name: str = "BAAI/bge-small-zh-v1.5",
                 notes_provider=None):
        if not _load_rag_deps():
            raise RuntimeError(f"RAG 依赖未安装: {RAG_IMPORT_ERROR}")
        self.notes_dir = notes_dir
        self.chroma_path = chroma_path
        self.cache_file = cache_file
        self.models_dir = models_dir
        self.embed_model_name = embed_model_name
        # 知识来源：优先用 notes_provider（应用内『笔记』页面的笔记库），
        # 没有提供时才回退到 bij 目录
        self.notes_provider = notes_provider
        self._fingerprint = ""
        self._fp_file = cache_file + ".fingerprint"

        self._lock = threading.Lock()
        self._index = None
        self._bm25 = None
        self._ready = False
        self._failed = False
        self._error = None

    # ---------- 语料来源：应用内笔记 ----------
    @staticmethod
    def _strip_html(html_text: str) -> str:
        if not html_text:
            return ""
        s = re.sub(r"<script.*?</script>", " ", html_text, flags=re.S | re.I)
        s = re.sub(r"<style.*?</style>", " ", s, flags=re.S | re.I)
        s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
        s = re.sub(r"</(p|div|li|h\d|tr)>", "\n", s, flags=re.I)
        s = re.sub(r"<[^>]+>", "", s)
        s = html_lib.unescape(s)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{2,}", "\n", s)
        return s.strip()

    def _note_to_text(self, note: Dict) -> str:
        title = (note.get("title") or "无标题").strip()
        body = self._strip_html(note.get("content") or "")
        if not body:
            return ""
        return f"《{title}》\n{body}"

    def _load_documents(self):
        from llama_index.core import Document
        docs = []
        if self.notes_provider:
            try:
                notes = self.notes_provider() or []
            except Exception as e:
                raise RuntimeError(f"读取笔记库失败: {e}")
            for n in notes:
                text = self._note_to_text(n)
                if len(text) < 20:
                    continue
                docs.append(Document(
                    text=text,
                    metadata={"file_name": (n.get("title") or "无标题")},
                ))
            return docs
        # 兜底：目录模式
        if not os.path.isdir(self.notes_dir):
            raise FileNotFoundError(f"笔记目录不存在: {self.notes_dir}")
        return SimpleDirectoryReader(self.notes_dir, recursive=True).load_data()

    def _compute_fingerprint(self) -> str:
        """笔记内容指纹：任何新增/修改/删除都会让指纹变化"""
        if self.notes_provider:
            try:
                notes = self.notes_provider() or []
            except Exception:
                notes = []
            raw = "||".join(
                "%s::%s::%s" % (n.get("filename"), n.get("updated_at"), n.get("content") or "")
                for n in notes
            )
        else:
            parts = []
            if os.path.isdir(self.notes_dir):
                for root, _dirs, files in os.walk(self.notes_dir):
                    for fn in sorted(files):
                        p = os.path.join(root, fn)
                        try:
                            st = os.stat(p)
                            parts.append(f"{p}:{st.st_size}:{st.st_mtime}")
                        except OSError:
                            pass
            raw = "|".join(sorted(parts))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _read_saved_fingerprint(self) -> str:
        try:
            with open(self._fp_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""

    def _save_fingerprint(self, fp: str):
        try:
            with open(self._fp_file, "w", encoding="utf-8") as f:
                f.write(fp)
        except Exception as e:
            print(f"⚠️ 保存索引指纹失败: {e}")

    def invalidate(self):
        """笔记增删改后调用：清掉就绪标记，下次问答自动重建索引"""
        with self._lock:
            self._ready = False
            self._failed = False
            self._error = None
            print("🔄 笔记有更新，RAG 索引将在下次问答时重建")

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str:
        return self._error or ""

    def ensure_ready(self) -> bool:
        """首次调用构建/加载索引；笔记变化后自动重建"""
        if self._ready:
            # 每次问答前轻量校验笔记指纹，变了就重建（数据量小，成本可忽略）
            try:
                if self._compute_fingerprint() != self._fingerprint:
                    print("🔄 检测到笔记有变化，重建 RAG 索引…")
                    self._ready = False
            except Exception:
                pass
            if self._ready:
                return True
        if self._failed:
            return False
        with self._lock:
            if self._ready:
                return True
            if self._failed:
                return False
            try:
                self._build()
                self._ready = True
                self._failed = False
                self._error = None
                print("✅ RAG 索引就绪（知识来源：应用内笔记）")
                return True
            except Exception as e:
                self._failed = True
                self._error = str(e)
                print(f"❌ RAG 初始化失败（后续请求将跳过 RAG 直接问答）: {e}")
                return False

    def _build(self):
        os.makedirs(self.models_dir, exist_ok=True)
        model_dir = snapshot_download(self.embed_model_name,
                                      cache_dir=self.models_dir)
        embed_model = HuggingFaceEmbedding(model_name=model_dir)

        chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        collection = chroma_client.get_or_create_collection("notes")
        vector_store = ChromaVectorStore(chroma_collection=collection)

        current_fp = self._compute_fingerprint()
        saved_fp = self._read_saved_fingerprint()
        need_build = (collection.count() == 0
                      or not os.path.exists(self.cache_file)
                      or saved_fp != current_fp)

        if need_build:
            documents = self._load_documents()
            if not documents:
                raise ValueError("笔记库为空：请先在『笔记』页面写几篇笔记，AI 才有知识来源")
            # 清掉旧语料，避免已删除或改名的笔记残留在索引里
            try:
                chroma_client.delete_collection("notes")
            except Exception:
                pass
            collection = chroma_client.get_or_create_collection("notes")
            vector_store = ChromaVectorStore(chroma_collection=collection)

            splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
            nodes = splitter.get_nodes_from_documents(documents)
            storage = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex(
                nodes=nodes, storage_context=storage, embed_model=embed_model
            )
            with open(self.cache_file, "wb") as f:
                pickle.dump(nodes, f)
            self._fingerprint = current_fp
            self._save_fingerprint(current_fp)
            print(f"📚 用笔记重建索引：{len(documents)} 篇笔记 / {len(nodes)} 个切片")
        else:
            self._index = VectorStoreIndex.from_vector_store(
                vector_store, embed_model=embed_model
            )
            with open(self.cache_file, "rb") as f:
                nodes = pickle.load(f)
            self._fingerprint = current_fp
            print(f"♻️ 加载已有索引：{len(nodes)} 个切片")

        # 中文分词（jieba 可选）
        try:
            import jieba
            tokenizer = lambda t: list(jieba.cut(t))
        except ImportError:
            tokenizer = None
            print("⚠️ 未安装 jieba，BM25 中文召回会退化（pip install jieba）")
        kwargs = dict(nodes=nodes, similarity_top_k=15)
        if tokenizer:
            kwargs["tokenizer"] = tokenizer
        self._bm25 = BM25Retriever.from_defaults(**kwargs)

    def retrieve(self, question: str, llm: LLMClient, top_k: int = 5) -> Dict:
        """
        返回 {'context', 'sources', 'entity', 'hit', 'error'}
        """
        if not self.ensure_ready():
            return {"context": "", "sources": [], "entity": "",
                    "hit": False, "error": self._error}

        entity = self._extract_entity(question, llm)
        vector_nodes = self._index.as_retriever(
            similarity_top_k=15).retrieve(question)
        bm25_nodes = self._bm25.retrieve(question)

        filtered, fallback, seen = [], [], set()
        for node in list(vector_nodes) + list(bm25_nodes):
            text = (node.text or "").strip()
            if len(text) < 20 or text in seen:
                continue
            seen.add(text)
            fname = node.metadata.get("file_name", "未知")
            block = f"[来源: {fname}]\n{text[:500]}"
            fallback.append(block)
            if entity:
                el = entity.lower()
                if el in text.lower() or el in fname.lower():
                    filtered.append(block)
            else:
                filtered.append(block)

        final = filtered[:top_k] if filtered else fallback[:top_k]
        sources = []
        for block in final:
            m = re.match(r"\[来源:\s*(.+?)\]", block)
            if m and m.group(1) not in sources:
                sources.append(m.group(1))

        return {
            "context": "\n\n".join(final),
            "sources": sources,
            "entity": entity,
            "hit": bool(filtered),
            "error": "",
        }

    def _extract_entity(self, question: str, llm: LLMClient) -> str:
        prompt = (
            "提取这句话中的核心技术名词或核心对象。"
            "要求：只输出提取到的那个词，不要带标点，不要解释，不要废话。\n"
            f"问题：{question}"
        )
        try:
            raw = llm.chat([{"role": "user", "content": prompt}], max_tokens=16)
        except Exception:
            return ""
        if not raw or "❌" in raw:
            return ""
        # 先长后短替换，避免 "是什么" 被 "什么" 先吃掉
        for ch in ["。", "，", ".", ",", "\n", " ", "“", "”",
                   "的变量", "的用法", "的简介", "是什么", "什么"]:
            raw = raw.replace(ch, "")
        return raw.strip()


# ============================================================
# 3. 门面：AIService（对外唯一入口）
# ============================================================
class AIService:
    # pywebview 注入 JS 桥时会递归暴露 js_api 的属性；标记该实例不序列化，
    # 避免遍历到 llm/rag 等 property 触发异常刷屏
    _serializable = False

    def __init__(self, api_key: str = None, platform: str = "guiji",
                 model: str = None, notes_dir: str = "bij",
                 enable_rag: bool = True, rag_kwargs: dict = None,
                 notes_provider=None):
        self.platform = platform
        self.api_key = (api_key or os.environ.get("SILICONFLOW_API_KEY", "")).strip()
        self.model = model
        self._notes_dir = notes_dir
        self._notes_provider = notes_provider
        self._enable_rag = enable_rag
        self._rag_kwargs = rag_kwargs or {}

        self._llm = None
        self._rag = None
        self._history: List[Dict] = []
        self._lock = threading.RLock()

    # ---------- 配置 ----------
    def set_api_key(self, key: str):
        with self._lock:
            self.api_key = (key or "").strip()
            self._llm = None

    def health(self) -> Dict:
        return {
            "has_key": bool(self.api_key),
            "platform": self.platform,
            "model": self.model or "(默认)",
            "rag_enabled": self._enable_rag,
            "rag_available": RAG_AVAILABLE,
            "rag_ready": bool(self._rag and self._rag.ready),
            "rag_error": (self._rag.error if self._rag else RAG_IMPORT_ERROR),
        }

    # ---------- 懒加载 ----------
    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            with self._lock:
                if self._llm is None:
                    if not self.api_key:
                        raise RuntimeError(
                            "API Key 未设置：请调用 set_api_key 或设置环境变量 "
                            "SILICONFLOW_API_KEY"
                        )
                    self._llm = LLMClient(self.api_key, self.platform, self.model)
        return self._llm

    @property
    def rag(self) -> Optional[RAGEngine]:
        if not self._enable_rag:
            return None
        if self._rag is None:
            with self._lock:
                if self._rag is None:
                    try:
                        self._rag = RAGEngine(
                            notes_dir=self._notes_dir,
                            notes_provider=self._notes_provider,
                            **self._rag_kwargs
                        )
                    except Exception as e:
                        print(f"⚠️ RAG 初始化异常（降级纯聊天）: {e}")
                        self._enable_rag = False
                        return None
        return self._rag

    def invalidate_rag(self):
        """笔记有新增/修改/删除时调用，让索引在下次问答时重建"""
        try:
            if self._rag is not None:
                self._rag.invalidate()
        except Exception:
            pass

    # ---------- 纯聊天（带会话历史） ----------
    def chat(self, text: str, max_tokens: int = 2048) -> str:
        with self._lock:
            self._history.append({"role": "user", "content": text})
            messages = [{"role": "system",
                         "content": "你是一个有用的 AI 助手。"}] + self._history
        try:
            reply = self.llm.chat(messages, max_tokens=max_tokens)
            with self._lock:
                self._history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            with self._lock:
                if self._history and self._history[-1]["role"] == "user":
                    self._history.pop()
            return f"❌ 请求出错: {e}"

    def clear_history(self):
        with self._lock:
            self._history = []

    # ---------- RAG 问答 ----------
    def _resolve_llm(self, config: dict = None) -> LLMClient:
        """按请求配置解析 LLM；无 config 或 config 无 key 时回退默认"""
        if not config:
            return self.llm
        api_key = (config.get("api_key") or "").strip()
        platform = config.get("platform") or self.platform
        model = config.get("model") or None
        if not api_key:
            return self.llm
        return LLMClient(api_key, platform, model, timeout=120)

    def teach(self, question: str, use_classifier: bool = True,
              config: Optional[dict] = None) -> Dict:
        """
        返回 {'answer', 'branch', 'sources', 'entity', 'note'}
        branch: 'A' 完整回答 / 'B' 部分回答 / 'C' 学习路径
        config: 可选 {'platform', 'api_key', 'model', 'double_check'}，
                支持页面级配置动态切换平台/Key/模型
        """
        llm = None
        has_key = bool(self.api_key) or bool(config and (config.get("api_key") or "").strip())
        if not has_key:
            return {"answer": "❌ 未配置 API Key，请先在页面中填入硅基流动 Key",
                    "branch": "C", "sources": [], "entity": ""}
        llm = self._resolve_llm(config)

        rag = self.rag
        if rag is None:
            try:
                ans = llm.chat([{"role": "user", "content": question}])
            except Exception as e:
                ans = f"❌ {e}"
            return {"answer": ans, "branch": "B", "sources": [], "entity": "",
                    "note": "RAG 未启用，已降级为纯聊天"}

        result = rag.retrieve(question, llm)
        entity = result.get("entity", "")
        sources = result.get("sources", [])
        context = result.get("context", "")

        if result.get("error"):
            try:
                ans = llm.chat([{"role": "user", "content": question}])
            except Exception as e:
                ans = f"❌ {e}"
            return {"answer": ans, "branch": "B", "sources": [],
                    "entity": entity, "note": f"RAG 不可用：{result['error']}"}

        if not context:
            branch = "C"
            prompt = BRANCH_C_PROMPT.format(question=question)
        elif use_classifier:
            # 三档判断：A=笔记能完整回答 / B=只能部分回答 / C=笔记里没有
            branch = self._classify(question, context, llm=llm)
            if branch == "A":
                prompt = BRANCH_A_PROMPT.format(question=question, context=context)
            elif branch == "B":
                prompt = BRANCH_B_PROMPT.format(question=question, context=context)
            else:
                branch = "C"
                prompt = BRANCH_C_PROMPT.format(question=question)
        else:
            # 关闭分类器时：有 context 就直接走 B
            branch = "B"
            prompt = BRANCH_B_PROMPT.format(question=question, context=context)

        try:
            answer = llm.chat([{"role": "user", "content": prompt}])
        except Exception as e:
            answer = f"❌ 生成失败: {e}"

        # C 分支是学习路径，不是从笔记里抽出来的，来源清空避免误导
        if branch == "C":
            sources = []

        return {"answer": answer, "branch": branch,
                "sources": sources, "entity": entity}

    def _classify(self, question: str, context: str, llm: LLMClient = None) -> str:
        llm = llm or self.llm
        try:
            raw = llm.chat(
                [{"role": "user",
                  "content": f"{CLASSIFY_PROMPT}\n\n用户问题：{question}\n\n相关片段：\n{context}"}],
                max_tokens=8,
            )
        except Exception:
            return "B"
        txt = (raw or "").upper()
        m = re.search(r"[ABC]", txt)
        return m.group(0) if m else "B"