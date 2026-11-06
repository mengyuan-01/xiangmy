import json
import requests


class AI_shed:
    def __init__(self):
        self.API_KEY = ""
        self.ai_moxing = ""

    def shuru_key(self):
        """输入 API Key 和模型"""
        self.API_KEY = input("请输入你的 API Key: ").strip()
        self.ai_moxing = input("请输入模型（直接回车用默认）: ").strip()

    def ai_peizhi(self):
        self.headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }
        self.payload = {
            "model": "",
            "messages": [],
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "max_tokens": 2048,
            "frequency_penalty": 0,
            "stream": False
        }
        self.timeout = 60

    def guiji_ai(self):
        self.BASE_URL = "https://api.siliconflow.cn/v1"
        self.CHAT_URL = self.BASE_URL + "/chat/completions"
        self.MODELS_URL = self.BASE_URL + "/models"
        self.ai_peizhi()
        self.payload["model"] = self.ai_moxing or "Qwen/Qwen3-32B"
        self.payload["max_tokens"] = 4096

    def zhupu_ai(self):
        self.BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
        self.CHAT_URL = self.BASE_URL + "/chat/completions"
        self.MODELS_URL = self.BASE_URL + "/models"
        self.ai_peizhi()
        self.payload["model"] = self.ai_moxing or "glm-4.7-flash"  # 免费模型
        self.payload["temperature"] = 0.95
        self.payload["top_p"] = 0.7
        self.payload["max_tokens"] = 4096


class ai_sheding(AI_shed):
    def __init__(self):
        super().__init__()
        self.lishi = []

    def _xuan_platform(self, plat):
        """根据平台名加载配置，返回 True/False"""
        if plat == "guiji":
            self.guiji_ai()
            return True
        elif plat == "zhupu":
            self.zhupu_ai()
            return True
        return False

    def liao_ai(self, wenben, plat="guiji"):
        """普通聊天（非流式）"""
        if not self._xuan_platform(plat):
            return "平台只能填 guiji 或 zhupu"

        self.lishi.append({"role": "user", "content": wenben})
        self.payload["messages"] = self.lishi

        try:
            r = requests.post(self.CHAT_URL, headers=self.headers,
                              json=self.payload, timeout=self.timeout)
            r.raise_for_status()
            huifu = r.json()["choices"][0]["message"]["content"]
            self.lishi.append({"role": "assistant", "content": huifu})
            return huifu
        except requests.exceptions.RequestException as e:
            self.lishi.pop()
            return f"请求出错：{e}"
        except (KeyError, IndexError):
            self.lishi.pop()
            return f"返回格式异常：{r.text}"

    def liao_ai_stream(self, wenben, plat="zhupu"):
        """流式聊天，逐字打印，返回完整回复"""
        if not self._xuan_platform(plat):
            print("平台只能填 guiji 或 zhupu")
            return ""

        self.lishi.append({"role": "user", "content": wenben})
        self.payload["messages"] = self.lishi
        self.payload["stream"] = True

        reply = ""
        try:
            with requests.post(self.CHAT_URL, headers=self.headers,
                               json=self.payload, stream=True,
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
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content")
                    if content:
                        print(content, end="", flush=True)
                        reply += content
            print()
            self.lishi.append({"role": "assistant", "content": reply})
        except requests.exceptions.RequestException as e:
            print(f"\n请求出错：{e}")
            self.lishi.pop()
        except (KeyError, IndexError):
            print(f"\n返回格式异常")
            self.lishi.pop()

        return reply

    def pandn_ai(self, wenti, plat="guiji"):
        """让AI只回答 是 / 否"""
        if not self._xuan_platform(plat):
            return False

        self.payload["temperature"] = 0.1
        self.payload["max_tokens"] = 10
        self.payload["stream"] = False
        self.payload["messages"] = [
            {"role": "system", "content": "你是一个判断助手，只回答「是」或「否」，不要解释。"},
            {"role": "user", "content": wenti}
        ]

        try:
            r = requests.post(self.CHAT_URL, headers=self.headers,
                              json=self.payload, timeout=self.timeout)
            r.raise_for_status()
            jieguo = r.json()["choices"][0]["message"]["content"].strip()
            return "是" in jieguo
        except requests.exceptions.RequestException as e:
            print(f"请求出错：{e}")
            return False
        except (KeyError, IndexError):
            print(f"返回格式异常")
            return False


if __name__ == "__main__":
    ai = ai_sheding()

    # 1. 先选平台
    pingtai = input("选择平台（guiji / zhupu，默认zhupu）：").strip() or "zhupu"

    # 2. 再输入 API Key 和模型
    ai.shuru_key()

    # 3. 智谱默认走流式
    liushi = pingtai == "zhupu"

    print(f"\n开始聊天，当前平台：{pingtai}，流式：{'开' if liushi else '关'}")
    print("exit/quit/退出 结束，clear 清空历史\n")

    while True:
        try:
            yonghu = input("你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出聊天")
            break

        if not yonghu:
            continue

        if yonghu.lower() in ("exit", "quit", "退出"):
            print("已退出聊天")
            break

        if yonghu.lower() == "clear":
            ai.lishi = []
            print("（已清空对话历史）\n")
            continue

        if liushi:
            print("AI：", end="", flush=True)
            ai.liao_ai_stream(yonghu, plat=pingtai)
            print()
        else:
            huifu = ai.liao_ai(yonghu, plat=pingtai)
            print(f"AI：{huifu}\n")