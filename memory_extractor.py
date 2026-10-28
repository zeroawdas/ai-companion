import json
import re

from llm_client import LLMClient

_EXTRACT_SYSTEM = """你是记忆提取器。请从对话中提取出关于「用户」的、值得长期记住的信息。

只提取这四类：
1. 用户的个人事实（年龄、职业、城市、身体状况等）
2. 用户的偏好与喜恶（喜欢/讨厌的食物、爱好、习惯等）
3. 用户经历的重要事件（考试、面试、旅行、生病等）
4. 用户与 AI 之间的关系信息（称呼、纪念日、约定等）

不要提取：
- 闲聊、寒暄、客套话
- AI 自己的话（只提取用户的信息）
- 临时情绪或一次性话题
- 假设、玩笑、反话

如果没有值得记住的信息，返回空数组 []。

只输出一个 JSON 数组，每个元素是一句以「用户」为主语的简洁事实陈述，不要输出任何其他文字。
示例格式：["用户是一名大四学生", "用户喜欢喝美式咖啡"]"""


class MemoryExtractor:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract(self, user_msg, assistant_msg):
        conversation = f"用户：{user_msg}\nAI：{assistant_msg}"
        raw = self.llm.invoke(_EXTRACT_SYSTEM, conversation)
        return self._parse(raw)

    def _parse(self, raw):
        if not raw:
            return []
        raw = raw.strip()
        # 去掉可能的 markdown 代码块包裹
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        # 优先直接解析
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [s.strip() for s in data if isinstance(s, str) and s.strip()]
        except Exception:
            pass

        # 降级：提取第一个 [...] 数组
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    return [s.strip() for s in data if isinstance(s, str) and s.strip()]
            except Exception:
                pass
        return []
