from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import Config


class LLMClient:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_API_BASE,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            streaming=True,
        )

    def stream_chat(self, system_prompt, recent_messages):
        """流式生成回复。

        recent_messages: list[dict]，元素形如 {"role": "user"/"assistant", "content": str}，
        按时间顺序排列，最后一条应为用户当前消息。
        """
        messages = [SystemMessage(content=system_prompt)]
        for m in recent_messages:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                messages.append(AIMessage(content=m["content"]))
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    def invoke(self, system_prompt, user_content):
        """非流式单次调用，用于记忆提取等结构化任务。"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        return self.llm.invoke(messages).content
