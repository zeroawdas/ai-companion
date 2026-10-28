import os

from config import Config

_UPDATE_SYSTEM = (
    "你是关系状态维护助手。请根据当前的「关系状态摘要」和本轮对话，更新一段关于用户与AI伴侣关系的简短摘要（不超过80字）。"
    "重点记录：用户近期的状态/情绪、正在关心或忙碌的事情、关系中的重要进展或约定。"
    "只输出更新后的摘要本身，不要任何解释。"
)


class RelationshipState:
    """关系状态摘要：维护一段常驻注入的简短文字，承载关系温度与用户近期状态。"""

    def __init__(self, llm):
        self.llm = llm
        self.state_path = os.path.join(Config.MEMORY_DIR, "relationship_state.txt")
        self.state = self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def _save(self):
        os.makedirs(Config.MEMORY_DIR, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            f.write(self.state)

    def get(self):
        return self.state

    def update(self, user_msg, assistant_msg):
        """基于本轮对话增量更新关系状态摘要（失败不阻塞主流程）。"""
        user_content = (
            f"当前关系状态摘要：{self.state or '（无）'}\n\n"
            f"本轮对话：\n用户：{user_msg}\nAI：{assistant_msg}"
        )
        try:
            new_state = self.llm.invoke(_UPDATE_SYSTEM, user_content)
            self.state = new_state.strip()
            self._save()
        except Exception as e:
            print(f"[RelationshipState] 更新失败：{e}")
