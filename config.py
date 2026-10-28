import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LLM（DeepSeek）──────────────────────────────────────
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

    # ── Embedding（阿里云千问）───────────────────────────────
    QIANWEN_API_KEY = os.getenv("QIANWEN_API_KEY")
    QIANWEN_API_BASE = os.getenv("QIANWEN_API_BASE", "https://dashscope.aliyuncs.com/api/v1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".ai-companion", "models"
    )

    # ── 联网搜索（博查 Bocha，国内直连）────────────────────────
    BOCHA_API_KEY = os.getenv("BOCHA_API_KEY")

    # ── 记忆参数 ────────────────────────────────────────────
    TOP_K_MEMORIES = int(os.getenv("TOP_K_MEMORIES", "5"))   # 检索注入的记忆条数
    RECENT_TURNS = int(os.getenv("RECENT_TURNS", "12"))      # 近期窗口保留的轮数
    # 记忆库目录：默认放用户主目录（英文路径），因为 FAISS 的 C++ 底层不支持中文路径写文件。
    MEMORY_DIR = os.getenv("MEMORY_DIR") or os.path.join(
        os.path.expanduser("~"), ".ai-companion", "memory_data"
    )

    @classmethod
    def validate(cls):
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY 未设置，请检查 .env 文件")
        return True
