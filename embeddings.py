import os

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from config import Config


class LocalEmbeddings(Embeddings):
    """本地 BGE 中文 embedding（fastembed + onnxruntime），免费离线、永不欠费。"""

    def __init__(self):
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        self._model = TextEmbedding(
            model_name=Config.EMBEDDING_MODEL,
            cache_dir=Config.MODEL_CACHE_DIR,
        )

    def embed_documents(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]
