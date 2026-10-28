import json
import os
import re
import time
import uuid

from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from config import Config
from embeddings import LocalEmbeddings

_PLACEHOLDER = "__init__"  # FAISS 需要至少一条数据才能建索引，占位用
_DEDUP_THRESHOLD = 0.4     # L2 距离阈值（BGE 已归一化），低于视为重复记忆


def _tokenize(text):
    """中文按字切分，英文/数字按词切分，用于 BM25 关键词检索。"""
    return re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text.lower())


class MemoryStore:
    """记忆存储：FAISS 向量索引 + BM25 关键词索引 + 结构化 JSON。"""

    def __init__(self):
        self.embeddings = LocalEmbeddings()
        self.index_path = os.path.join(Config.MEMORY_DIR, "faiss_index")
        self.memories_path = os.path.join(Config.MEMORY_DIR, "memories.json")
        self.available = True
        self.memories = self._load_memories()
        self.vector_store = self._load_or_create_index()
        self._bm25 = self._build_bm25()

    # ── 加载 / 保存 ─────────────────────────────────────────
    def _load_memories(self):
        if os.path.exists(self.memories_path):
            with open(self.memories_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_memories(self):
        with open(self.memories_path, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def _load_or_create_index(self):
        try:
            if os.path.exists(self.index_path):
                return FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            return FAISS.from_texts([_PLACEHOLDER], self.embeddings)
        except Exception as e:
            self.available = False
            print(f"[MemoryStore] embedding 不可用，记忆功能已降级：{e}")
            return None

    def _persist(self):
        os.makedirs(Config.MEMORY_DIR, exist_ok=True)
        self.vector_store.save_local(self.index_path)
        self._save_memories()

    def _build_bm25(self):
        corpus = [_tokenize(m["content"]) for m in self.memories]
        return BM25Okapi(corpus) if corpus else None

    # ── 写入（带去重）────────────────────────────────────────
    def add_memory(self, content):
        if not self.available:
            return None
        content = content.strip()
        if not content:
            return None
        # 去重：与最相似的现有记忆对比，低于阈值视为重复
        if self.memories:
            try:
                hits = self.vector_store.similarity_search_with_score(content, k=1)
                if hits:
                    doc, score = hits[0]
                    if doc.page_content != _PLACEHOLDER and score < _DEDUP_THRESHOLD:
                        return None
            except Exception as e:
                print(f"[MemoryStore] 去重检测失败：{e}")

        memory_id = uuid.uuid4().hex[:12]
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        meta = {"memory_id": memory_id, "created_at": created_at}
        self.vector_store.add_texts([content], metadatas=[meta])
        self.memories.append(
            {"id": memory_id, "content": content, "created_at": created_at}
        )
        self._persist()
        self._bm25 = self._build_bm25()
        return memory_id

    # ── 检索（混合：BM25 + 向量 + RRF 融合）──────────────────
    def search(self, query, top_k=None):
        if not self.available or not self.memories:
            return []
        top_k = top_k or Config.TOP_K_MEMORIES
        recall_k = top_k * 2

        rrf = {}  # memory_id -> RRF 分数
        meta_by_id = {m["id"]: m for m in self.memories}

        # 1. 向量召回（语义）
        try:
            vector_hits = self.vector_store.similarity_search_with_score(query, k=recall_k)
            for rank, (doc, score) in enumerate(vector_hits):
                if doc.page_content == _PLACEHOLDER:
                    continue
                mid = doc.metadata.get("memory_id", doc.page_content)
                rrf[mid] = rrf.get(mid, 0) + 1.0 / (60 + rank)
        except Exception as e:
            print(f"[MemoryStore] 向量检索失败：{e}")

        # 2. BM25 召回（关键词）
        if self._bm25 is not None:
            scores = self._bm25.get_scores(_tokenize(query))
            ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:recall_k]
            for rank, (idx, score) in enumerate(ranked):
                if score <= 0:
                    continue
                mid = self.memories[idx]["id"]
                rrf[mid] = rrf.get(mid, 0) + 1.0 / (60 + rank)

        # 3. 按 RRF 分数排序取 top-k
        ranked_ids = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for mid, _ in ranked_ids:
            m = meta_by_id.get(mid)
            if m:
                results.append({"content": m["content"], "created_at": m["created_at"]})
        return results

    def get_all_memories(self):
        return self.memories

    def clear(self):
        self.memories = []
        self._bm25 = None
        if not self.available:
            return
        self.vector_store = FAISS.from_texts([_PLACEHOLDER], self.embeddings)
        self._persist()
