from typing import List, Dict

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# 리랭커 (선택적)

class Reranker:
    def __init__(self, use_rerank: bool):
        self.use_rerank = use_rerank and CrossEncoder is not None
        if self.use_rerank:
            try:
                self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception:
                self.use_rerank = False

    def rerank(self, query: str, docs: List[Dict], topk: int = 5) -> List[Dict]:
        if not self.use_rerank or not docs:
            return docs[:topk]
        pairs = [[query, d["body"]] for d in docs]
        scores = self.model.predict(pairs)
        for d, s in zip(docs, scores):
            d["score"] = float(s)
        ranked = sorted(docs, key=lambda x: x["score"], reverse=True)[:topk]
        return ranked
