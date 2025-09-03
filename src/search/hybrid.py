from typing import List, Dict

# 하이브리드 검색기

class HybridSearcher:
    def __init__(self, bm25_searcher, vector_searcher, alpha: float = 0.6, beta: float = 0.4):
        self.bm25 = bm25_searcher
        self.vector = vector_searcher
        self.alpha = alpha
        self.beta = beta

    def search(self, query: str, topn: int = 20) -> List[Dict]:
        bm25_hits = self.bm25.search(query, topn)
        vec_hits = self.vector.search(query, topn)
        scores = {}
        texts = {}
        meta = {}
        for h in bm25_hits:
            scores[h["id"]] = scores.get(h["id"], 0) + self.alpha * h["score"]
            texts[h["id"]] = h.get("body", "")
            meta[h["id"]] = {"rule_id": h.get("rule_id", "")}
        for h in vec_hits:
            scores[h["id"]] = scores.get(h["id"], 0) + self.beta * h["score"]
            texts[h["id"]] = h.get("body", "")
            meta[h["id"]] = {"rule_id": h.get("rule_id", "")}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
        return [{"id": i, "score": s, "body": texts.get(i, ""), "rule_id": meta.get(i, {}).get("rule_id", "")} for i, s in ranked]
