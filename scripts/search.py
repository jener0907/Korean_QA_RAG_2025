import sys
from src.utils.config import get_config
from src.indexer.bm25 import BM25Indexer
from src.indexer.vector import VectorIndexer
from src.search.hybrid import HybridSearcher
from src.rerank.reranker import Reranker

# 검색 테스트 스크립트

def main():
    if len(sys.argv) < 2:
        print("질문을 입력하세요")
        return
    query = " ".join(sys.argv[1:])
    cfg = get_config()
    bm25 = BM25Indexer("index/bm25")
    vec = VectorIndexer(cfg.use_qdrant, cfg.qdrant_url, cfg.use_faiss)
    searcher = HybridSearcher(bm25, vec, cfg.alpha_bm25, cfg.beta_vector)
    hits = searcher.search(query, cfg.topn)
    rerank = Reranker(cfg.use_rerank)
    if cfg.use_rerank:
        hits = rerank.rerank(query, hits, cfg.topk)
    else:
        hits = hits[: cfg.topk]
    for h in hits:
        print(h.get("rule_id", ""), h.get("score", 0), h.get("body", "")[:80])

if __name__ == "__main__":
    main()
