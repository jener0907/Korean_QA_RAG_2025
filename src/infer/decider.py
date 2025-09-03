import json
from typing import List
from pathlib import Path

from src.utils.config import get_config
from src.indexer.bm25 import BM25Indexer
from src.indexer.vector import VectorIndexer
from src.search.hybrid import HybridSearcher
from src.rerank.reranker import Reranker
from src.infer import gpt_client, ollama_client

# 최종 판단자

class Decider:
    def __init__(self):
        self.config = get_config()
        # 예외 규칙 로딩
        exc_path = Path("rules/exception_rules.json")
        if exc_path.exists():
            self.exceptions = json.loads(exc_path.read_text(encoding="utf-8"))
        else:
            self.exceptions = []
        # 검색기 준비
        self.bm25 = BM25Indexer("index/bm25")
        self.vector = VectorIndexer(self.config.use_qdrant, self.config.qdrant_url, self.config.use_faiss)
        self.searcher = HybridSearcher(self.bm25, self.vector, self.config.alpha_bm25, self.config.beta_vector)
        self.reranker = Reranker(self.config.use_rerank)

    def _check_exception(self, query: str):
        for rule in self.exceptions:
            if rule["pattern"] in query:
                return rule["answer"]
        return None

    def judge(self, query: str) -> str:
        hit = self._check_exception(query)
        if hit:
            return hit
        hits = self.searcher.search(query, self.config.topn)
        if self.config.use_rerank:
            hits = self.reranker.rerank(query, hits, self.config.topk)
        else:
            hits = hits[: self.config.topk]
        context = [h["body"] for h in hits]
        rule_hints = ", ".join(h.get("rule_id", "") for h in hits if h.get("rule_id"))
        if self.config.openai_api_key:
            answer = gpt_client.generate_one_line_answer(query, context, rule_hints, self.config)
        else:
            answer = ollama_client.generate_one_line_answer(query, context, rule_hints, self.config)
        return answer.replace("\n", "")
