import os
from typing import List, Dict

try:
    from whoosh import index
    from whoosh.fields import Schema, TEXT, ID
    from whoosh.qparser import QueryParser
except Exception:
    index = None

# BM25 인덱서

class BM25Indexer:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        if index is None:
            self.docs: List[Dict] = []
            return
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        schema = Schema(id=ID(stored=True, unique=True), rule_id=ID(stored=True), body=TEXT(stored=True))
        if index.exists_in(index_dir):
            self.ix = index.open_dir(index_dir)
        else:
            self.ix = index.create_in(index_dir, schema)

    def build(self, docs: List[Dict]):
        if index is None:
            self.docs = docs
            return
        writer = self.ix.writer()
        for doc in docs:
            writer.update_document(id=doc["id"], rule_id=doc.get("rule_id", ""), body=doc["body"])
        writer.commit()

    def search(self, query: str, topn: int = 20) -> List[Dict]:
        if index is None:
            res = []
            for d in getattr(self, 'docs', [])[:topn]:
                if query in d["body"]:
                    res.append({"id": d["id"], "rule_id": d.get("rule_id", ""), "score": 1.0, "body": d["body"]})
            return res
        qp = QueryParser("body", schema=self.ix.schema)
        q = qp.parse(query)
        results = []
        with self.ix.searcher() as searcher:
            hits = searcher.search(q, limit=topn)
            for hit in hits:
                results.append({
                    "id": hit["id"],
                    "rule_id": hit.get("rule_id", ""),
                    "score": hit.score,
                    "body": hit["body"],
                })
        return results
