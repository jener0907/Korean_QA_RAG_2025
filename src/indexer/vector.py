from typing import List, Dict, Any

try:
    import numpy as np
except Exception:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # 모델 로딩 실패 시
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

try:
    from qdrant_client import QdrantClient, models
except Exception:
    QdrantClient = None

# 벡터 인덱서

class VectorIndexer:
    def __init__(self, use_qdrant: bool, qdrant_url: str, use_faiss: bool):
        self.use_qdrant = use_qdrant and QdrantClient is not None
        self.use_faiss = use_faiss and faiss is not None
        self.dim = 384
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.dim = self.model.get_sentence_embedding_dimension()
            except Exception:
                self.model = None
        else:
            self.model = None
        if self.use_qdrant:
            try:
                self.client = QdrantClient(url=qdrant_url)
            except Exception:
                self.client = None
        elif self.use_faiss:
            self.index = faiss.IndexFlatIP(self.dim)
            self.ids: List[str] = []
        else:
            self.vectors: List[np.ndarray] = []
            self.ids: List[str] = []
            self.docs: Dict[str, str] = {}

    def _embed(self, texts: List[str]) -> Any:
        if self.model and np is not None:
            return self.model.encode(texts)
        if np is None:
            return [[0.0] * self.dim for _ in texts]
        # 모델 없으면 해시 기반 임베딩
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            vecs.append(rng.standard_normal(self.dim))
        return np.vstack(vecs)

    def build(self, docs: List[Dict]):
        if not docs:
            return
        embeddings = self._embed([d["body"] for d in docs])
        if self.use_qdrant and self.client:
            try:
                self.client.recreate_collection(
                    "rules", vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
                )
            except Exception:
                pass
            vectors = [emb.tolist() for emb in embeddings]
            payload = [{"id": d["id"], "rule_id": d.get("rule_id", ""), "body": d["body"]} for d in docs]
            ids = [d["id"] for d in docs]
            self.client.upsert(collection_name="rules", points=models.Batch(ids=ids, vectors=vectors, payloads=payload))
        elif self.use_faiss and np is not None:
            self.index.add(np.array(embeddings).astype(np.float32))
            self.ids.extend([d["id"] for d in docs])
            self.docs = {d["id"]: d for d in docs}
        else:
            for emb, d in zip(embeddings, docs):
                if np is not None:
                    self.vectors.append(emb)
                else:
                    self.vectors.append([0.0] * self.dim)
                self.ids.append(d["id"])
                self.docs[d["id"]] = d

    def search(self, query: str, topn: int = 20) -> List[Dict]:
        if self.use_qdrant and self.client:
            q_emb = self._embed([query])[0]
            hits = self.client.search(collection_name="rules", query_vector=q_emb.tolist(), limit=topn)
            return [{"id": h.id, "rule_id": h.payload.get("rule_id", ""), "score": h.score, "body": h.payload.get("body", "")} for h in hits]
        elif self.use_faiss and np is not None and hasattr(self, 'index') and self.index.ntotal > 0:
            q_emb = np.array(self._embed([query])).astype(np.float32)
            scores, idxs = self.index.search(q_emb, topn)
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx < len(self.ids):
                    doc_id = self.ids[idx]
                    d = self.docs.get(doc_id, {})
                    results.append({"id": doc_id, "rule_id": d.get("rule_id", ""), "score": float(score), "body": d.get("body", "")})
            return results
        elif getattr(self, 'vectors', []) and np is not None:
            q_emb = self._embed([query])[0]
            sims = np.dot(self.vectors, q_emb) / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q_emb) + 1e-9)
            idxs = np.argsort(sims)[::-1][:topn]
            results = []
            for idx in idxs:
                doc_id = self.ids[idx]
                d = self.docs.get(doc_id, {})
                results.append({"id": doc_id, "rule_id": d.get("rule_id", ""), "score": float(sims[idx]), "body": d.get("body", "")})
            return results
        else:
            return []
