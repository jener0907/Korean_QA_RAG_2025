import os
from dataclasses import dataclass
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

@dataclass
class Config:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    use_qdrant: bool = os.getenv("USE_QDRANT", "true").lower() == "true"
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    use_faiss: bool = os.getenv("USE_FAISS", "false").lower() == "true"
    use_rerank: bool = os.getenv("USE_RERANK", "false").lower() == "true"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "your-20b-quant")
    alpha_bm25: float = float(os.getenv("ALPHA_BM25", 0.6))
    beta_vector: float = float(os.getenv("BETA_VECTOR", 0.4))
    topn: int = int(os.getenv("TOPN", 20))
    topk: int = int(os.getenv("TOPK", 5))

def get_config() -> Config:
    return Config()
