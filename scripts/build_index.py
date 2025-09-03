import os
import glob
import json

from src.utils.config import get_config
from src.extract.pdf_extract import extract_text_from_pdf
from src.extract.docx_extract import extract_text_from_docx
from src.preprocess.normalize import normalize_text
from src.chunk.split_rules import split_rules
from src.indexer.bm25 import BM25Indexer
from src.indexer.vector import VectorIndexer

# 인덱스 빌드 스크립트

def main():
    config = get_config()
    raw_dir = "data_raw"
    docs = []
    for path in glob.glob(os.path.join(raw_dir, "*")):
        text = ""
        if path.lower().endswith(".pdf"):
            text = extract_text_from_pdf(path)
        elif path.lower().endswith(".docx"):
            text = extract_text_from_docx(path)
        if not text:
            continue
        text = normalize_text(text)
        docs.extend(split_rules(text))
    os.makedirs("data_curated", exist_ok=True)
    curated_path = os.path.join("data_curated", "rules.jsonl")
    with open(curated_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    # 인덱스 작성
    bm25 = BM25Indexer("index/bm25")
    bm25.build(docs)
    vec = VectorIndexer(config.use_qdrant, config.qdrant_url, config.use_faiss)
    vec.build(docs)
    print(f"Indexed {len(docs)} documents.")

if __name__ == "__main__":
    main()
