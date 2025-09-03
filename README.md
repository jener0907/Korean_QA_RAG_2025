# Korean QA RAG 2025

간단한 한국어 문장 판단 RAG 시스템의 최소 제품(MVP)입니다. 국립국어원 자료를 기반으로 규정을 검색하고 LLM이 한 줄 판단을 내려줍니다.

## 빠른 시작
```
1) pip install -r requirements.txt
2) cp .env.example .env && OPENAI_API_KEY=...
3) python -m scripts.build_index && uvicorn src.server.api:app --reload
```

## Qdrant 실행(Docker)
```
docker run -p 6333:6333 qdrant/qdrant
```

## Ollama 모델 교체
`.env`의 `OLLAMA_MODEL` 값을 원하는 모델 태그로 변경하면 됩니다.

## 구조
- `data_raw/`: 원문 PDF/DOCX
- `data_curated/`: 정제된 규정 JSONL
- `index/`: BM25 및 벡터 인덱스
- `scripts/`: 빌드/검색/판단/평가 도구
- `src/`: 모듈 소스 코드
