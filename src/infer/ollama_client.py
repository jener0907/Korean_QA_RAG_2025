from typing import List
try:
    import requests
except Exception:
    requests = None

# Ollama 로컬 LLM 클라이언트

def generate_one_line_answer(query: str, context: List[str], rule_hints: str, config) -> str:
    prompt = (
        "너의 출력은 한 줄, ‘~가 맞다.(이유)’ 형식이어야 한다. 근거가 없으면 추정이라고 명시하라.\n"
        f"질의: {query}\n후보 규정 요약: {rule_hints}\n근거 발췌: {' '.join(context)}"
    )
    if requests is None:
        return "판단불가.(requests 미설치)"
    try:
        resp = requests.post(
            f"{config.ollama_base_url}/api/generate",
            json={"model": config.ollama_model, "prompt": prompt},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("response", "").strip().replace("\n", "")
        return answer or "판단불가.(응답 없음)"
    except Exception as e:
        return f"판단불가.(Ollama 오류: {e})"
