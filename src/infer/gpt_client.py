from typing import List
from . import decider  # for type hints (not used)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# GPT API 클라이언트

def generate_one_line_answer(query: str, context: List[str], rule_hints: str, config) -> str:
    if OpenAI is None or not config.openai_api_key:
        return "판단불가.(API키 없음)"
    client = OpenAI(api_key=config.openai_api_key)
    system_prompt = "너의 출력은 한 줄, ‘~가 맞다.(이유)’ 형식이어야 한다. 근거가 없으면 추정이라고 명시하라."
    user_prompt = f"질의: {query}\n후보 규정 요약: {rule_hints}\n근거 발췌: {' '.join(context)}"
    try:
        resp = client.chat.completions.create(model=config.openai_model, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        answer = resp.choices[0].message.content.strip().replace("\n", "")
        return answer
    except Exception as e:
        return f"판단불가.(API 오류: {e})"
