import re
import unicodedata

# 텍스트 정규화 함수

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # 특수 따옴표 통일
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    # 괄호 통일
    text = text.replace("（", "(").replace("）", ")")
    # 연속 공백/빈 줄 축소
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()
