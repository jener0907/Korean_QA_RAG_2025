try:
    from docx import Document
except Exception:
    Document = None

# DOCX 파일에서 텍스트 추출

def extract_text_from_docx(path: str) -> str:
    if Document is None:
        return ""
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return text
