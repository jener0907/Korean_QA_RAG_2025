try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# PDF 파일에서 텍스트 추출

def extract_text_from_pdf(path: str) -> str:
    if fitz is None:
        return ""
    doc = fitz.open(path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text
