import fitz  # PyMuPDF

def load_resume(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    blocks = []

    for page in doc:
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if len(text) > 20:
                blocks.append(text)

    return "\n".join(blocks)
