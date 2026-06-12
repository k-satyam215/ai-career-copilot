import fitz  # PyMuPDF

MIN_BLOCK_LENGTH = 20


def load_resume(pdf_path: str) -> str:
    """Extract readable text blocks from a resume PDF."""
    doc = fitz.open(pdf_path)
    blocks = []

    for page in doc:
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if len(text) > MIN_BLOCK_LENGTH:
                blocks.append(text)

    doc.close()
    return "\n".join(blocks)
