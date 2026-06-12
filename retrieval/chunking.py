CHUNK_SIZE = 1200  # large chunks for resumes -> fewer splits, faster pipeline


def chunk_resume(text: str) -> list:
    """Split resume text into fixed-size chunks."""
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
