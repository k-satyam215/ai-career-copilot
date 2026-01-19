def chunk_resume(text: str):
    """
    Resume-specific chunking
    Large chunks, fewer splits → faster pipeline
    """
    CHUNK_SIZE = 1200  # large chunks for resumes

    return [
        text[i:i + CHUNK_SIZE]
        for i in range(0, len(text), CHUNK_SIZE)
    ]
