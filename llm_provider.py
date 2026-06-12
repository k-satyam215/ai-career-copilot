import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.3

_llm = None


def get_llm():
    """Return a cached ChatGroq client, configured from environment variables."""
    global _llm

    if _llm is not None:
        return _llm

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
        )

    _llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=float(os.getenv("GROQ_TEMPERATURE", DEFAULT_TEMPERATURE)),
        groq_api_key=api_key,
    )

    return _llm


def reset_llm_cache():
    """Reset the cached client (useful for tests)."""
    global _llm
    _llm = None
