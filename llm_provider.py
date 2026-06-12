import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.3


def get_llm():
    """Return a ChatGroq client configured from environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to Streamlit secrets or .env file."
        )

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=float(os.getenv("GROQ_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
        groq_api_key=api_key,
    )


def reset_llm_cache():
    pass
