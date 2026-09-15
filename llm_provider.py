import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REASONING_EFFORT = "low"


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
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
        reasoning_effort=os.getenv("GROQ_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
        max_retries=2,
        groq_api_key=api_key,
    )


def reset_llm_cache():
    pass
