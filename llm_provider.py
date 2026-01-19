# llm_provider.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

_LLM = None


def get_llm():
    global _LLM

    if _LLM is not None:
        return _LLM

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Check your .env file."
        )

    _LLM = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=api_key,
    )

    return _LLM
