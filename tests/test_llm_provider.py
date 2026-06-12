"""Tests for llm_provider."""
import pytest
from unittest.mock import patch, MagicMock
import os


def test_get_llm_raises_without_key():
    from llm_provider import get_llm, reset_llm_cache
    reset_llm_cache()
    with patch.dict(os.environ, {}, clear=True):
        # Remove GROQ_API_KEY if present
        env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                get_llm()
    reset_llm_cache()


def test_get_llm_returns_client_with_key():
    from llm_provider import get_llm, reset_llm_cache
    reset_llm_cache()
    mock_groq = MagicMock()
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key-123"}):
        with patch("llm_provider.ChatGroq", return_value=mock_groq):
            result = get_llm()
    assert result is mock_groq
    reset_llm_cache()


def test_get_llm_cached():
    from llm_provider import get_llm, reset_llm_cache
    reset_llm_cache()
    mock_groq = MagicMock()
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        with patch("llm_provider.ChatGroq", return_value=mock_groq) as mock_cls:
            get_llm()
            get_llm()
            # ChatGroq should only be instantiated once
            assert mock_cls.call_count == 1
    reset_llm_cache()


def test_reset_llm_cache():
    from llm_provider import get_llm, reset_llm_cache
    reset_llm_cache()
    mock_groq = MagicMock()
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        with patch("llm_provider.ChatGroq", return_value=mock_groq) as mock_cls:
            get_llm()
            reset_llm_cache()
            get_llm()
            assert mock_cls.call_count == 2
    reset_llm_cache()


def test_custom_model_env_var():
    from llm_provider import get_llm, reset_llm_cache
    reset_llm_cache()
    mock_groq = MagicMock()
    with patch.dict(os.environ, {"GROQ_API_KEY": "key", "GROQ_MODEL": "custom-model"}):
        with patch("llm_provider.ChatGroq", return_value=mock_groq) as mock_cls:
            get_llm()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs.get("model") == "custom-model"
    reset_llm_cache()
