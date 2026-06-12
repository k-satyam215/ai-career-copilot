"""Tests for llm_provider."""
import os
from unittest.mock import MagicMock, patch

import pytest


def test_get_llm_raises_without_key():
    from llm_provider import get_llm, reset_llm_cache
    reset_llm_cache()
    with patch.dict(os.environ, {}, clear=True):
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


def test_get_llm_returns_chagroq_instance():
    from llm_provider import get_llm, reset_llm_cache
    reset_llm_cache()
    mock_groq = MagicMock()
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        with patch("llm_provider.ChatGroq", return_value=mock_groq):
            result1 = get_llm()
            result2 = get_llm()
            assert result1 is mock_groq
            assert result2 is mock_groq
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


def test_get_llm_uses_default_model():
    from llm_provider import get_llm, reset_llm_cache, DEFAULT_MODEL
    reset_llm_cache()
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False):
        with patch("llm_provider.ChatGroq") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_llm()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs.get("model") == DEFAULT_MODEL
    reset_llm_cache()
