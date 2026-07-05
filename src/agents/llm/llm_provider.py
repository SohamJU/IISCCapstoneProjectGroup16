"""LLM provider factory — single entry point for all agents.

Currently configured for Groq API via the langchain ChatOpenAI wrapper.
Add new factory functions here to support additional providers.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter


# ── Defaults ──────────────────────────────────────────────────────────────
_DEFAULT_MODEL = "openai/gpt-oss-120b"
_DEFAULT_TEMPERATURE = 0
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MAX_RETRIES = 3

# Rate-limiter settings
_REQUESTS_PER_SECOND = 0.5
_CHECK_EVERY_N_SECONDS = 0.1
_MAX_BUCKET_SIZE = 10


def get_llm(
    model: str = _DEFAULT_MODEL,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    base_url: str = _DEFAULT_BASE_URL,
) -> ChatOpenAI:
    """Return a rate-limited ChatOpenAI instance pointing at Groq.

    Parameters
    ----------
    model : str
        Model identifier to use.
    temperature : float
        Sampling temperature (0 = deterministic).
    max_tokens : int
        Maximum tokens in the response.
    base_url : str
        API endpoint URL.

    Returns
    -------
    ChatOpenAI
        A fully configured, rate-limited LLM client.

    Raises
    ------
    ValueError
        If the ``GROQ_API_KEY`` environment variable is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Add it to your .env file."
        )

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=_REQUESTS_PER_SECOND,
        check_every_n_seconds=_CHECK_EVERY_N_SECONDS,
        max_bucket_size=_MAX_BUCKET_SIZE,
    )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        max_retries=_DEFAULT_MAX_RETRIES,
        rate_limiter=rate_limiter,
    )
