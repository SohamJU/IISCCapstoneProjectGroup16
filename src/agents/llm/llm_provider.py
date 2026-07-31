"""LLM provider factory — single entry point for all agents.

Currently configured for Groq API via the langchain ChatOpenAI wrapper.
Add new factory functions here to support additional providers.

Three factories are exposed:

* :func:`get_llm` — general purpose chat model used by the ReAct agents.
* :func:`get_router_llm` — small, deterministic model used for intent
  classification. Kept separate so routing cost/latency can be tuned
  independently of the agents.
* :func:`get_synthesis_llm` — used by the graph's synthesis node to merge
  multiple agent outputs into one coherent reply.

All three share a single process-wide rate limiter so the Groq
requests-per-minute budget is respected across the whole application
rather than per-client.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import SecretStr


# ── Defaults ──────────────────────────────────────────────────────────────
# Groq rate limits are enforced PER MODEL, and each model has its own
# independent bucket. Verified from live x-ratelimit-* headers on the free
# tier:
#
#   openai/gpt-oss-120b      8,000 TPM | 1,000 RPD | 200,000 TPD
#   openai/gpt-oss-20b       8,000 TPM | 1,000 RPD   <- separate bucket
#   llama-3.3-70b-versatile 12,000 TPM | 1,000 RPD   <- separate bucket
#   llama-3.1-8b-instant     6,000 TPM | 14,400 RPD  <- separate bucket
#
# Running the router, the agents and the synthesis step all on one model
# therefore wastes free capacity. Routing and synthesis are easy, bounded
# tasks, so they run on gpt-oss-20b — same family and same structured-output
# behaviour, but a completely separate quota. This roughly doubles the number
# of conversations the free tier supports.
_DEFAULT_MODEL = os.getenv("SUPPORT_LLM_MODEL", "openai/gpt-oss-120b")
_ROUTER_MODEL = os.getenv("SUPPORT_ROUTER_MODEL", "openai/gpt-oss-20b")
_SYNTHESIS_MODEL = os.getenv("SUPPORT_SYNTHESIS_MODEL", "openai/gpt-oss-20b")
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_BASE_URL = os.getenv("SUPPORT_LLM_BASE_URL", "https://api.groq.com/openai/v1")
_DEFAULT_MAX_RETRIES = 3

# ── Token budgets ─────────────────────────────────────────────────────────
# CRITICAL: Groq reserves `prompt_tokens + max_tokens` against your quota at
# request time, regardless of how many tokens the model actually emits.
# Measured directly from a 429 payload:
#
#   prompt ~2,821 tokens + max_tokens 4,096  ->  "Requested 7278"
#
# So max_tokens is not a safety cap, it is a *pre-charge*. Setting it to 4096
# cost ~3,000 tokens of quota on every single call for nothing. With the free
# tier's 8,000 tokens/minute that also meant one ReAct step nearly exhausted
# the per-minute bucket, causing 429s mid-turn.
#
# These values are sized to observed real output plus reasoning-token headroom.
# Raise only if you see truncated answers (finish_reason == "length").
_DEFAULT_MAX_TOKENS = int(os.getenv("SUPPORT_LLM_MAX_TOKENS", "2048"))

# Routing returns a small structured object — it needs almost nothing.
_ROUTER_MAX_TOKENS = int(os.getenv("SUPPORT_ROUTER_MAX_TOKENS", "384"))
_SYNTHESIS_MAX_TOKENS = int(os.getenv("SUPPORT_SYNTHESIS_MAX_TOKENS", "1024"))

# ── Rate limiter ──────────────────────────────────────────────────────────
# Groq's free tier is ~30 requests/minute for this model. A single user turn
# now costs 1 router call + 1-2 agent ReAct loops (2-4 calls each), so the
# old 0.5 rps ceiling made every turn take 15+ seconds. Default to 2 rps and
# make it tunable — lower it if you start seeing 429s.
_REQUESTS_PER_SECOND = float(os.getenv("SUPPORT_LLM_RPS", "2.0"))
_CHECK_EVERY_N_SECONDS = 0.05
_MAX_BUCKET_SIZE = 5

# ── Reasoning effort ──────────────────────────────────────────────────────
# gpt-oss models emit a chain of thought before every answer, and a ReAct loop
# pays that cost on every step.
#
# Measured on "show me wireless headphones under $250":
#   default effort -> 51s, 3 tool calls
#   low            -> 41s, 2 tool calls, and it picked bad filter arguments
#                     (sorted by price, surfacing no-name products)
#   medium         -> 30s, 1 tool call, premium correctly-ranked results
#
# "medium" wins on both axes: better reasoning picks the right tool arguments
# first time, so the loop terminates in one round-trip instead of two or three.
# Cutting effort below this is a false economy.
_AGENT_REASONING_EFFORT = os.getenv("SUPPORT_AGENT_REASONING_EFFORT", "medium")

# Routing is a single constrained classification — low effort is genuinely
# enough and keeps the per-turn overhead near 1s.
_ROUTER_REASONING_EFFORT = os.getenv("SUPPORT_ROUTER_REASONING_EFFORT", "low")


_RATE_LIMITER: InMemoryRateLimiter | None = None


def _get_rate_limiter() -> InMemoryRateLimiter:
    """Return the process-wide rate limiter shared by every LLM client.

    Previously each ``get_llm()`` call built its own limiter, which meant N
    agents could each independently burn the full quota. One shared limiter
    makes the configured rate an actual application-wide ceiling.
    """
    global _RATE_LIMITER
    if _RATE_LIMITER is None:
        _RATE_LIMITER = InMemoryRateLimiter(
            requests_per_second=_REQUESTS_PER_SECOND,
            check_every_n_seconds=_CHECK_EVERY_N_SECONDS,
            max_bucket_size=_MAX_BUCKET_SIZE,
        )
    return _RATE_LIMITER


def _require_api_key() -> str:
    """Return the Groq API key or raise with an actionable message."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. Add it to your .env file."
        )
    return api_key


def get_llm(
    model: str = _DEFAULT_MODEL,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    base_url: str = _DEFAULT_BASE_URL,
    reasoning_effort: str | None = _AGENT_REASONING_EFFORT,
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
    reasoning_effort : str | None
        ``"low"``, ``"medium"`` or ``"high"`` for reasoning models. ``None``
        leaves the provider default.

    Returns
    -------
    ChatOpenAI
        A fully configured, rate-limited LLM client.

    Raises
    ------
    ValueError
        If the ``GROQ_API_KEY`` environment variable is not set.
    """
    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": SecretStr(_require_api_key()),
        "base_url": base_url,
        "max_retries": _DEFAULT_MAX_RETRIES,
        "rate_limiter": _get_rate_limiter(),
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    return ChatOpenAI(**kwargs)


def get_router_llm() -> ChatOpenAI:
    """Return the LLM used for intent classification.

    Temperature is pinned to 0 and the token budget is small — routing
    returns a short structured object, never prose.
    """
    return get_llm(
        model=_ROUTER_MODEL,
        temperature=0.0,
        max_tokens=_ROUTER_MAX_TOKENS,
        reasoning_effort=_ROUTER_REASONING_EFFORT,
    )


def get_synthesis_llm() -> ChatOpenAI:
    """Return the LLM used to merge multiple agent outputs into one reply."""
    return get_llm(
        model=_SYNTHESIS_MODEL,
        temperature=0.2,
        max_tokens=_SYNTHESIS_MAX_TOKENS,
        reasoning_effort="low",
    )
