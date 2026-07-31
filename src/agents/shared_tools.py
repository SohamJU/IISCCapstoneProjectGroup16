"""Tools shared by more than one specialist agent.

Policy lookup used to live only on the Return Agent, yet every agent's prompt
told it to "never invent policy terms" and to "use tools for all policy
details". The Order Agent therefore had no way to answer "what's your
cancellation policy?" — it either refused or made something up. The same
applies to shipping questions on the Product Agent.

This module exposes one retrieval tool over the full knowledge base
(returns, shipping, warranty, payments) so any agent can ground a policy
answer.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.rag.policy_store import format_policy_sections, get_policy_store
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

_DEFAULT_TOP_K = 3
_MAX_TOP_K = 6


def _search_policies_local(query: str, top_k: int) -> str:
    """Search the local section-aware policy index."""
    store = get_policy_store()
    results = store.search(query, top_k=top_k)
    return format_policy_sections(results)


def _search_policies_pinecone(query: str, top_k: int) -> str | None:
    """Try Pinecone retrieval; return ``None`` when it is unavailable.

    Pinecone is optional. When ``PINECONE_API_KEY`` is unset (the current
    default) this returns ``None`` immediately rather than surfacing a stack
    trace string into the model's context.
    """
    try:
        from src.rag.retriever import format_matches, get_retriever

        matches = get_retriever().search_policies(query=query, top_k=top_k)
        if matches:
            return format_matches(matches)
    except Exception as exc:
        _LOGGER.debug(
            "Pinecone policy retrieval unavailable, using local index: %s", exc
        )
    return None


@tool
def lookup_support_policy(query: str, top_k: int = _DEFAULT_TOP_K) -> str:
    """Look up official store policy on returns, refunds, shipping, delivery, warranty, or payments.

    Use this whenever the customer asks what the rules are — return windows,
    refund timelines, shipping costs, delivery estimates, warranty coverage,
    accepted payment methods, cancellation rules. Always ground policy answers
    in this tool's output and cite the section you used. Never state a policy
    term that does not appear in the returned text.

    Args:
        query: The customer's policy question, in natural language.
        top_k: How many policy sections to return (1-6).

    Returns:
        Relevant policy sections with their source headings.
    """
    cleaned = query.strip()
    if not cleaned:
        return "Please provide a policy question to look up."

    bounded_k = max(1, min(int(top_k), _MAX_TOP_K))

    from_pinecone = _search_policies_pinecone(cleaned, bounded_k)
    if from_pinecone:
        return from_pinecone

    return _search_policies_local(cleaned, bounded_k)
