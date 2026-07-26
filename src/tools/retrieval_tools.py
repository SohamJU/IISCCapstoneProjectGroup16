"""Retriever helper tools."""

from __future__ import annotations

from langchain_core.tools import tool

from src.rag.retriever import format_matches, get_retriever


@tool
def policy_search(query: str, top_k: int = 5) -> str:
	"""Search indexed policy documents using vector retrieval."""
	try:
		retriever = get_retriever()
		matches = retriever.search_policies(query=query, top_k=top_k)
		return format_matches(matches)
	except Exception as exc:
		return f"Policy retrieval error: {exc}"


@tool
def review_search(query: str, product_id: str = "", top_k: int = 5) -> str:
	"""Search indexed product review snippets using vector retrieval."""
	try:
		retriever = get_retriever()
		pid = product_id.strip() or None
		matches = retriever.search_reviews(query=query, product_id=pid, top_k=top_k)
		return format_matches(matches)
	except Exception as exc:
		return f"Review retrieval error: {exc}"

