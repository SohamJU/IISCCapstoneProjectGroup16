"""Product retriever — thin wrapper around hybrid search for RAG pipelines.

This module provides the ``ProductRetriever`` class which is the canonical
interface for fetching product candidates within the RAG pipeline.
Agent tools import from here rather than calling hybrid_search directly,
keeping the dependency direction clean.

Usage::

    from src.rag.retriever import ProductRetriever

    retriever = ProductRetriever()
    docs = retriever.retrieve("wireless earbuds long battery life", top_k=5)
    for doc in docs:
        print(doc["title"], doc["price"])
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ProductRetriever:
    """Retriever wrapping hybrid BM25 + Pinecone vector search."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        price_min: float | None = None,
        price_max: float | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-K most relevant products for a query.

        Parameters
        ----------
        query : str
            Natural-language product search query.
        top_k : int
            Number of results to return.
        price_min : float | None
            Minimum price filter (USD).
        price_max : float | None
            Maximum price filter (USD).
        category : str | None
            Category substring filter.

        Returns
        -------
        list[dict]
            Ranked product dicts with ``_rrf_score`` field.
        """
        from src.search.hybrid_search import hybrid_search  # noqa: PLC0415

        return hybrid_search(
            query=query,
            top_k=top_k,
            price_min=price_min,
            price_max=price_max,
            category=category,
        )

    def retrieve_as_context(
        self,
        query: str,
        top_k: int = 5,
        price_min: float | None = None,
        price_max: float | None = None,
        category: str | None = None,
    ) -> str:
        """Retrieve products and format them as an LLM context string.

        Returns
        -------
        str
            Formatted product context ready for injection into a prompt.
        """
        results = self.retrieve(
            query=query,
            top_k=top_k,
            price_min=price_min,
            price_max=price_max,
            category=category,
        )
        if not results:
            return "No relevant products found."

        lines = [f"Found {len(results)} products matching '{query}':\n"]
        for i, p in enumerate(results, 1):
            price_str = f"${p['price']:.2f}" if p.get("price") else "N/A"
            rating_str = (
                f"4{p['average_rating']:.1f}/5.0 ({p.get('rating_count', 0)} reviews)"
                if p.get("average_rating")
                else "No ratings"
            )
            lines.append(
                f"{i}. **{p.get('title', 'Unknown')}**\n"
                f"   Price: {price_str} | {rating_str}\n"
                f"   Category: {p.get('main_category', 'N/A')} | "
                f"Store: {p.get('store', 'N/A')}"
            )
        return "\n".join(lines)
