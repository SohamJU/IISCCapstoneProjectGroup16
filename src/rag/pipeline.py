"""End-to-end RAG pipeline for product recommendations.

Composes the retrieval and generation steps into a single callable:

    query → retrieve (hybrid BM25 + Pinecone) → format context → LLM response

Usage::

    from src.rag.pipeline import ProductRAGPipeline

    pipeline = ProductRAGPipeline()
    answer = pipeline.run("What are the best noise-cancelling headphones under $200?")
    print(answer)
"""

from __future__ import annotations

import logging
from typing import Any

from src.rag.retriever import ProductRetriever

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """\
You are a helpful product recommendation assistant for an electronics \
and appliances store. Answer the user's question using ONLY the product \
information provided below. Be concise and helpful. If no relevant \
products are found, say so clearly.\
"""


class ProductRAGPipeline:
    """Query-retrieve-generate pipeline for product recommendations.

    Parameters
    ----------
    top_k : int
        Number of products to retrieve per query.
    system_prompt : str | None
        Custom system prompt. Uses a default if ``None``.
    """

    def __init__(
        self,
        top_k: int = 5,
        system_prompt: str | None = None,
    ) -> None:
        self._top_k = top_k
        self._retriever = ProductRetriever()
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    def run(
        self,
        query: str,
        price_min: float | None = None,
        price_max: float | None = None,
        category: str | None = None,
    ) -> str:
        """Run the full RAG pipeline for a product query.

        Parameters
        ----------
        query : str
            Natural-language user query.
        price_min : float | None
            Optional minimum price filter.
        price_max : float | None
            Optional maximum price filter.
        category : str | None
            Optional category filter.

        Returns
        -------
        str
            The LLM-generated recommendation response.
        """
        # 1. Retrieve relevant products
        context = self._retriever.retrieve_as_context(
            query=query,
            top_k=self._top_k,
            price_min=price_min,
            price_max=price_max,
            category=category,
        )
        logger.debug("Retrieved context:\n%s", context)

        # 2. Build the full prompt
        full_prompt = (
            f"{self._system_prompt}\n\n"
            f"## Retrieved Products\n\n{context}\n\n"
            f"## User Question\n\n{query}"
        )

        # 3. Generate response via LLM
        try:
            from src.agents.llm import get_llm  # noqa: PLC0415

            llm = get_llm()
            response = llm.invoke(full_prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            # Return the retrieved context directly as a fallback
            return f"Here are the products I found:\n\n{context}"

    def retrieve_only(
        self,
        query: str,
        price_min: float | None = None,
        price_max: float | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve products without LLM generation (useful for evaluation).

        Returns
        -------
        list[dict]
            Raw ranked product dicts from the retriever.
        """
        return self._retriever.retrieve(
            query=query,
            top_k=self._top_k,
            price_min=price_min,
            price_max=price_max,
            category=category,
        )
