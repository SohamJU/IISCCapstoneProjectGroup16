"""Hybrid search: BM25 + Pinecone vector retrieval fused with Reciprocal Rank Fusion.

Combines two complementary retrieval strategies:

1. **BM25** (keyword / exact-match) — great for product names, model numbers,
   and queries containing specific brand/spec terms.
2. **Dense vector via Pinecone** (semantic) — great for intent-based queries like
   "something to watch movies in bed" or "quiet appliance for small kitchen".

Results are fused using **Reciprocal Rank Fusion (RRF)**:

    score_rrf(d) = Σ_i  1 / (k + rank_i(d))

where ``k=60`` is the standard smoothing constant.  Documents appearing in
both lists get a combined score; those in only one list still rank based on
that single score.

Fallback strategy
-----------------
If Pinecone is not yet configured (``PINECONE_API_KEY`` is a placeholder or
not set), the hybrid search transparently falls back to BM25-only results
with no error raised.

Usage::

    from src.search.hybrid_search import hybrid_search

    results = hybrid_search(
        query="wireless earbuds with long battery",
        price_max=150.0,
        top_k=5,
    )
    for r in results:
        print(r["title"], r["price"])
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF smoothing constant


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    """Return the RRF contribution of a document at position ``rank`` (1-indexed)."""
    return 1.0 / (k + rank)


def _reciprocal_rank_fusion(
    *ranked_lists: list[dict[str, Any]],
    id_key: str = "product_id",
) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists into a single ranking using RRF.

    Parameters
    ----------
    *ranked_lists
        Variable number of ranked result lists. Each list item must have
        the field specified by ``id_key``.
    id_key : str
        Field used as the unique document identifier.

    Returns
    -------
    list[dict]
        Merged and re-ranked list, sorted by descending RRF score.
        The ``_rrf_score`` field is added to each result dict.
    """
    scores: dict[str, float] = {}
    records: dict[str, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = str(doc.get(id_key, ""))
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank)
            if doc_id not in records:
                records[doc_id] = doc

    fused = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    result = []
    for doc_id in fused:
        row = {**records[doc_id], "_rrf_score": round(scores[doc_id], 6)}
        result.append(row)
    return result


def hybrid_search(
    query: str,
    top_k: int = 5,
    price_min: float | None = None,
    price_max: float | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Run BM25 + Pinecone vector search in parallel and return RRF-fused results.

    Falls back to BM25-only if Pinecone is not configured.

    Parameters
    ----------
    query : str
        Natural-language product search query.
    top_k : int
        Number of final results to return.
    price_min : float | None
        Minimum price filter (applied post-fusion).
    price_max : float | None
        Maximum price filter (applied post-fusion).
    category : str | None
        Case-insensitive category substring filter (applied post-fusion).

    Returns
    -------
    list[dict]
        Top-K products, ranked by RRF fusion score, with ``_rrf_score``.
    """
    from src.agents.product_agent.config import BM25_TOP_K, VECTOR_TOP_K  # noqa: PLC0415
    from src.search.bm25_index import get_bm25_index  # noqa: PLC0415

    bm25_results: list[dict[str, Any]] = []
    vector_results: list[dict[str, Any]] = []

    def _run_bm25() -> list[dict[str, Any]]:
        index = get_bm25_index()
        return index.search(
            query=query,
            top_k=BM25_TOP_K,
            price_min=price_min,
            price_max=price_max,
            category=category,
        )

    def _run_vector() -> list[dict[str, Any]]:
        from src.embeddings.embedder import get_embedder  # noqa: PLC0415
        from src.embeddings.vector_store import PineconeVectorStore  # noqa: PLC0415

        store = PineconeVectorStore()
        if not store.is_available():
            logger.info(
                "Pinecone not configured — hybrid search using BM25 only."
            )
            return []

        q_vec = get_embedder().embed_query(query)

        # Build Pinecone metadata filter for hard constraints
        filter_dict: dict[str, Any] = {}
        if category:
            filter_dict["main_category"] = {"$eq": category}
        if price_min is not None:
            filter_dict.setdefault("price", {})["$gte"] = price_min
        if price_max is not None:
            filter_dict.setdefault("price", {})["$lte"] = price_max

        return store.query(
            embedding=q_vec,
            top_k=VECTOR_TOP_K,
            filter_dict=filter_dict or None,
        )

    # Run BM25 and Pinecone concurrently
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_run_bm25): "bm25",
            pool.submit(_run_vector): "vector",
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                result = future.result()
                if source == "bm25":
                    bm25_results = result
                else:
                    vector_results = result
            except Exception as exc:
                logger.warning("%s search failed: %s", source, exc)

    # Fuse all non-empty lists
    active_lists = [r for r in [bm25_results, vector_results] if r]
    if not active_lists:
        return []

    fused = _reciprocal_rank_fusion(*active_lists)

    # Post-fusion hard filter (catches Pinecone metadata filter gaps)
    filtered = []
    for item in fused:
        price = item.get("price")
        if price_min is not None and (price is None or float(price) < price_min):
            continue
        if price_max is not None and (price is None or float(price) > price_max):
            continue
        if category is not None:
            cat = str(item.get("main_category", "") or "").lower()
            if category.lower() not in cat:
                continue
        filtered.append(item)

    return filtered[:top_k]
