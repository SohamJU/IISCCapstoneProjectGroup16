"""Tool functions available to the Product Recommendation Agent.

Each tool is decorated with ``@tool`` so LangGraph can bind them
to the LLM's tool-calling interface.

Tools
-----
search_products
    **Primary tool.** Semantic + keyword hybrid search over the full
    product catalog.  Use this for all natural-language queries.

query_products
    **Secondary tool.** Raw SQL SELECT against ``product_catalog``.
    Use ONLY when the user gives explicit, exact structured filters
    (e.g., exact category name, exact price comparison).

get_twitter_samples
    Placeholder — kept for demo purposes.  Returns an empty result.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import tool

from src.data.postgresql import execute_sql_query


# ── SQL read-only guard ───────────────────────────────────────────────────────
_WRITE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

_MAX_RESULT_ROWS = 5  # truncate large result sets for token budget


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_product(p: dict[str, Any], rank: int) -> dict[str, Any]:
    """Return a token-efficient subset of a product dict for the LLM."""
    return {
        "rank": rank,
        "product_id": p.get("product_id", ""),
        "title": p.get("title", ""),
        "price": p.get("price"),
        "average_rating": p.get("average_rating"),
        "rating_count": p.get("rating_count"),
        "main_category": p.get("main_category", ""),
        "store": p.get("store", ""),
        "is_bestseller": p.get("is_bestseller", False),
    }


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_products(
    query: str,
    price_min: float | None = None,
    price_max: float | None = None,
    category: str | None = None,
    top_k: int = 5,
) -> str:
    """Search the product catalog using semantic + keyword hybrid retrieval.

    This is the PRIMARY tool for all product queries. It understands
    natural language — no need to construct SQL. Use it for questions like:
    "best noise-cancelling headphones under $200", "gaming laptops",
    "quiet washing machines", "wireless earbuds with long battery life".

    IMPORTANT RULES:
    - Call this tool ONCE per user intent.
    - If results are returned, present them to the user immediately.
    - If no results are found, tell the user directly — do NOT retry with
      different wording.

    Args:
        query: Natural-language description of what the user is looking for.
        price_min: Optional minimum price in USD (e.g. 50.0).
        price_max: Optional maximum price in USD (e.g. 300.0).
        category: Optional category filter (e.g. "Electronics", "Computers").
                  Leave None to search across all categories.
        top_k: Number of results to return (default 5, max 10).

    Returns:
        JSON-formatted ranked product results, or a message if none found.
    """
    # Import lazily to avoid slow startup when tool is not used
    from src.search.bm25_index import get_bm25_index

    top_k = min(int(top_k), 10)

    try:
        from src.search.hybrid_search import hybrid_search  # noqa: PLC0415

        results = hybrid_search(
            query=query,
            top_k=top_k,
            price_min=price_min,
            price_max=price_max,
            category=category,
        )
    except Exception as exc:
        return f"Search error: {exc}. Try using query_products with a SQL query instead."


    if not results:
        filters_used = []
        if price_min is not None:
            filters_used.append(f"price >= ${price_min}")
        if price_max is not None:
            filters_used.append(f"price <= ${price_max}")
        if category:
            filters_used.append(f"category contains '{category}'")
        filter_str = (
            f" with filters ({', '.join(filters_used)})" if filters_used else ""
        )
        return (
            f"No products found for '{query}'{filter_str}. "
            "Try broadening the search (remove price filters or use a more "
            "general description)."
        )

    formatted = [_format_product(p, i + 1) for i, p in enumerate(results)]
    return json.dumps(formatted, indent=2, default=str)


@tool
def query_products(sql_query: str) -> str:
    """Execute a read-only SQL SELECT query against the product_catalog table.

    Use this tool ONLY for exact structured lookups that cannot be expressed
    as a natural-language query — for example:
      - Counting products in a specific category
      - Looking up a product by exact product_id
      - Aggregating by price ranges with GROUP BY

    For all other product searches, prefer ``search_products``.

    Args:
        sql_query: A valid SQL SELECT statement targeting the product_catalog table.

    Returns:
        JSON-formatted results, or an error/safety message.
    """
    # Safety: reject any write operation
    if _WRITE_PATTERN.search(sql_query):
        return (
            "ERROR: Only SELECT queries are allowed. "
            "Write operations (INSERT, UPDATE, DELETE, DROP, etc.) are blocked."
        )

    result: list[dict[str, Any]] | str = execute_sql_query(sql_query)

    # If execute_sql_query returned an error string, pass it through
    if isinstance(result, str):
        return result

    if not result:
        return "No results found for the given query."

    # Truncate if too many rows
    total = len(result)
    if total > _MAX_RESULT_ROWS:
        result = result[:_MAX_RESULT_ROWS]
        footer = f"\n... (showing {_MAX_RESULT_ROWS} of {total} results)"
    else:
        footer = ""

    return json.dumps(result, indent=2, default=str) + footer


@tool
def get_twitter_samples(query: str) -> str:
    """Retrieve relevant past customer support conversations from Twitter.

    Use this tool to find similar past customer queries and how support
    agents responded. Results are for tone/style reference only — product
    search data always takes priority over Twitter information.

    Args:
        query: A search query related to the user's question.

    Returns:
        Matching conversation snippets, or a message if none found.
    """
    # TODO: replace with Pinecone vector-store retrieval (Phase 2)
    return "No relevant Twitter samples found."
