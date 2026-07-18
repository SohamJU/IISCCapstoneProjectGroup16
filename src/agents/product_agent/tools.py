"""Tool functions available to the Product Recommendation Agent.

Each tool is decorated with ``@tool`` so LangGraph can bind them
to the LLM's tool-calling interface.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from src.agents.common import limit_rows, reject_write_sql
from src.data.postgresql import execute_sql_query
from src.rag.retriever import format_matches, get_retriever

_MAX_RESULT_ROWS = 5
# truncate large result sets for token budget
# (limit to 5 to prevent 413 error)


@tool
def query_products(sql_query: str) -> str:
    """Execute a read-only SQL SELECT query against the product_catalog table.

    Use this tool to look up products, filter by category, price, rating,
    or any other column in the product_catalog table. Only SELECT queries
    are permitted.

    Args:
        sql_query: A valid SQL SELECT statement targeting the product_catalog table.

    Returns:
        JSON-formatted results, or an error/safety message.
    """
    allowed, error = reject_write_sql(sql_query)
    if not allowed:
        return error

    result: list[dict[str, Any]] | str = execute_sql_query(sql_query)

    # If execute_sql_query returned an error string, pass it through
    if isinstance(result, str):
        return result

    if not result:
        return "No results found for the given query."

    result, footer = limit_rows(result, max_rows=_MAX_RESULT_ROWS)

    return json.dumps(result, indent=2, default=str) + footer


@tool
def search_product_reviews(query: str, product_id: str = "", top_k: int = 5) -> str:
    """Search indexed product review snippets using vector retrieval.

    Args:
        query: Review-related natural language query.
        product_id: Optional product ID (ASIN) to narrow search.
        top_k: Number of matches to return.

    Returns:
        Matching review snippets, or an error message.
    """
    try:
        retriever = get_retriever()
        pid = product_id.strip() or None
        matches = retriever.search_reviews(query=query, product_id=pid, top_k=top_k)
        return format_matches(matches)
    except Exception as exc:
        return f"Review retrieval error: {exc}"
