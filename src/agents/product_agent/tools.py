"""Tool functions available to the Product Recommendation Agent.

Each tool is decorated with ``@tool`` so LangGraph can bind them
to the LLM's tool-calling interface.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import tool

from src.data.postgresql import execute_sql_query


# ── SQL read-only guard ───────────────────────────────────────────────────
_WRITE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

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
    agents responded. Results are for tone/style reference only — SQL
    product data always takes priority over Twitter information.

    Args:
        query: A search query related to the user's question.

    Returns:
        Matching conversation snippets, or a message if none found.
    """
    # TODO: replace with vector-store retrieval
    return "No relevant Twitter samples found."
