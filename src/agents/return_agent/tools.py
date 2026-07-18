"""Tool functions for the Return Agent."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import tool

from src.data.postgresql import execute_sql_query

_WRITE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

_MAX_RESULT_ROWS = 8

@tool
def query_returns(sql_query: str) -> str:
    """Execute a read-only SQL SELECT query against return-related tables."""
    if _WRITE_PATTERN.search(sql_query):
        return (
            "ERROR: Only SELECT queries are allowed. "
            "Write operations are blocked."
        )

    result = execute_sql_query(sql_query)
    if isinstance(result, str):
        return result

    if not result:
        return "No return results found for the provided query."

    if len(result) > _MAX_RESULT_ROWS:
        payload = result[:_MAX_RESULT_ROWS]
        footer = f"\n... (showing {_MAX_RESULT_ROWS} of {len(result)} rows)"
    else:
        payload = result
        footer = ""

    return json.dumps(payload, indent=2, default=str) + footer
