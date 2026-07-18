"""Tool functions for the Order Agent."""

from __future__ import annotations

import json
import random
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
def query_orders(sql_query: str) -> str:
    """Execute a read-only SQL SELECT query against order-related tables."""
    if _WRITE_PATTERN.search(sql_query):
        return (
            "ERROR: Only SELECT queries are allowed. "
            "Write operations are blocked."
        )

    result = execute_sql_query(sql_query)
    if isinstance(result, str):
        return result

    if not result:
        return "No order results found for the provided query."

    if len(result) > _MAX_RESULT_ROWS:
        payload = result[:_MAX_RESULT_ROWS]
        footer = f"\n... (showing {_MAX_RESULT_ROWS} of {len(result)} rows)"
    else:
        payload = result
        footer = ""

    return json.dumps(payload, indent=2, default=str) + footer


@tool
def fetch_order_history(customer_id: str, limit: int = 10) -> str:
    """Fetch the recent order history for a customer."""
    if not customer_id:
        return "ERROR: customer_id is required to fetch order history."

    safe_customer_id = customer_id.replace("'", "''")
    safe_limit = max(1, min(limit, 20))
    sql_query = (
        "SELECT order_id, order_date, status, total_amount, est_delivery_date "
        "FROM orders "
        f"WHERE customer_id = '{safe_customer_id}' "
        "ORDER BY order_date DESC "
        f"LIMIT {safe_limit}"
    )

    result = execute_sql_query(sql_query)
    if isinstance(result, str):
        return result

    if not result:
        return "No order history found for that customer."

    return json.dumps(result, indent=2, default=str)


@tool
def manage_order(
    action: str,
    customer_id: str | None = None,
    order_id: str | None = None,
    total_amount: float | None = None,
) -> str:
    """Simulate placing or cancelling an order.

    This is a dummy operation. It returns a confirmation message without
    modifying the database.
    """
    action = action.strip().lower()
    if action == "place":
        if not customer_id:
            return "ERROR: customer_id is required to place an order."
        order_id = f"ORD-DUMMY-{random.randrange(100000, 999999)}"
        amount_text = f" for ${total_amount:.2f}" if total_amount is not None else ""
        return f"Order {order_id} has been placed successfully{amount_text}."

    if action == "cancel":
        if not order_id:
            return "ERROR: order_id is required to cancel an order."
        return f"Order {order_id} has been cancelled successfully."

    return "ERROR: action must be either 'place' or 'cancel'."
