"""Tool functions available to the Recommendation Agent."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from langchain_core.tools import tool

from src.agents.recommendation_agent.config import DEFAULT_RECOMMENDATION_LIMIT
from src.data.postgresql import execute_sql_query_params


def _safe_limit(limit: int) -> int:
    return max(1, min(limit, 20))


@tool
def get_customer_profile(customer_id: str) -> str:
    """Fetch profile information for a customer."""
    rows = execute_sql_query_params(
        """
        SELECT customer_id, first_name, last_name, email,
               loyalty_tier, city, state
        FROM customers
        WHERE customer_id = %s
        """,
        (customer_id,),
    )
    if isinstance(rows, str):
        return rows
    if not rows:
        return f"No profile found for customer_id={customer_id}."
    return json.dumps(rows[0], indent=2, default=str)


@tool
def get_customer_order_history(customer_id: str, limit: int = 20) -> str:
    """Fetch recent customer order items with product context."""
    safe_limit = _safe_limit(limit)
    rows = execute_sql_query_params(
        """
        SELECT
            o.order_id,
            o.order_date,
            o.status,
            oi.order_item_id,
            oi.product_id,
            oi.quantity,
            oi.unit_price,
            pc.title,
            pc.main_category,
            pc.average_rating
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        LEFT JOIN product_catalog pc ON pc.product_id = oi.product_id
        WHERE o.customer_id = %s
        ORDER BY o.order_date DESC
        LIMIT %s
        """,
        (customer_id, safe_limit),
    )

    if isinstance(rows, str):
        return rows
    if not rows:
        return f"No order history found for customer_id={customer_id}."
    return json.dumps(rows, indent=2, default=str)


@tool
def recommend_for_customer(
    customer_id: str,
    budget: float = 500.0,
    limit: int = DEFAULT_RECOMMENDATION_LIMIT,
) -> str:
    """Generate personalized recommendations from profile + historical purchases."""
    safe_limit = _safe_limit(limit)

    customer_rows = execute_sql_query_params(
        "SELECT customer_id, loyalty_tier FROM customers WHERE customer_id = %s",
        (customer_id,),
    )
    if isinstance(customer_rows, str):
        return customer_rows
    if not customer_rows:
        return f"Unknown customer_id={customer_id}."

    history_rows = execute_sql_query_params(
        """
        SELECT oi.product_id, pc.main_category
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        LEFT JOIN product_catalog pc ON pc.product_id = oi.product_id
        WHERE o.customer_id = %s
        """,
        (customer_id,),
    )
    if isinstance(history_rows, str):
        return history_rows

    purchased_ids = [str(r.get("product_id")) for r in history_rows if r.get("product_id")]
    category_counter = Counter(
        str(r.get("main_category"))
        for r in history_rows
        if r.get("main_category")
    )
    top_categories = [cat for cat, _ in category_counter.most_common(3)]

    if top_categories:
        rec_rows = execute_sql_query_params(
            """
            SELECT product_id, title, main_category, price, average_rating, rating_count
            FROM product_catalog
            WHERE main_category = ANY(%s)
              AND price <= %s
              AND NOT (product_id = ANY(%s))
            ORDER BY average_rating DESC NULLS LAST, rating_count DESC NULLS LAST
            LIMIT %s
            """,
            (top_categories, budget, purchased_ids if purchased_ids else [""], safe_limit),
        )
    else:
        rec_rows = execute_sql_query_params(
            """
            SELECT product_id, title, main_category, price, average_rating, rating_count
            FROM product_catalog
            WHERE price <= %s
            ORDER BY average_rating DESC NULLS LAST, rating_count DESC NULLS LAST
            LIMIT %s
            """,
            (budget, safe_limit),
        )

    if isinstance(rec_rows, str):
        return rec_rows
    if not rec_rows:
        return "No recommendations found for the given budget and profile context."

    payload: dict[str, Any] = {
        "customer_id": customer_id,
        "top_categories_from_history": top_categories,
        "recommendations": rec_rows,
    }
    return json.dumps(payload, indent=2, default=str)
