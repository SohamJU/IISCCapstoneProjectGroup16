"""Tool functions available to the Recommendation Agent."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agents.authz import require_customer_id
from src.agents.recommendation_agent.config import DEFAULT_RECOMMENDATION_LIMIT
from src.data.postgresql import execute_sql_query_params


# NOTE: none of these tools take a customer_id parameter, deliberately. They
# used to, which meant the model chose whose profile, order history and
# purchase patterns to read — so a customer who mentioned someone else's ID
# could pull that person's email and buying history. Identity now comes from
# the injected ``config`` only; see :mod:`src.agents.authz`.


def _safe_limit(limit: int) -> int:
    return max(1, min(limit, 20))


@tool
def get_customer_profile(config: RunnableConfig) -> str:
    """Fetch the signed-in customer's own profile information.

    Takes no arguments — it always reads the current customer's profile.
    """
    customer_id, error = require_customer_id(config)
    if error:
        return error

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
        return "No profile found on this account."
    return json.dumps(rows[0], indent=2, default=str)


@tool
def get_customer_order_history(config: RunnableConfig, limit: int = 20) -> str:
    """Fetch the signed-in customer's own recent order items with product context.

    Takes no customer identifier — it always reads the current customer's history.

    Args:
        limit: How many recent order items to return (1-20).
    """
    customer_id, error = require_customer_id(config)
    if error:
        return error

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
        return "No order history found on this account."
    return json.dumps(rows, indent=2, default=str)


@tool
def recommend_for_customer(
    config: RunnableConfig,
    budget: float = 500.0,
    limit: int = DEFAULT_RECOMMENDATION_LIMIT,
) -> str:
    """Recommend products for the signed-in customer from their own history.

    Takes no customer identifier — recommendations are always personalised to
    the current customer.

    Args:
        budget: Maximum price per recommended product.
        limit: How many products to recommend.
    """
    customer_id, error = require_customer_id(config)
    if error:
        return error

    safe_limit = _safe_limit(limit)

    customer_rows = execute_sql_query_params(
        "SELECT customer_id, loyalty_tier FROM customers WHERE customer_id = %s",
        (customer_id,),
    )
    if isinstance(customer_rows, str):
        return customer_rows
    if not customer_rows:
        return "I couldn't find a profile on this account."

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

    purchased_ids = [
        str(r.get("product_id")) for r in history_rows if r.get("product_id")
    ]
    category_counter = Counter(
        str(r.get("main_category")) for r in history_rows if r.get("main_category")
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
            (
                top_categories,
                budget,
                purchased_ids if purchased_ids else [""],
                safe_limit,
            ),
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
        "top_categories_from_history": top_categories,
        "recommendations": rec_rows,
    }
    return json.dumps(payload, indent=2, default=str)
