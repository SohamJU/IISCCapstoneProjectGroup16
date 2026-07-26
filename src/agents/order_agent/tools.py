"""Tool functions available to the Order Agent."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agents.authz import authorize_order, require_customer_id
from src.data.postgresql import execute_sql_query_params, execute_sql_write


_MAX_ITEMS_PER_ORDER = 20
_ADDRESS_MIN_CHARS = 8
_ADDRESS_MAX_CHARS = 300

# NOTE: every tool below that names an order takes ``config`` and resolves the
# order through :func:`~src.agents.authz.authorize_order` first. ``config`` is
# injected by LangChain and is absent from the schema the model sees, so the
# model cannot supply an identity. Any new tool added here that reads or writes
# customer-owned rows must do the same — a plain ``WHERE order_id = %s`` is a
# cross-customer data leak.


def _next_id(prefix: str, sql: str, width: int) -> str:
    """Generate next sequential ID by reading latest value from database."""
    rows = execute_sql_query_params(sql)
    if isinstance(rows, str) or not rows:
        return f"{prefix}-{1:0{width}d}"

    raw = str(rows[0].get("id", ""))
    parts = raw.split("-")
    if len(parts) != 2 or not parts[1].isdigit():
        return f"{prefix}-{1:0{width}d}"
    return f"{prefix}-{int(parts[1]) + 1:0{width}d}"


def _to_float(value: object) -> float:
    """Normalize numeric values loaded from database."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


@tool
def get_order_status(order_id: str, config: RunnableConfig) -> str:
    """Fetch current status details for one of the signed-in customer's orders.

    Only works for orders belonging to the current customer.

    Args:
        order_id: The order identifier, e.g. "ORD-000123".
    """
    customer_id, error = require_customer_id(config)
    if error:
        return error
    cleaned, error = authorize_order(order_id, config)
    if error:
        return error

    rows = execute_sql_query_params(
        """
        SELECT order_id, status, tracking_number,
               order_date, est_delivery_date, actual_delivery_date,
               payment_status, total_amount
        FROM orders
        WHERE order_id = %s AND customer_id = %s
        """,
        (cleaned, customer_id),
    )

    if isinstance(rows, str):
        return rows
    if not rows:
        return f"No order found for order_id={cleaned}."
    return json.dumps(rows[0], indent=2, default=str)


@tool
def track_order(order_id: str, config: RunnableConfig) -> str:
    """Return shipment-centric tracking details for the customer's own order.

    Only works for orders belonging to the current customer.

    Args:
        order_id: The order identifier, e.g. "ORD-000123".
    """
    customer_id, error = require_customer_id(config)
    if error:
        return error
    cleaned, error = authorize_order(order_id, config)
    if error:
        return error

    rows = execute_sql_query_params(
        """
        SELECT order_id, status, tracking_number,
               est_delivery_date, actual_delivery_date
        FROM orders
        WHERE order_id = %s AND customer_id = %s
        """,
        (cleaned, customer_id),
    )

    if isinstance(rows, str):
        return rows
    if not rows:
        return f"No order found for order_id={cleaned}."
    return json.dumps(rows[0], indent=2, default=str)


@tool
def list_order_items(order_id: str, config: RunnableConfig) -> str:
    """List what the customer actually purchased in one of their own orders.

    Call this for any question about the CONTENTS of an order — "what did I
    buy/purchase/order in ORD-000123", "what's in this order", "show me the
    items", "how many things did I order". ``get_order_status`` and
    ``track_order`` return shipment-level fields only and cannot answer these.

    Only works for orders belonging to the current customer.

    Args:
        order_id: The order identifier, e.g. "ORD-000123".

    Returns:
        JSON with the order-level summary plus one entry per line item
        (product title, quantity, unit price and line total).
    """
    customer_id, error = require_customer_id(config)
    if error:
        return error
    cleaned, error = authorize_order(order_id, config)
    if error:
        return error

    rows = execute_sql_query_params(
        """
        SELECT
            oi.product_id,
            pc.title AS product_title,
            oi.quantity,
            oi.unit_price,
            oi.item_status,
            o.status AS order_status,
            o.order_date,
            o.total_amount
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        LEFT JOIN product_catalog pc ON pc.product_id = oi.product_id
        WHERE o.order_id = %s AND o.customer_id = %s
        ORDER BY oi.order_item_id
        """,
        (cleaned, customer_id),
    )

    if isinstance(rows, str):
        return rows
    if not rows:
        return f"No items found for order_id={cleaned}. Please check the order number."

    items: list[dict[str, Any]] = []
    for row in rows:
        quantity = int(row.get("quantity") or 0)
        unit_price = _to_float(row.get("unit_price"))
        items.append(
            {
                "product_id": row.get("product_id"),
                # A missing catalog row must not blank the item out of the
                # answer — fall back to the ASIN so the customer sees the line.
                "product_title": row.get("product_title") or str(row.get("product_id")),
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": round(unit_price * quantity, 2),
                "item_status": row.get("item_status"),
            }
        )

    payload = {
        "order_id": cleaned,
        "order_status": rows[0].get("order_status"),
        "order_date": rows[0].get("order_date"),
        "order_total": _to_float(rows[0].get("total_amount")),
        "item_count": len(items),
        "items": items,
    }
    return json.dumps(payload, indent=2, default=str)


@tool
def list_customer_orders(config: RunnableConfig, limit: int = 5) -> str:
    """List the signed-in customer's own recent orders.

    Takes no customer identifier — it always operates on the signed-in
    customer. Use this when they ask about "my orders" or cannot remember an
    order number.

    Args:
        limit: How many recent orders to return (1-20).
    """
    customer_id, error = require_customer_id(config)
    if error:
        return error

    safe_limit = max(1, min(limit, 20))
    rows = execute_sql_query_params(
        """
        SELECT order_id, status, order_date, total_amount, payment_status
        FROM orders
        WHERE customer_id = %s
        ORDER BY order_date DESC
        LIMIT %s
        """,
        (customer_id, safe_limit),
    )

    if isinstance(rows, str):
        return rows
    if not rows:
        return "You don't have any orders on this account yet."
    return json.dumps(rows, indent=2, default=str)


@tool
def cancel_order(order_id: str, config: RunnableConfig) -> str:
    """Cancel one of the signed-in customer's own orders, if status allows.

    Only works for orders belonging to the current customer.

    Args:
        order_id: The order identifier, e.g. "ORD-000123".
    """
    customer_id, error = require_customer_id(config)
    if error:
        return error
    cleaned, error = authorize_order(order_id, config)
    if error:
        return error

    rows = execute_sql_query_params(
        "SELECT order_id, status FROM orders WHERE order_id = %s AND customer_id = %s",
        (cleaned, customer_id),
    )
    if isinstance(rows, str):
        return rows
    if not rows:
        return f"No order found for order_id={cleaned}."

    status = str(rows[0].get("status", "")).lower()
    if status in {"delivered", "returned", "cancelled"}:
        return (
            f"Order {cleaned} cannot be cancelled because current status is '{status}'."
        )

    # The ownership predicate is repeated on the writes themselves. authorize_order
    # already passed, but a bare `WHERE order_id = %s` here would still be one
    # refactor away from cancelling a stranger's order.
    order_update = execute_sql_write(
        "UPDATE orders SET status = %s WHERE order_id = %s AND customer_id = %s",
        ("cancelled", cleaned, customer_id),
    )
    if isinstance(order_update, str):
        return order_update

    item_update = execute_sql_write(
        "UPDATE order_items SET item_status = %s WHERE order_id = %s AND customer_id = %s",
        ("cancelled", cleaned, customer_id),
    )
    if isinstance(item_update, str):
        return item_update

    return f"Order {cleaned} has been cancelled successfully."


@tool
def place_order(shipping_address: str, items_json: str, config: RunnableConfig) -> str:
    """Create a new order for the signed-in customer.

    Takes no customer identifier — the order is always placed on the signed-in
    customer's account.

    Args:
        shipping_address: Delivery address text.
        items_json: JSON list with objects: {"product_id": str, "quantity": int}.
    """
    customer_id, error = require_customer_id(config)
    if error:
        return error

    try:
        items = json.loads(items_json)
    except json.JSONDecodeError:
        return (
            "Invalid items_json format. Provide a JSON list, for example: "
            "[{\"product_id\":\"B00MCW7G9M\",\"quantity\":1}]"
        )

    if not isinstance(items, list) or not items:
        return "items_json must be a non-empty JSON list."

    shipping_address_clean = shipping_address.strip()
    if len(shipping_address_clean) < _ADDRESS_MIN_CHARS:
        return "shipping_address looks too short. Please provide a complete address."
    if len(shipping_address_clean) > _ADDRESS_MAX_CHARS:
        return "shipping_address is too long."

    if len(items) > _MAX_ITEMS_PER_ORDER:
        return f"Order supports at most {_MAX_ITEMS_PER_ORDER} line items in one request."

    # Session identity should always resolve to a real customer row. If it does
    # not, the session is malformed — fail rather than writing an orphan order.
    customer_rows = execute_sql_query_params(
        "SELECT customer_id FROM customers WHERE customer_id = %s",
        (customer_id,),
    )
    if isinstance(customer_rows, str):
        return customer_rows
    if not customer_rows:
        return (
            "I couldn't verify your account, so I can't place this order. "
            "Please sign in again."
        )

    clean_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            return "Each item in items_json must be an object."
        product_id = str(item.get("product_id", "")).strip()
        try:
            quantity = int(item.get("quantity", 0))
        except Exception:
            quantity = 0
        if not product_id or quantity <= 0:
            return "Each item must include product_id and quantity>0."
        clean_items.append({"product_id": product_id, "quantity": quantity})

    # Consolidate duplicate product lines to reduce DB write noise.
    consolidated: dict[str, int] = {}
    for item in clean_items:
        product_id = str(item["product_id"])
        consolidated[product_id] = consolidated.get(product_id, 0) + int(item["quantity"])
    clean_items = [{"product_id": pid, "quantity": qty} for pid, qty in consolidated.items()]

    product_ids = [i["product_id"] for i in clean_items]
    product_rows = execute_sql_query_params(
        """
        SELECT product_id, title, price
        FROM product_catalog
        WHERE product_id = ANY(%s)
        """,
        (product_ids,),
    )
    if isinstance(product_rows, str):
        return product_rows

    products_map = {str(row["product_id"]): row for row in product_rows}
    missing = [pid for pid in product_ids if pid not in products_map]
    if missing:
        return f"These product_ids were not found: {', '.join(missing)}"

    order_id = _next_id(
        "ORD",
        "SELECT order_id AS id FROM orders ORDER BY order_id DESC LIMIT 1",
        width=6,
    )
    tracking_number = f"TRK-{uuid4().hex[:12].upper()}"
    now = datetime.now(UTC).date()
    est_delivery = now + timedelta(days=5)

    total_amount = 0.0
    enriched_items: list[dict[str, Any]] = []
    for item in clean_items:
        product = products_map[item["product_id"]]
        unit_price = _to_float(product.get("price"))
        line_total = unit_price * int(item["quantity"])
        total_amount += line_total
        enriched_items.append(
            {
                "product_id": item["product_id"],
                "quantity": int(item["quantity"]),
                "unit_price": unit_price,
                "title": str(product.get("title", "")),
            }
        )

    order_insert = execute_sql_write(
        """
        INSERT INTO orders (
            order_id, customer_id, order_date, status, tracking_number,
            est_delivery_date, actual_delivery_date, payment_status,
            shipping_address, total_amount
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            order_id,
            customer_id,
            str(now),
            "processing",
            tracking_number,
            str(est_delivery),
            None,
            "paid",
            shipping_address_clean,
            round(total_amount, 2),
        ),
    )
    if isinstance(order_insert, str):
        return order_insert

    for item in enriched_items:
        order_item_id = _next_id(
            "OI",
            "SELECT order_item_id AS id FROM order_items ORDER BY order_item_id DESC LIMIT 1",
            width=7,
        )
        item_insert = execute_sql_write(
            """
            INSERT INTO order_items (
                order_item_id, order_id, product_id, customer_id,
                quantity, unit_price, item_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                order_item_id,
                order_id,
                item["product_id"],
                customer_id,
                item["quantity"],
                item["unit_price"],
                "processing",
            ),
        )
        if isinstance(item_insert, str):
            return item_insert

    payload = {
        "order_id": order_id,
        "tracking_number": tracking_number,
        "status": "processing",
        "estimated_delivery_date": str(est_delivery),
        "total_amount": round(total_amount, 2),
        "items": enriched_items,
    }
    return json.dumps(payload, indent=2, default=str)
