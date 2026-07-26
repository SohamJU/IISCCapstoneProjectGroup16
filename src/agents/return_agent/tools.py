"""Tool functions available to the Return Agent."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agents.authz import (
    authorize_order,
    authorize_order_item,
    authorize_return,
    require_customer_id,
)
from src.agents.return_agent.config import KNOWLEDGE_BASE_DIR, RETURN_WINDOW_DAYS
from src.data.postgresql import execute_sql_query_params, execute_sql_write
from src.rag.retriever import format_matches, get_retriever


# NOTE: returns are a write path against another customer's money. Every tool
# here that names an order, order item or return resolves it through
# :mod:`src.agents.authz` first — see the module docstring there.


def _next_return_id() -> str:
    rows = execute_sql_query_params(
        "SELECT return_id AS id FROM returns ORDER BY return_id DESC LIMIT 1"
    )
    if isinstance(rows, str) or not rows:
        return "RET-000001"

    raw = str(rows[0].get("id", ""))
    parts = raw.split("-")
    if len(parts) != 2 or not parts[1].isdigit():
        return "RET-000001"
    return f"RET-{int(parts[1]) + 1:06d}"


def _load_policy_docs() -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    if not KNOWLEDGE_BASE_DIR.exists():
        return docs

    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
            docs.append((path.name, text))
        except Exception:
            continue
    return docs


@tool
def lookup_return_policy(query: str) -> str:
    """Search return/refund policy guidance using Pinecone retrieval.

    Falls back to local markdown scan when vector retrieval is unavailable.
    """
    query_lower = query.strip().lower()
    if not query_lower:
        return "Please provide a policy question to search."

    try:
        retriever = get_retriever()
        matches = retriever.search_policies(query=query, top_k=6)
        if matches:
            return format_matches(matches)
    except Exception:
        # Fall back to local file search if Pinecone path is unavailable.
        pass

    docs = _load_policy_docs()
    if not docs:
        return "Policy documents are unavailable right now."

    best_matches: list[dict[str, str]] = []
    query_terms = [term for term in query_lower.split() if len(term) > 2]

    for name, text in docs:
        lines = text.splitlines()
        for line in lines:
            line_lower = line.lower()
            if query_lower in line_lower or any(term in line_lower for term in query_terms):
                best_matches.append({"source": name, "line": line.strip()})
                if len(best_matches) >= 8:
                    return json.dumps(best_matches, indent=2)

    if not best_matches:
        return "No matching policy text found. Try a more specific policy question."

    return json.dumps(best_matches, indent=2)


@tool
def list_order_items(order_id: str, config: RunnableConfig) -> str:
    """List the items in the customer's own order, with return eligibility.

    ALWAYS call this first when a customer wants to return something and has
    given you an order ID. It resolves the internal order_item_id for you.

    Customers do not know their order_item_id — it is an internal database key
    they have never seen. Never ask them for it. Use this tool, then:
      * exactly one returnable item -> proceed with it, naming the product;
      * several -> ask which one by PRODUCT NAME, not by ID.

    Only works for orders belonging to the current customer.

    Args:
        order_id: The order identifier, e.g. "ORD-000123".

    Returns:
        JSON list of items with order_item_id, product title, quantity, price
        and status, plus a summary of how many are present.
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
            oi.order_item_id,
            oi.product_id,
            pc.title AS product_title,
            oi.quantity,
            oi.unit_price,
            oi.item_status,
            o.status AS order_status,
            o.order_date
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

    returnable = [
        row
        for row in rows
        if str(row.get("item_status", "")).lower() not in {"cancelled", "returned"}
        and str(row.get("order_status", "")).lower() not in {"cancelled", "returned"}
    ]

    payload = {
        "order_id": cleaned,
        "item_count": len(rows),
        "returnable_count": len(returnable),
        "items": rows,
    }

    if len(returnable) == 1:
        payload["guidance"] = (
            f"Only one returnable item ({returnable[0].get('product_title') or returnable[0].get('product_id')}). "
            "Proceed with its order_item_id — do NOT ask the customer to choose or to supply an ID."
        )
    elif len(returnable) > 1:
        payload["guidance"] = (
            "Multiple returnable items. Ask the customer which one by PRODUCT NAME. "
            "Never show or request the order_item_id."
        )
    else:
        payload["guidance"] = "No returnable items on this order."

    return json.dumps(payload, indent=2, default=str)


def _check_return_eligibility(
    order_id: str,
    order_item_id: str,
    config: RunnableConfig | None,
) -> str:
    """Eligibility check as a plain function.

    Kept separate from the ``@tool`` wrapper below because ``create_return_request``
    needs to reuse it. ``@tool`` turns a function into a ``StructuredTool``
    object, which is NOT directly callable — invoking one like a function
    raises ``'StructuredTool' object is not callable``. Tools must therefore
    never call each other directly; they share plain helpers like this instead.

    ``config`` is threaded through rather than dropped: this helper is the gate
    ``create_return_request`` relies on, so it must enforce ownership itself
    instead of trusting its caller to have done it.
    """
    customer_id, error = require_customer_id(config)
    if error:
        return error
    clean_order, clean_item, error = authorize_order_item(
        order_id, order_item_id, config
    )
    if error:
        return error

    rows = execute_sql_query_params(
        """
        SELECT
            o.order_id,
            o.status AS order_status,
            o.order_date,
            oi.order_item_id,
            oi.item_status,
            oi.product_id,
            oi.quantity,
            oi.unit_price
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_id = %s AND oi.order_item_id = %s AND o.customer_id = %s
        """,
        (clean_order, clean_item, customer_id),
    )

    if isinstance(rows, str):
        return rows
    if not rows:
        return f"No matching order/order_item found for {order_id} / {order_item_id}."

    row = rows[0]
    order_status = str(row.get("order_status", "")).lower()
    item_status = str(row.get("item_status", "")).lower()

    if order_status in {"cancelled", "returned"} or item_status in {"cancelled", "returned"}:
        return json.dumps(
            {
                "eligible": False,
                "reason": f"Current status does not allow return (order={order_status}, item={item_status}).",
            },
            indent=2,
        )

    order_date_value = row.get("order_date")
    if order_date_value is None:
        return json.dumps(
            {
                "eligible": False,
                "reason": "Order date missing; cannot verify return window.",
            },
            indent=2,
        )

    if isinstance(order_date_value, str):
        order_date = datetime.fromisoformat(order_date_value).date()
    else:
        order_date = order_date_value

    today = datetime.now(UTC).date()
    age_days = (today - order_date).days
    eligible = age_days <= RETURN_WINDOW_DAYS

    payload = {
        "eligible": eligible,
        "age_days": age_days,
        "return_window_days": RETURN_WINDOW_DAYS,
        "order_id": clean_order,
        "order_item_id": clean_item,
    }
    if not eligible:
        payload["reason"] = "Return window exceeded for standard returns."

    return json.dumps(payload, indent=2, default=str)


@tool
def check_return_eligibility(
    order_id: str, order_item_id: str, config: RunnableConfig
) -> str:
    """Check whether one of the customer's own order items can be returned.

    Call list_order_items first to obtain the order_item_id — never ask the
    customer for it. Only works for orders belonging to the current customer.

    Args:
        order_id: The order identifier, e.g. "ORD-000123".
        order_item_id: The internal item id from list_order_items.

    Returns:
        JSON with `eligible`, the item's age in days, and the return window.
    """
    return _check_return_eligibility(order_id, order_item_id, config)


@tool
def create_return_request(
    order_id: str, order_item_id: str, reason: str, config: RunnableConfig
) -> str:
    """Create a return request for one of the customer's own order items.

    Call list_order_items first to obtain the order_item_id — never ask the
    customer for it. Only works for orders belonging to the current customer.

    Args:
        order_id: The order identifier, e.g. "ORD-000123".
        order_item_id: The internal item id from list_order_items.
        reason: The customer's stated reason for returning.

    Returns:
        JSON describing the created return, or why it could not be created.
    """
    customer_id, error = require_customer_id(config)
    if error:
        return error
    clean_order, clean_item, error = authorize_order_item(
        order_id, order_item_id, config
    )
    if error:
        return error

    order_id, order_item_id = clean_order, clean_item

    eligibility = _check_return_eligibility(order_id, order_item_id, config)
    try:
        parsed = json.loads(eligibility)
    except json.JSONDecodeError:
        return eligibility

    if not parsed.get("eligible", False):
        return json.dumps(
            {
                "created": False,
                "message": "Return request not created because item is not eligible.",
                "eligibility": parsed,
            },
            indent=2,
            default=str,
        )

    existing = execute_sql_query_params(
        """
        SELECT return_id, status
        FROM returns
        WHERE order_id = %s AND order_item_id = %s
        ORDER BY request_date DESC
        LIMIT 1
        """,
        (order_id, order_item_id),
    )
    if isinstance(existing, str):
        return existing
    if existing:
        status = str(existing[0].get("status", "")).lower()
        if status in {"pending", "approved", "refunded"}:
            return json.dumps(
                {
                    "created": False,
                    "message": "An active return request already exists for this order item.",
                    "existing_return_id": existing[0].get("return_id"),
                    "existing_status": status,
                },
                indent=2,
                default=str,
            )

    row_data = execute_sql_query_params(
        """
        SELECT oi.product_id,
               (oi.quantity * oi.unit_price) AS refund_amount
        FROM order_items oi
        WHERE oi.order_id = %s AND oi.order_item_id = %s AND oi.customer_id = %s
        """,
        (order_id, order_item_id, customer_id),
    )
    if isinstance(row_data, str):
        return row_data
    if not row_data:
        return "Could not load order item details for return creation."

    item = row_data[0]
    return_id = _next_return_id()

    write_result = execute_sql_write(
        """
        INSERT INTO returns (
            return_id, order_id, order_item_id, product_id, customer_id,
            reason, status, refund_amount, request_date
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            return_id,
            order_id,
            order_item_id,
            item.get("product_id"),
            # The authenticated customer, never the row's own customer_id. If
            # those two ever disagreed, writing the row's value would file the
            # refund against whoever the row claimed to belong to.
            customer_id,
            reason,
            "pending",
            float(item.get("refund_amount", 0.0)),
            str(datetime.now(UTC).date()),
        ),
    )
    if isinstance(write_result, str):
        return write_result

    payload = {
        "created": True,
        "return_id": return_id,
        "status": "pending",
        "order_id": order_id,
        "order_item_id": order_item_id,
    }
    return json.dumps(payload, indent=2)


@tool
def get_return_status(return_id: str, config: RunnableConfig) -> str:
    """Fetch current status of one of the customer's own return requests.

    Only works for returns belonging to the current customer.

    Args:
        return_id: The return identifier, e.g. "RET-000123".
    """
    customer_id, error = require_customer_id(config)
    if error:
        return error
    cleaned, error = authorize_return(return_id, config)
    if error:
        return error

    rows = execute_sql_query_params(
        """
        SELECT return_id, order_id, order_item_id, status,
               refund_amount, request_date, reason
        FROM returns
        WHERE return_id = %s AND customer_id = %s
        """,
        (cleaned, customer_id),
    )

    if isinstance(rows, str):
        return rows
    if not rows:
        return f"No return found for return_id={cleaned}."
    return json.dumps(rows[0], indent=2, default=str)
