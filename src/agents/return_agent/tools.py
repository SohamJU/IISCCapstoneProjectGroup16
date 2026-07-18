"""Tool functions available to the Return Agent."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

from langchain_core.tools import tool

from src.agents.return_agent.config import KNOWLEDGE_BASE_DIR, RETURN_WINDOW_DAYS
from src.data.postgresql import execute_sql_query_params, execute_sql_write
from src.rag.retriever import format_matches, get_retriever


_RETURN_ID_RE = re.compile(r"^RET-\d{6}$")


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
def check_return_eligibility(order_id: str, order_item_id: str) -> str:
    """Check whether a specific order item is eligible for return."""
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
        WHERE o.order_id = %s AND oi.order_item_id = %s
        """,
        (order_id, order_item_id),
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
        "order_id": order_id,
        "order_item_id": order_item_id,
    }
    if not eligible:
        payload["reason"] = "Return window exceeded for standard returns."

    return json.dumps(payload, indent=2, default=str)


@tool
def create_return_request(order_id: str, order_item_id: str, reason: str) -> str:
    """Create a return request if item is eligible."""
    eligibility = check_return_eligibility(order_id=order_id, order_item_id=order_item_id)
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
        SELECT oi.product_id, oi.customer_id,
               (oi.quantity * oi.unit_price) AS refund_amount
        FROM order_items oi
        WHERE oi.order_id = %s AND oi.order_item_id = %s
        """,
        (order_id, order_item_id),
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
            item.get("customer_id"),
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
def get_return_status(return_id: str) -> str:
    """Fetch current status details for a return request."""
    if not _RETURN_ID_RE.match(return_id.strip()):
        return "Invalid return_id format. Expected format like RET-000123."

    rows = execute_sql_query_params(
        """
        SELECT return_id, order_id, order_item_id, status,
               refund_amount, request_date, reason
        FROM returns
        WHERE return_id = %s
        """,
        (return_id,),
    )

    if isinstance(rows, str):
        return rows
    if not rows:
        return f"No return found for return_id={return_id}."
    return json.dumps(rows[0], indent=2, default=str)
