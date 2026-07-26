"""Per-customer authorization for tools that touch customer-owned data.

Why this exists
---------------
Customer identity used to travel to the specialists as *prose*::

    CUSTOMER IDENTITY: The customer is authenticated as customer_id 'AF...'.
    Use this directly in tool calls.

That is a suggestion to a language model, not an access control. Every
order/return tool accepted whatever ``order_id`` the model passed and queried
it with no ownership predicate, so asking "what did I purchase in ORD-006041"
while signed in as a different customer returned another person's order in
full — and ``cancel_order`` / ``create_return_request`` would happily write
against it.

The fix has two halves:

1. **Identity the model cannot set.** The authenticated ``customer_id`` is
   carried in ``RunnableConfig["configurable"]``, injected by the graph node
   from :class:`~src.agents.graph.state.SupportState`. LangChain excludes the
   ``config`` parameter from the JSON schema it shows the model, so the model
   has no way to supply, guess or override it — no amount of prompt injection
   reaches this value.

2. **Ownership checked in the SQL predicate.** Every customer-scoped tool
   resolves the row with ``AND customer_id = %s`` before doing anything else.
   The check lives in the query, not in a post-filter, so a missed branch
   cannot leak a row.

Fail-closed
-----------
No authenticated customer means no access to customer-owned data. An
unidentified session can still browse the catalog and read policy, which is
all a signed-out visitor should be able to do anyway.

Enumeration
-----------
"That order does not exist" and "that order belongs to someone else" return
the *same* message. Distinguishing them would turn any of these tools into an
oracle for probing which order IDs are real.
"""

from __future__ import annotations

import re

from langchain_core.runnables import RunnableConfig

from src.data.postgresql import execute_sql_query_params
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

__all__ = [
    "NOT_AUTHENTICATED_MESSAGE",
    "authorize_order",
    "authorize_order_item",
    "authorize_return",
    "current_customer_id",
    "require_customer_id",
]

#: Deliberately permissive on digit count so a customer typing "ORD-1234" gets
#: a "no such order" answer rather than a format lecture.
_ORDER_ID_RE = re.compile(r"^ORD-\d{1,10}$", re.IGNORECASE)
_RETURN_ID_RE = re.compile(r"^RET-\d{1,10}$", re.IGNORECASE)
_ORDER_ITEM_ID_RE = re.compile(r"^OI-\d{1,10}$", re.IGNORECASE)

NOT_AUTHENTICATED_MESSAGE = (
    "I can't look up order or return details without a signed-in account. "
    "Please sign in and try again — I can still help with product questions "
    "and general policy in the meantime."
)


def _denied(kind: str, identifier: str) -> str:
    """Uniform refusal for 'no such record' and 'not yours' alike.

    Both cases must be indistinguishable, otherwise the difference between the
    two replies tells an attacker which IDs exist.
    """
    return (
        f"I couldn't find {kind} {identifier} on your account. "
        "Please check the number and try again."
    )


def current_customer_id(config: RunnableConfig | None) -> str | None:
    """Return the authenticated customer for this invocation, if any.

    Reads only from ``configurable``, which is supplied by the graph node.
    Tool arguments are never consulted — that is the whole point.
    """
    if not config:
        return None
    configurable = config.get("configurable") or {}
    raw = configurable.get("customer_id")
    if raw is None:
        return None
    cleaned = str(raw).strip()
    return cleaned or None


def require_customer_id(config: RunnableConfig | None) -> tuple[str, str]:
    """Resolve the authenticated customer or explain why access is refused.

    Returns:
        ``(customer_id, "")`` when authenticated, else ``("", message)``.
    """
    customer_id = current_customer_id(config)
    if not customer_id:
        return "", NOT_AUTHENTICATED_MESSAGE
    return customer_id, ""


def authorize_order(
    order_id: str,
    config: RunnableConfig | None,
) -> tuple[str, str]:
    """Verify the caller owns ``order_id``.

    Returns:
        ``(normalised_order_id, "")`` when the order exists AND belongs to the
        authenticated customer, else ``("", refusal_message)``.
    """
    customer_id, error = require_customer_id(config)
    if error:
        return "", error

    cleaned = (order_id or "").strip().upper()
    if not _ORDER_ID_RE.match(cleaned):
        return "", "Invalid order_id format. Expected something like ORD-000123."

    rows = execute_sql_query_params(
        "SELECT order_id FROM orders WHERE order_id = %s AND customer_id = %s",
        (cleaned, customer_id),
    )
    if isinstance(rows, str):  # database error surfaced as a string
        return "", rows
    if not rows:
        _LOGGER.warning(
            "authz: denied order access customer=%s order=%s", customer_id, cleaned
        )
        return "", _denied("order", cleaned)

    return cleaned, ""


def authorize_order_item(
    order_id: str,
    order_item_id: str,
    config: RunnableConfig | None,
) -> tuple[str, str, str]:
    """Verify the caller owns both ``order_id`` and the line item within it.

    ``order_items`` carries its own ``customer_id``, so both columns are
    checked: a line item must belong to the caller *and* sit on an order that
    belongs to the caller.

    Returns:
        ``(order_id, order_item_id, "")`` on success, else ``("", "", message)``.
    """
    customer_id, error = require_customer_id(config)
    if error:
        return "", "", error

    clean_order = (order_id or "").strip().upper()
    clean_item = (order_item_id or "").strip().upper()
    if not _ORDER_ID_RE.match(clean_order):
        return "", "", "Invalid order_id format. Expected something like ORD-000123."
    if not _ORDER_ITEM_ID_RE.match(clean_item):
        return "", "", _denied("that item on order", clean_order)

    rows = execute_sql_query_params(
        """
        SELECT oi.order_item_id
        FROM order_items oi
        JOIN orders o ON o.order_id = oi.order_id
        WHERE oi.order_id = %s
          AND oi.order_item_id = %s
          AND o.customer_id = %s
          AND oi.customer_id = %s
        """,
        (clean_order, clean_item, customer_id, customer_id),
    )
    if isinstance(rows, str):
        return "", "", rows
    if not rows:
        _LOGGER.warning(
            "authz: denied item access customer=%s order=%s item=%s",
            customer_id,
            clean_order,
            clean_item,
        )
        return "", "", _denied("that item on order", clean_order)

    return clean_order, clean_item, ""


def authorize_return(
    return_id: str,
    config: RunnableConfig | None,
) -> tuple[str, str]:
    """Verify the caller owns ``return_id``.

    Ownership is taken from the parent order rather than ``returns.customer_id``
    alone, so a return row written with a mismatched customer cannot be read by
    the wrong person.

    Returns:
        ``(normalised_return_id, "")`` on success, else ``("", message)``.
    """
    customer_id, error = require_customer_id(config)
    if error:
        return "", error

    cleaned = (return_id or "").strip().upper()
    if not _RETURN_ID_RE.match(cleaned):
        return "", "Invalid return_id format. Expected something like RET-000123."

    rows = execute_sql_query_params(
        """
        SELECT r.return_id
        FROM returns r
        JOIN orders o ON o.order_id = r.order_id
        WHERE r.return_id = %s
          AND o.customer_id = %s
          AND r.customer_id = %s
        """,
        (cleaned, customer_id, customer_id),
    )
    if isinstance(rows, str):
        return "", rows
    if not rows:
        _LOGGER.warning(
            "authz: denied return access customer=%s return=%s", customer_id, cleaned
        )
        return "", _denied("return", cleaned)

    return cleaned, ""
