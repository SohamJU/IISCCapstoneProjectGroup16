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
from src.utils.text import normalise_typography

_LOGGER = get_logger(__name__)

__all__ = [
    "NOT_AUTHENTICATED_MESSAGE",
    "THIRD_PARTY_REFUSAL",
    "account_holder_name",
    "is_known_customer_name",
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

THIRD_PARTY_REFUSAL = (
    "I can only access the account you're signed in to, so I can't look up "
    "anyone else's orders, returns or account details — not by name, email or "
    "any other identifier. If you'd like to see your own orders, just ask."
)

#: Cache of customer_id -> display name. The account holder's name is needed on
#: every turn to prevent misattribution, and it never changes within a session.
_NAME_CACHE: dict[str, str] = {}


def account_holder_name(config: RunnableConfig | None) -> str:
    """Return the signed-in customer's display name, or "" if unknown.

    Exists because of a presentation bug rather than an access-control one:
    asked for "the orders placed by Mason Smith" while signed in as someone
    else, the agent correctly returned the *signed-in* customer's orders — and
    then captioned them "the account for Mason Smith". No data crossed the
    boundary, but the customer saw their own orders attributed to a stranger,
    which reads exactly like a breach.

    Telling the agent whose account it is actually holding lets it name the
    right person, and lets :func:`detect_misattribution` verify that it did.
    """
    customer_id = current_customer_id(config)
    if not customer_id:
        return ""
    if customer_id in _NAME_CACHE:
        return _NAME_CACHE[customer_id]

    rows = execute_sql_query_params(
        "SELECT first_name, last_name FROM customers WHERE customer_id = %s",
        (customer_id,),
    )
    if isinstance(rows, str) or not rows:
        return ""

    name = f"{rows[0].get('first_name') or ''} {rows[0].get('last_name') or ''}".strip()
    if name:
        _NAME_CACHE[customer_id] = name
    return name


#: Cache of candidate name -> "is this a real customer". Bounded by how many
#: distinct names ever reach the attribution guard, which is very few.
_KNOWN_NAME_CACHE: dict[str, bool] = {}


def is_known_customer_name(candidate: str) -> bool:
    """True when ``candidate`` matches a real customer's first and last name.

    Used by the attribution guard to decide whether a name in a reply is a
    person or a product. Matching on the word set rather than on the exact
    string means "Mason Smith" and "Smith, Mason" both resolve, while
    "Maytag Refrigerator Water" resolves to nobody and is left alone.
    """
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", candidate)]
    if len(words) < 2:
        return False

    key = " ".join(sorted(words))
    if key in _KNOWN_NAME_CACHE:
        return _KNOWN_NAME_CACHE[key]

    rows = execute_sql_query_params(
        """
        SELECT 1
        FROM customers
        WHERE LOWER(first_name) = ANY(%s) AND LOWER(last_name) = ANY(%s)
        LIMIT 1
        """,
        (words, words),
    )
    known = not isinstance(rows, str) and bool(rows)
    _KNOWN_NAME_CACHE[key] = known
    return known


def _clean_identifier(raw: str) -> str:
    """Normalise an identifier arriving from the model.

    The model frequently passes back an id it read from an earlier turn, where
    it had written the hyphen as U+2011. Without this the format check rejects
    a perfectly valid order number as malformed.
    """
    return normalise_typography(raw or "").strip().upper()


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

    cleaned = _clean_identifier(order_id)
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

    clean_order = _clean_identifier(order_id)
    clean_item = _clean_identifier(order_item_id)
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

    cleaned = _clean_identifier(return_id)
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
