"""Resolve dataset placeholders against the live database.

Ground-truth files refer to data symbolically — ``{OWN_ORDER}``,
``{OTHER_CUSTOMER}`` — rather than by literal ID. Hardcoding ``ORD-000055``
would tie the datasets to one snapshot of the demo data and silently rot the
moment anything is reseeded: cases would still "run", just against orders that
no longer exist, and the failures would look like agent regressions.

Resolving at run time also lets one authored case express the thing that
actually matters — *this order belongs to somebody else* — instead of a
literal that happens to be true today.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.data.postgresql import execute_sql_query_params
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

_PLACEHOLDER_PATTERN = re.compile(r"\{([A-Z_]+)\}")


class FixtureError(RuntimeError):
    """Raised when the database cannot supply the data a dataset needs."""


@dataclass(frozen=True)
class Fixtures:
    """Concrete values backing the dataset placeholders."""

    values: dict[str, str]

    def resolve(self, text: str) -> str:
        """Substitute every ``{PLACEHOLDER}`` in ``text``."""

        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in self.values:
                raise FixtureError(
                    f"dataset references unknown placeholder {{{key}}}; "
                    f"known: {', '.join(sorted(self.values))}"
                )
            return self.values[key]

        return _PLACEHOLDER_PATTERN.sub(_replace, text)

    def resolve_all(self, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(self.resolve(value) for value in values)


def _rows(sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
    result = execute_sql_query_params(sql, params)
    if isinstance(result, str):
        raise FixtureError(f"database error while building fixtures: {result}")
    return result


def build_fixtures() -> Fixtures:
    """Query the database for a coherent set of evaluation fixtures.

    Picks two distinct customers who both own orders, then derives every other
    placeholder from them so the ownership relationships the cases assert are
    guaranteed to hold.
    """
    customers = _rows(
        """
        SELECT o.customer_id, c.first_name, c.last_name, c.email,
               COUNT(*) AS order_count
        FROM orders o
        JOIN order_items oi ON oi.order_id = o.order_id
        JOIN customers c ON c.customer_id = o.customer_id
        GROUP BY o.customer_id, c.first_name, c.last_name, c.email
        HAVING COUNT(*) > 2
        ORDER BY order_count DESC
        LIMIT 2
        """
    )
    if len(customers) < 2:
        raise FixtureError(
            "need at least two customers with orders to evaluate cross-account rules"
        )

    customer_a = str(customers[0]["customer_id"])
    customer_b = str(customers[1]["customer_id"])

    values: dict[str, str] = {
        "CUSTOMER": customer_a,
        "OTHER_CUSTOMER": customer_b,
        # Names and email drive the third-party-lookup cases: the realistic way
        # a customer refers to someone else is by name, not by customer_id.
        "CUSTOMER_NAME": (
            f"{customers[0]['first_name']} {customers[0]['last_name']}".strip()
        ),
        "OTHER_CUSTOMER_NAME": (
            f"{customers[1]['first_name']} {customers[1]['last_name']}".strip()
        ),
        "OTHER_CUSTOMER_EMAIL": str(customers[1]["email"] or ""),
    }

    # An order of A's that has line items, so item-level cases have something
    # to assert against.
    own = _rows(
        """
        SELECT o.order_id, o.total_amount, o.status,
               MIN(oi.order_item_id) AS order_item_id,
               COUNT(oi.order_item_id) AS item_count
        FROM orders o
        JOIN order_items oi ON oi.order_id = o.order_id
        WHERE o.customer_id = %s
        GROUP BY o.order_id, o.total_amount, o.status
        ORDER BY o.order_id
        LIMIT 1
        """,
        (customer_a,),
    )
    if not own:
        raise FixtureError(f"customer {customer_a} has no order with line items")

    values["OWN_ORDER"] = str(own[0]["order_id"])
    values["OWN_ORDER_ITEM"] = str(own[0]["order_item_id"])
    values["OWN_ORDER_TOTAL"] = f"{float(own[0]['total_amount']):.2f}"
    values["OWN_ORDER_STATUS"] = str(own[0]["status"])

    # An order belonging to B — the target of every cross-account case.
    other = _rows(
        """
        SELECT o.order_id, MIN(oi.order_item_id) AS order_item_id
        FROM orders o
        JOIN order_items oi ON oi.order_id = o.order_id
        WHERE o.customer_id = %s
        GROUP BY o.order_id
        ORDER BY o.order_id
        LIMIT 1
        """,
        (customer_b,),
    )
    if not other:
        raise FixtureError(f"customer {customer_b} has no order with line items")

    values["OTHER_ORDER"] = str(other[0]["order_id"])
    values["OTHER_ORDER_ITEM"] = str(other[0]["order_item_id"])

    # A product that genuinely exists, for the catalog cases.
    product = _rows(
        """
        SELECT product_id, title, price
        FROM product_catalog
        WHERE title IS NOT NULL AND price IS NOT NULL
        ORDER BY rating_count DESC NULLS LAST
        LIMIT 1
        """
    )
    if not product:
        raise FixtureError("product_catalog is empty")
    values["PRODUCT_ID"] = str(product[0]["product_id"])
    values["PRODUCT_TITLE"] = str(product[0]["title"])

    # Returns are optional — a freshly seeded database may have none. Cases
    # that reference them are skipped rather than failed.
    other_return = _rows(
        """
        SELECT r.return_id
        FROM returns r
        JOIN orders o ON o.order_id = r.order_id
        WHERE o.customer_id = %s
        ORDER BY r.return_id
        LIMIT 1
        """,
        (customer_b,),
    )
    if other_return:
        values["OTHER_RETURN"] = str(other_return[0]["return_id"])

    own_return = _rows(
        """
        SELECT r.return_id
        FROM returns r
        JOIN orders o ON o.order_id = r.order_id
        WHERE o.customer_id = %s
        ORDER BY r.return_id
        LIMIT 1
        """,
        (customer_a,),
    )
    if own_return:
        values["OWN_RETURN"] = str(own_return[0]["return_id"])

    # An ID that is well-formed but deliberately absent, for "unknown order"
    # cases. Derived by walking past the highest real order.
    highest = _rows("SELECT order_id FROM orders ORDER BY order_id DESC LIMIT 1")
    if highest:
        digits = str(highest[0]["order_id"]).split("-")[-1]
        values["MISSING_ORDER"] = f"ORD-{int(digits) + 5000:06d}"
    else:
        values["MISSING_ORDER"] = "ORD-999999"

    _LOGGER.info("evaluation fixtures resolved: %s", sorted(values))
    return Fixtures(values=values)


def required_placeholders(text: str) -> set[str]:
    """Return the placeholder names referenced by ``text``."""
    return set(_PLACEHOLDER_PATTERN.findall(text))
