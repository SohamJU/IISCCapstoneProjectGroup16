"""Cross-customer access control tests.

These are regression tests for a real data leak: every order/return tool used
to query by ``order_id`` alone, so a signed-in customer could read — and
cancel, and file returns against — any other customer's orders just by naming
the ID. See :mod:`src.agents.authz`.

The tests below run against the live database because that is where the
ownership predicate lives. A mocked DB would happily "prove" the fix while the
SQL still lacked the ``customer_id`` clause.
"""

from __future__ import annotations

import json

import pytest

from src.agents.authz import NOT_AUTHENTICATED_MESSAGE
from src.agents.order_agent.tools import (
    cancel_order,
    get_order_status,
    list_customer_orders,
    list_order_items,
    place_order,
    track_order,
)
from src.agents.product_agent.tools import query_products
from src.agents.recommendation_agent.tools import (
    get_customer_order_history,
    get_customer_profile,
    recommend_for_customer,
)
from src.agents.return_agent.tools import (
    check_return_eligibility,
    create_return_request,
    get_return_status,
)
from src.agents.return_agent.tools import list_order_items as return_list_order_items
from src.data.postgresql import execute_sql_query_params


def _cfg(customer_id: str | None) -> dict:
    """Build the invocation config the graph would supply."""
    return {"configurable": {"customer_id": customer_id}}


@pytest.fixture(scope="module")
def two_customers() -> tuple[str, str]:
    """Two distinct customers who each own at least one order."""
    rows = execute_sql_query_params(
        """
        SELECT customer_id, COUNT(*) AS order_count
        FROM orders
        GROUP BY customer_id
        HAVING COUNT(*) > 1
        ORDER BY order_count DESC
        LIMIT 2
        """
    )
    if isinstance(rows, str) or len(rows) < 2:
        pytest.skip("database unavailable or lacks two customers with orders")
    return str(rows[0]["customer_id"]), str(rows[1]["customer_id"])


@pytest.fixture(scope="module")
def victim_order(two_customers: tuple[str, str]) -> str:
    """An order owned by customer B — the one customer A must never see."""
    _, customer_b = two_customers
    rows = execute_sql_query_params(
        "SELECT order_id FROM orders WHERE customer_id = %s ORDER BY order_id LIMIT 1",
        (customer_b,),
    )
    if isinstance(rows, str) or not rows:
        pytest.skip("no order available for the second customer")
    return str(rows[0]["order_id"])


@pytest.fixture(scope="module")
def own_order(two_customers: tuple[str, str]) -> str:
    """An order genuinely owned by customer A."""
    customer_a, _ = two_customers
    rows = execute_sql_query_params(
        "SELECT order_id FROM orders WHERE customer_id = %s ORDER BY order_id LIMIT 1",
        (customer_a,),
    )
    if isinstance(rows, str) or not rows:
        pytest.skip("no order available for the first customer")
    return str(rows[0]["order_id"])


# ══════════════════════════════════════════════════════════════════════════
# Reads across the account boundary
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "read_tool",
    [get_order_status, track_order, list_order_items, return_list_order_items],
    ids=[
        "get_order_status",
        "track_order",
        "list_order_items",
        "return_list_order_items",
    ],
)
def test_read_tools_refuse_another_customers_order(
    read_tool, two_customers: tuple[str, str], victim_order: str
) -> None:
    customer_a, _ = two_customers
    result = read_tool.invoke({"order_id": victim_order}, config=_cfg(customer_a))
    assert "couldn't find" in result.lower()


@pytest.mark.parametrize(
    "read_tool",
    [get_order_status, track_order, list_order_items],
    ids=["get_order_status", "track_order", "list_order_items"],
)
def test_read_tools_still_serve_the_owner(
    read_tool, two_customers: tuple[str, str], own_order: str
) -> None:
    """The lockdown must not break the legitimate path."""
    customer_a, _ = two_customers
    result = read_tool.invoke({"order_id": own_order}, config=_cfg(customer_a))
    assert own_order in result
    assert "couldn't find" not in result.lower()


def test_denial_does_not_reveal_whether_the_order_exists(
    two_customers: tuple[str, str], victim_order: str
) -> None:
    """A real-but-foreign order and a nonexistent one must look identical.

    Any difference turns the tool into an oracle for enumerating order IDs.
    """
    customer_a, _ = two_customers
    foreign = get_order_status.invoke(
        {"order_id": victim_order}, config=_cfg(customer_a)
    )
    missing = get_order_status.invoke(
        {"order_id": "ORD-999999"}, config=_cfg(customer_a)
    )
    assert foreign.replace(victim_order, "X") == missing.replace("ORD-999999", "X")


def test_list_customer_orders_ignores_any_caller_supplied_identity(
    two_customers: tuple[str, str],
) -> None:
    """The tool exposes no customer_id argument, so B's ID cannot be injected."""
    customer_a, customer_b = two_customers
    assert "customer_id" not in list_customer_orders.args

    result = list_customer_orders.invoke({"limit": 5}, config=_cfg(customer_a))
    owned = execute_sql_query_params(
        "SELECT order_id FROM orders WHERE customer_id = %s", (customer_a,)
    )
    owned_ids = {str(row["order_id"]) for row in owned}

    for entry in json.loads(result):
        assert str(entry["order_id"]) in owned_ids


def test_recommendation_tools_take_no_customer_argument() -> None:
    """Whose profile/history is read must not be a model-chosen parameter."""
    for recommendation_tool in (
        get_customer_profile,
        get_customer_order_history,
        recommend_for_customer,
    ):
        assert "customer_id" not in recommendation_tool.args


def test_profile_reads_only_the_signed_in_customer(
    two_customers: tuple[str, str],
) -> None:
    customer_a, customer_b = two_customers
    result = get_customer_profile.invoke({}, config=_cfg(customer_a))
    assert customer_a in result
    assert customer_b not in result


# ══════════════════════════════════════════════════════════════════════════
# Writes across the account boundary
# ══════════════════════════════════════════════════════════════════════════


def test_cancel_order_refuses_another_customers_order(
    two_customers: tuple[str, str], victim_order: str
) -> None:
    customer_a, _ = two_customers
    before = execute_sql_query_params(
        "SELECT status FROM orders WHERE order_id = %s", (victim_order,)
    )
    result = cancel_order.invoke({"order_id": victim_order}, config=_cfg(customer_a))
    after = execute_sql_query_params(
        "SELECT status FROM orders WHERE order_id = %s", (victim_order,)
    )

    assert "couldn't find" in result.lower()
    # The refusal must be a no-op, not just a discouraging message.
    assert before[0]["status"] == after[0]["status"]


def test_return_tools_refuse_another_customers_order_item(
    two_customers: tuple[str, str], victim_order: str
) -> None:
    customer_a, _ = two_customers
    items = execute_sql_query_params(
        "SELECT order_item_id FROM order_items WHERE order_id = %s LIMIT 1",
        (victim_order,),
    )
    if isinstance(items, str) or not items:
        pytest.skip("victim order has no line items")
    item_id = str(items[0]["order_item_id"])

    eligibility = check_return_eligibility.invoke(
        {"order_id": victim_order, "order_item_id": item_id}, config=_cfg(customer_a)
    )
    assert "couldn't find" in eligibility.lower()

    created = create_return_request.invoke(
        {"order_id": victim_order, "order_item_id": item_id, "reason": "changed mind"},
        config=_cfg(customer_a),
    )
    assert "couldn't find" in created.lower()

    # And nothing was written.
    rows = execute_sql_query_params(
        "SELECT return_id FROM returns WHERE order_item_id = %s AND customer_id = %s",
        (item_id, customer_a),
    )
    assert not rows


def test_get_return_status_refuses_another_customers_return(
    two_customers: tuple[str, str],
) -> None:
    customer_a, customer_b = two_customers
    rows = execute_sql_query_params(
        """
        SELECT r.return_id
        FROM returns r
        JOIN orders o ON o.order_id = r.order_id
        WHERE o.customer_id = %s
        LIMIT 1
        """,
        (customer_b,),
    )
    if isinstance(rows, str) or not rows:
        pytest.skip("second customer has no returns")

    result = get_return_status.invoke(
        {"return_id": str(rows[0]["return_id"])}, config=_cfg(customer_a)
    )
    assert "couldn't find" in result.lower()


# ══════════════════════════════════════════════════════════════════════════
# Unauthenticated sessions fail closed
# ══════════════════════════════════════════════════════════════════════════


def test_customer_tools_refuse_when_not_signed_in(own_order: str) -> None:
    unauthenticated = _cfg(None)

    assert (
        get_order_status.invoke({"order_id": own_order}, config=unauthenticated)
        == NOT_AUTHENTICATED_MESSAGE
    )
    assert (
        list_order_items.invoke({"order_id": own_order}, config=unauthenticated)
        == NOT_AUTHENTICATED_MESSAGE
    )
    assert (
        cancel_order.invoke({"order_id": own_order}, config=unauthenticated)
        == NOT_AUTHENTICATED_MESSAGE
    )
    assert (
        list_customer_orders.invoke({"limit": 3}, config=unauthenticated)
        == NOT_AUTHENTICATED_MESSAGE
    )
    assert get_customer_profile.invoke({}, config=unauthenticated) == (
        NOT_AUTHENTICATED_MESSAGE
    )
    assert (
        place_order.invoke(
            {"shipping_address": "1 Test Street, Springfield", "items_json": "[]"},
            config=unauthenticated,
        )
        == NOT_AUTHENTICATED_MESSAGE
    )


def test_tools_fail_closed_when_config_is_missing_entirely(own_order: str) -> None:
    """A caller that forgets to pass config must be denied, not defaulted."""
    assert get_order_status.invoke({"order_id": own_order}) == NOT_AUTHENTICATED_MESSAGE


# ══════════════════════════════════════════════════════════════════════════
# The product agent's raw-SQL escape hatch
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT customer_id, email FROM customers LIMIT 5",
        "SELECT * FROM orders LIMIT 5",
        "SELECT * FROM returns LIMIT 5",
        "SELECT * FROM public.customers LIMIT 5",
        "SELECT p.title FROM product_catalog p JOIN orders o ON TRUE LIMIT 5",
        "WITH x AS (SELECT email FROM customers) SELECT * FROM x",
        "SELECT (SELECT email FROM customers LIMIT 1) AS leak",
        # Comma join — the first version of this guard captured only the first
        # identifier after FROM and let the second table straight through.
        "SELECT c.email FROM product_catalog p, customers c LIMIT 3",
        "SELECT c.email FROM product_catalog AS p, public.customers AS c LIMIT 3",
        "SELECT * FROM customer_sessions LIMIT 5",
        "SELECT * FROM order_items LIMIT 5",
        "SELECT title FROM product_catalog WHERE price > (SELECT AVG(total_amount) FROM orders)",
        "SELECT title FROM product_catalog -- ignore\nUNION SELECT email FROM customers",
    ],
)
def test_query_products_cannot_read_customer_tables(sql: str) -> None:
    """Blocking writes was never enough — this hatch could read anything."""
    result = query_products.invoke({"sql_query": sql})
    assert result.startswith("ERROR:"), f"not blocked: {sql}"
    assert "@" not in result  # no email address made it back


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT main_category, COUNT(*) AS n FROM product_catalog GROUP BY main_category LIMIT 3",
        "SELECT title, price FROM product_catalog WHERE price < 50 LIMIT 3",
        # A product legitimately named after a forbidden table must not trip
        # the token scan — string literals are stripped before scanning.
        "SELECT title FROM product_catalog WHERE title ILIKE '%orders%' LIMIT 3",
    ],
)
def test_query_products_still_answers_catalog_questions(sql: str) -> None:
    result = query_products.invoke({"sql_query": sql})
    assert not result.startswith("ERROR:"), f"wrongly blocked: {sql}"
