"""Tests for resolving a product NAME to an orderable product_id.

Regression for a dead end in the purchase flow: the Order Agent could place an
order only when given a ``product_id``, an internal ASIN the customer has never
seen. Asked to "order the second one" straight after a recommendation, it had
no way to turn the product title into that key, so it asked the customer for
the ID — which they cannot supply.

``find_product`` closes the gap. These tests run against the live catalog,
because the matching behaviour being asserted lives in the SQL.
"""

from __future__ import annotations

import json

import pytest

from src.agents.order_agent.tools import _title_terms, find_product
from src.data.postgresql import execute_sql_query_params


@pytest.fixture(scope="module")
def catalog_product() -> dict:
    """A real catalog product with a substantial title."""
    rows = execute_sql_query_params(
        """
        SELECT product_id, title, price
        FROM product_catalog
        WHERE title IS NOT NULL AND price > 0 AND LENGTH(title) > 40
        ORDER BY rating_count DESC NULLS LAST
        LIMIT 1
        """
    )
    if isinstance(rows, str) or not rows:
        pytest.skip("product catalog unavailable")
    return rows[0]


def _call(name: str, **kwargs) -> dict:
    return json.loads(find_product.invoke({"product_name": name, **kwargs}))


# ══════════════════════════════════════════════════════════════════════════
# Term extraction
# ══════════════════════════════════════════════════════════════════════════


def test_markdown_and_punctuation_are_stripped() -> None:
    """Titles arrive copied out of the agent's own markdown table."""
    terms = _title_terms('**HP 17 Business Laptop** – 17.3" HD+ Display')
    assert "HP" not in terms  # two characters, below the length floor
    assert "Business" in terms
    assert all(term.isalnum() for term in terms)


def test_conversational_filler_is_dropped() -> None:
    """Left in an AND-match, "with"/"please" match no catalog title at all."""
    terms = _title_terms("please order me the second one with the best display")
    for filler in ("please", "order", "the", "second", "one", "with"):
        assert filler not in [t.lower() for t in terms]


def test_category_words_are_kept() -> None:
    """ "laptop" discriminates a laptop from a laptop sleeve — it must survive."""
    assert "laptop" in [t.lower() for t in _title_terms("Dell laptop touchscreen")]


def test_a_single_usable_term_is_still_searched() -> None:
    """Regression: the relaxation floor used to skip the query entirely."""
    result = _call("Dell")
    assert "matches" in result, "single-term name returned no result structure"
    assert result["match_count"] >= 1


# ══════════════════════════════════════════════════════════════════════════
# Resolution
# ══════════════════════════════════════════════════════════════════════════


def test_exact_title_resolves_to_that_product(catalog_product: dict) -> None:
    """The common case: the agent copies a title from its own recommendation."""
    result = _call(str(catalog_product["title"]))
    assert result["match_count"] >= 1
    assert str(catalog_product["product_id"]) in {
        m["product_id"] for m in result["matches"]
    }


def test_result_carries_the_id_and_price_place_order_needs(
    catalog_product: dict,
) -> None:
    result = _call(str(catalog_product["title"]))
    match = result["matches"][0]
    assert match["product_id"]
    assert isinstance(match["price"], float)
    assert match["title"]


def test_partial_name_still_resolves() -> None:
    """A half-remembered "the Dell Inspiron one" must not dead-end."""
    result = _call("Dell Inspiron")
    assert result["match_count"] >= 1
    assert all("dell" in m["title"].lower() for m in result["matches"])


def test_unmatchable_name_returns_zero_not_an_error() -> None:
    result = _call("Nonexistent Frobnicator 9000 Quantum Edition")
    assert result["match_count"] == 0
    assert "product ID" in result["guidance"]  # tells the agent NOT to ask


def test_guidance_forbids_asking_for_an_id_on_every_path(
    catalog_product: dict,
) -> None:
    """Whatever the outcome, the tool must never push the agent to ask."""
    for name in [str(catalog_product["title"]), "laptop computer", "zzzz nothing"]:
        guidance = _call(name).get("guidance", "")
        assert "never" in guidance.lower() or "do not" in guidance.lower()


def test_limit_is_bounded() -> None:
    assert len(_call("laptop computer", limit=999)["matches"]) <= 10
    assert len(_call("laptop computer", limit=2)["matches"]) <= 2


def test_empty_name_is_rejected_gracefully() -> None:
    assert (
        "provide the product name"
        in find_product.invoke({"product_name": "   "}).lower()
    )


@pytest.mark.parametrize("reference", ["the second one", "that one", "it please"])
def test_bare_reference_sends_the_agent_back_to_the_conversation(
    reference: str,
) -> None:
    """ "the second one" is answerable from context, never by asking the customer."""
    response = find_product.invoke({"product_name": reference}).lower()
    assert "conversation" in response
    assert "do not ask the customer" in response


def test_matching_relaxes_only_down_to_a_floor() -> None:
    """Relaxing too far would match an unrelated product and look confident.

    A long title whose later terms are wrong should still resolve on its
    leading terms, and the tool reports which terms it actually matched on so
    a wrong resolution is diagnosable.
    """
    result = _call("Dell Inspiron nonsensetokenxyz anothernonsensetoken")
    if result["match_count"]:
        assert len(result["matched_on"]) >= 2
        assert "nonsensetokenxyz" not in result["matched_on"]


# ══════════════════════════════════════════════════════════════════════════
# Agent wiring
# ══════════════════════════════════════════════════════════════════════════


def test_order_agent_exposes_the_lookup_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """place_order is unusable without it; the two must ship together.

    The LLM provider is stubbed so this asserts the wiring without needing an
    API key or a network call.
    """
    import src.agents.base_agent as base_agent

    monkeypatch.setattr(base_agent, "get_llm", lambda: object())
    monkeypatch.setattr(base_agent, "create_react_agent", lambda **kwargs: object())

    from src.agents.order_agent.agent import OrderAgent

    tool_names = {getattr(t, "name", "") for t in OrderAgent()._tools}
    assert "find_product" in tool_names
    assert "place_order" in tool_names


def test_find_product_takes_no_customer_identifier() -> None:
    """The catalog is public; this tool must not become customer-scoped."""
    assert "customer_id" not in find_product.args
