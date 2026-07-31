"""Regression tests for Unicode punctuation breaking identifier parsing.

Asked "find my latest order and return it", the system resolved the order, said
"your most recent order is ORD‑006762" — with U+2011 NON-BREAKING HYPHEN — and
then asked the customer for the order id it had just supplied.

The routes were correct and the graph ran them in the right order. The failure
was that ``_extract_facts`` matched ``\\bORD-\\d{6}\\b`` with an ASCII hyphen
against text containing U+2011, extracted nothing, and handed the return agent
an empty fact set.

The same look-alike character breaks identifier handling in three places, all
covered below.
"""

from __future__ import annotations

import pytest

from src.agents.graph.nodes import _extract_facts
from src.agents.router import RouterAgent
from src.utils.text import normalise_typography

#: The exact character models emit in place of a hyphen.
NB_HYPHEN = "‑"


def test_the_character_really_is_different() -> None:
    """Guards against someone 'simplifying' these tests with a plain hyphen."""
    assert NB_HYPHEN != "-"
    assert f"ORD{NB_HYPHEN}006762" != "ORD-006762"


# ══════════════════════════════════════════════════════════════════════════
# The shared helper
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ORD‑006762", "ORD-006762"),  # non-breaking hyphen
        ("ORD‐006762", "ORD-006762"),  # hyphen
        ("ORD–006762", "ORD-006762"),  # en dash
        ("ORD—006762", "ORD-006762"),  # em dash
        ("ORD−006762", "ORD-006762"),  # minus sign
    ],
)
def test_dash_variants_fold_to_ascii(raw: str, expected: str) -> None:
    assert normalise_typography(raw) == expected


def test_smart_quotes_and_spaces_fold() -> None:
    assert normalise_typography("can’t") == "can't"
    assert normalise_typography("“quoted”") == '"quoted"'
    assert normalise_typography("a b") == "a b"  # narrow no-break space


def test_thousands_separators_only_stripped_on_request() -> None:
    assert normalise_typography("$1,234.56") == "$1,234.56"
    assert normalise_typography("$1,234.56", strip_thousands=True) == "$1234.56"


def test_ordinary_text_is_untouched() -> None:
    plain = "Order ORD-000055 was delivered on 2022-12-08 for $178.44."
    assert normalise_typography(plain) == plain


# ══════════════════════════════════════════════════════════════════════════
# Fact hand-off between specialists — the reported bug
# ══════════════════════════════════════════════════════════════════════════


def test_order_id_with_unicode_hyphen_is_extracted() -> None:
    """The exact sentence that broke the hand-off."""
    answer = (
        f"Your most recent order is **ORD{NB_HYPHEN}006762**, placed on "
        "February 24 2025."
    )
    assert _extract_facts(answer) == {"order_id": "ORD-006762"}


def test_return_id_with_unicode_hyphen_is_extracted() -> None:
    assert _extract_facts(f"Return RET{NB_HYPHEN}000042 is pending.") == {
        "return_id": "RET-000042"
    }


def test_ascii_identifiers_still_extract() -> None:
    assert _extract_facts("Order ORD-006762 is on its way.") == {
        "order_id": "ORD-006762"
    }


def test_extracted_fact_is_normalised_not_merely_found() -> None:
    """The value passed on must be usable by the next agent's tools.

    Carrying the U+2011 form forward would simply move the failure downstream
    into the authorization check.
    """
    fact = _extract_facts(f"Order ORD{NB_HYPHEN}006762.")["order_id"]
    assert "-" in fact
    assert NB_HYPHEN not in fact


def test_no_identifier_yields_no_facts() -> None:
    assert _extract_facts("I couldn't find any orders on your account.") == {}


# ══════════════════════════════════════════════════════════════════════════
# Authorization — the model passing back an id it read earlier
# ══════════════════════════════════════════════════════════════════════════


def test_authz_accepts_an_identifier_carrying_a_unicode_hyphen() -> None:
    """A valid order number must not be rejected as malformed.

    Uses an unauthenticated config so the assertion is about the *format*
    check, with no database access.
    """
    from src.agents.authz import NOT_AUTHENTICATED_MESSAGE, authorize_order

    _, error = authorize_order(f"ORD{NB_HYPHEN}000055", {"configurable": {}})
    # Fails on authentication, NOT on format — which is the point.
    assert error == NOT_AUTHENTICATED_MESSAGE
    assert "format" not in error.lower()


def test_authz_still_rejects_genuinely_malformed_identifiers() -> None:
    from src.agents.authz import authorize_order

    _, error = authorize_order(
        "ORD-ABCDEF", {"configurable": {"customer_id": "CUST-A"}}
    )
    assert "format" in error.lower()


# ══════════════════════════════════════════════════════════════════════════
# Router fast path — a customer pasting an id copied out of the chat
# ══════════════════════════════════════════════════════════════════════════


def test_router_fastpath_matches_a_pasted_identifier() -> None:
    router = RouterAgent(use_llm_fallback=False)
    assert router.route(f"Where is my order ORD{NB_HYPHEN}000123?") == "order"


def test_router_fastpath_matches_a_pasted_return_identifier() -> None:
    router = RouterAgent(use_llm_fallback=False)
    assert router.route(f"What is the status of RET{NB_HYPHEN}000042?") == "return"
