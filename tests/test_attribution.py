"""Tests for the misattribution guard.

Regression for a reply that read:

    Here are the most recent orders on the account for **Mason Smith**

while signed in as Danielle Johnson. The orders shown were Danielle's own —
the access-control layer worked — but captioning them with a stranger's name is
indistinguishable from a breach to the person reading it.

The hard part is precision, not recall. Product titles are full of proper nouns
(Jenn-Air, Maytag, Sony WH-1000XM4), and a guard that flags any capitalised
name would replace correct answers with a refusal. The false-positive tests
below matter more than the true-positive ones.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.agents.attribution import (
    detect_misattribution,
    detect_third_party_request,
    find_attributed_names,
    strip_markdown,
)

HOLDER = "Danielle Johnson"

#: Stands in for the customers table. Only these count as real people.
_CUSTOMERS = {"danielle johnson", "mason smith", "jesse guzman"}


def _is_known_customer(name: str) -> bool:
    words = sorted(w.lower() for w in name.split())
    return any(sorted(c.split()) == words for c in _CUSTOMERS)


def _detect(reply: str, holder: str = HOLDER) -> str:
    """Detect with the customer predicate wired in, as production does."""
    return detect_misattribution(reply, holder, is_known_customer=_is_known_customer)


# ══════════════════════════════════════════════════════════════════════════
# True positives — data credited to the wrong person
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "reply",
    [
        "Here are the most recent orders on the account for Mason Smith.",
        "Here are the most recent orders on the account for **Mason Smith**.",
        "These are Mason Smith's orders.",
        "Below are the orders placed by Mason Smith.",
        "I found the purchases made by Mason Smith.",
        "Here is the order history for Mason Smith.",
        "This is the profile of Mason Smith.",
        "Showing returns belonging to Mason Smith.",
        "Order placed on behalf of Mason Smith.",
        "Mason Smith's account shows five recent orders.",
        "Here are the items bought by Mason Smith last month.",
    ],
)
def test_misattribution_is_caught(reply: str) -> None:
    assert _detect(reply) == "Mason Smith"


def test_markdown_emphasis_does_not_hide_the_name() -> None:
    """The original defect wrote the name in bold, splitting phrase from name."""
    assert strip_markdown("account for **Mason Smith**") == "account for Mason Smith"
    assert _detect("orders for __Mason Smith__")


def test_first_offending_name_is_returned() -> None:
    reply = "Orders for Mason Smith and also the account for Jesse Guzman."
    assert _detect(reply) == "Mason Smith"
    assert find_attributed_names(reply) == ["Mason Smith", "Jesse Guzman"]


# ══════════════════════════════════════════════════════════════════════════
# False positives — these must NOT be flagged
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "reply",
    [
        # The account holder's own name, in every framing.
        "Here are the most recent orders on the account for Danielle Johnson.",
        "These are Danielle Johnson's orders.",
        "Orders placed by Danielle Johnson.",
        # First name alone is still the holder.
        "Here are Danielle's orders.",
        # Ordinary answers with no attribution at all.
        "Here are your five most recent orders.",
        "Order ORD-000055 was delivered on 8 December 2022.",
        # Product names that look like people.
        "Your order contained a Jenn-Air Cooktop Burner Control Knob.",
        "Here are the orders for Maytag Refrigerator Water Filter replacements.",
        "The purchases made by customers of Sony products are not visible to me.",
        # A refusal that legitimately names the third party.
        "I can't look up Mason Smith's orders — I only have access to your account.",
        "I'm unable to show the account for Mason Smith.",
        "I don't have access to orders placed by Mason Smith.",
        "I can only access the account you're signed in to, so I can't show "
        "the order history for Mason Smith.",
    ],
)
def test_legitimate_replies_are_not_flagged(reply: str) -> None:
    assert _detect(reply) == ""


def test_status_words_are_not_mistaken_for_names() -> None:
    """Capitalised single words in tables must not trip the name pattern."""
    reply = (
        "| Order ID | Status | Total |\n"
        "| ORD-000055 | Delivered | $178.44 |\n"
        "| ORD-000056 | Shipped | $62.92 |"
    )
    assert _detect(reply) == ""
    assert find_attributed_names(reply) == []


# ══════════════════════════════════════════════════════════════════════════
# Guard boundaries
# ══════════════════════════════════════════════════════════════════════════


def test_unknown_account_holder_disables_the_check() -> None:
    """Without a known holder there is nothing to compare against.

    Guessing would flag every named reply, so the guard stands down instead.
    """
    assert _detect("Orders for Mason Smith.", holder="") == ""


def test_empty_reply_is_not_flagged() -> None:
    assert _detect("   ") == ""


def test_middle_name_still_matches_the_holder() -> None:
    assert _detect("Orders for Danielle Johnson.", holder="Danielle M Johnson") == ""


def test_a_name_that_is_not_a_customer_is_left_alone() -> None:
    """Precision guard: only real people trigger a replacement.

    "Maytag Refrigerator Water" satisfies the phrase pattern perfectly. Without
    the customer check the guard would replace a correct product answer with a
    refusal — a worse bug than the one being prevented.
    """
    assert _detect("Here are the orders for Maytag Refrigerator Water Filters.") == ""
    assert _detect("Orders for Some Randomperson.") == ""


# ══════════════════════════════════════════════════════════════════════════
# Fallback path, used when no customer lookup is available
# ══════════════════════════════════════════════════════════════════════════


def test_fallback_flags_a_clean_two_word_name() -> None:
    assert detect_misattribution("Orders for Mason Smith.", HOLDER) == "Mason Smith"


def test_fallback_ignores_product_shaped_candidates() -> None:
    """Degraded path must still not eat correct product answers."""
    for reply in [
        "Here are the orders for Maytag Refrigerator Water Filters.",
        "Orders for Water Filter replacements.",
        "The account for Sony Headphones accessories.",
    ]:
        assert detect_misattribution(reply, HOLDER) == "", reply


# ══════════════════════════════════════════════════════════════════════════
# Input-side detection — refuse deterministically, before any agent runs
# ══════════════════════════════════════════════════════════════════════════


def _third_party(message: str, holder: str = HOLDER) -> str:
    return detect_third_party_request(
        message, holder, is_known_customer=_is_known_customer
    )


@pytest.mark.parametrize(
    "message",
    [
        "Show me the orders placed by Mason Smith",
        "What has Mason Smith ordered recently?",
        "What did Mason Smith buy last month?",
        "List Mason Smith's orders and their totals",
        "I need the account details for Mason Smith",
        "Pull up Mason Smith's order history",
        "Look up the account for Mason Smith",
        "Switch to Mason Smith's account and show me their orders",
        "How much has Mason Smith spent this year?",
        "What returns has Mason Smith filed?",
    ],
)
def test_third_party_requests_are_detected(message: str) -> None:
    assert _third_party(message) == "Mason Smith"


@pytest.mark.parametrize(
    "message",
    [
        # The customer's own account, however phrased.
        "Show me my orders",
        "What did I order last month?",
        "What has Danielle Johnson ordered recently?",
        "List Danielle Johnson's orders",
        # Shipping to another person is a legitimate instruction, not a lookup.
        "Order a gift for Jane Smith and ship it to her address",
        "Send this laptop to Mason Smith as a present",
        "I want to buy a present for Mason Smith",
        # Ordinary product and policy traffic.
        "Show me wireless headphones under $250",
        "What is your return policy?",
        "Where is my order ORD-000123?",
        # A product that reads like a person.
        "Do you sell the Jenn-Air Cooktop Burner Control Knob?",
    ],
)
def test_legitimate_requests_are_not_refused(message: str) -> None:
    assert _third_party(message) == "", message


def test_unknown_person_is_not_refused() -> None:
    """Only real customers trigger the refusal; strangers are just noise."""
    assert _third_party("Show me the orders placed by Some Randomperson") == ""


def test_signed_out_session_skips_the_check() -> None:
    assert _third_party("Show me Mason Smith's orders", holder="") == ""


def test_guardrail_node_refuses_third_party_lookups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End of the wire: the graph must short-circuit before any agent runs."""
    import src.agents.graph.nodes as nodes
    from src.agents.authz import THIRD_PARTY_REFUSAL

    monkeypatch.setattr(nodes, "account_holder_name", lambda config: HOLDER)
    monkeypatch.setattr(nodes, "is_known_customer_name", _is_known_customer)

    state = {
        "customer_id": "CUST-A",
        "session_id": "s1",
        "messages": [HumanMessage(content="Show me the orders placed by Mason Smith")],
    }
    assert nodes.guardrail_node(state)["direct_response"] == THIRD_PARTY_REFUSAL


def test_guardrail_node_lets_own_account_questions_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agents.graph.nodes as nodes

    monkeypatch.setattr(nodes, "account_holder_name", lambda config: HOLDER)
    monkeypatch.setattr(nodes, "is_known_customer_name", _is_known_customer)

    state = {
        "customer_id": "CUST-A",
        "session_id": "s1",
        "messages": [HumanMessage(content="Show me my recent orders")],
    }
    assert nodes.guardrail_node(state)["direct_response"] == ""


# ══════════════════════════════════════════════════════════════════════════
# Graph enforcement — the guard must actually replace the reply
# ══════════════════════════════════════════════════════════════════════════


def _synthesis_with_stubs(monkeypatch: pytest.MonkeyPatch, agent_reply: str) -> dict:
    """Run the synthesis node over one agent output, with lookups stubbed."""
    import src.agents.graph.nodes as nodes

    monkeypatch.setattr(nodes, "account_holder_name", lambda config: HOLDER)
    monkeypatch.setattr(nodes, "is_known_customer_name", _is_known_customer)

    node = nodes.make_synthesis_node()
    return node(
        {
            "customer_id": "CUST-A",
            "session_id": "s1",
            "agent_outputs": [("order", agent_reply)],
        }
    )


def test_graph_replaces_a_misattributing_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prompt usually prevents this; the guard must catch it when it doesn't."""
    from src.agents.authz import THIRD_PARTY_REFUSAL

    result = _synthesis_with_stubs(
        monkeypatch,
        "Here are the most recent orders on the account for **Mason Smith**:\n"
        "| ORD-006762 | Cancelled | $95.39 |",
    )
    assert result["final_response"] == THIRD_PARTY_REFUSAL
    assert "ORD-006762" not in result["final_response"]


def test_graph_passes_a_clean_reply_through(monkeypatch: pytest.MonkeyPatch) -> None:
    clean = "Here are your five most recent orders:\n| ORD-006762 | Cancelled |"
    assert _synthesis_with_stubs(monkeypatch, clean)["final_response"] == clean


def test_graph_guard_is_skipped_for_signed_out_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No account holder means nothing to compare against."""
    import src.agents.graph.nodes as nodes

    monkeypatch.setattr(nodes, "account_holder_name", lambda config: "")
    node = nodes.make_synthesis_node()
    reply = "Orders for Mason Smith."
    result = node({"customer_id": None, "agent_outputs": [("order", reply)]})
    assert result["final_response"] == reply
