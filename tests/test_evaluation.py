"""Tests for the evaluation harness itself.

A scoring harness that is wrong produces confident, plausible numbers — which
is worse than no harness at all, because nobody re-checks a number that looks
reasonable. These tests pin the scoring semantics: what counts as a pass, what
a refusal looks like, and that the hallucination guard actually fires.

They use stub agents throughout: no LLM calls, no database.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.checks import run_checks
from src.evaluation.fixtures import FixtureError, Fixtures, required_placeholders
from src.evaluation.loader import AGENT_DATASETS, DATASET_DIR, load_raw_cases
from src.evaluation.report import build_json_payload, format_console, format_markdown
from src.evaluation.routing import (
    RoutingCase,
    evaluate_routing,
    load_routing_cases,
)
from src.evaluation.runner import run_agent_cases, run_case
from src.evaluation.schema import AgentReport, CaseResult, Checks, EvalCase


# ══════════════════════════════════════════════════════════════════════════
# Check primitives
# ══════════════════════════════════════════════════════════════════════════


def test_all_checks_pass_on_a_correct_answer() -> None:
    checks = Checks(
        expect_tools=("list_order_items",),
        must_contain=("ORD-000055",),
        must_not_contain=("ORD-006041",),
        allowed_ids=("ORD-000055",),
    )
    results = run_checks(checks, "Order ORD-000055 contained 2 items.", ["list_order_items"])
    assert all(r.passed for r in results)


def test_missing_expected_tool_fails() -> None:
    results = run_checks(Checks(expect_tools=("list_order_items",)), "Sure!", ["track_order"])
    failed = [r for r in results if not r.passed]
    assert [r.name for r in failed] == ["calls:list_order_items"]
    assert "track_order" in failed[0].detail


def test_forbidden_tool_fails() -> None:
    results = run_checks(Checks(forbid_tools=("cancel_order",)), "Cancelled.", ["cancel_order"])
    assert any(not r.passed and r.name == "never_calls:cancel_order" for r in results)


def test_expect_any_tools_passes_when_one_matches() -> None:
    checks = Checks(expect_any_tools=("search_products", "query_products"))
    results = run_checks(checks, "Here are results.", ["query_products"])
    assert all(r.passed for r in results)


def test_expect_any_tools_fails_when_none_match() -> None:
    checks = Checks(expect_any_tools=("search_products", "query_products"))
    results = run_checks(checks, "Here are results.", ["get_order_status"])
    assert any(not r.passed for r in results)


def test_leaked_substring_fails() -> None:
    checks = Checks(must_not_contain=("ORD-006041",))
    results = run_checks(checks, "Order ORD-006041 contained a fridge.", [])
    leak = next(r for r in results if r.name.startswith("excludes:"))
    assert not leak.passed
    assert "LEAKED" in leak.detail


@pytest.mark.parametrize(
    "answer",
    [
        "I couldn't find order ORD-999999 on your account.",
        "I'm sorry, but I can't locate that order.",
        "That order is not on your account.",
        "I'm unable to access those details.",
        "Please sign in and try again.",
    ],
)
def test_refusal_detection_accepts_natural_phrasings(answer: str) -> None:
    """Agents phrase refusals freely; the detector must not demand one wording."""
    results = run_checks(Checks(expect_refusal=True), answer, [])
    assert all(r.passed for r in results), answer


def test_refusal_detection_rejects_a_compliant_answer() -> None:
    results = run_checks(
        Checks(expect_refusal=True), "Order ORD-000055 contained 2 items totalling $178.", []
    )
    assert any(not r.passed and r.name == "is_refusal" for r in results)


def test_invented_identifier_is_caught() -> None:
    """A fabricated order number must fail even when everything else looks right."""
    checks = Checks(allowed_ids=("ORD-000055",))
    results = run_checks(checks, "Your order ORD-123456 is on its way.", [])
    guard = next(r for r in results if r.name == "no_invented_ids")
    assert not guard.passed
    assert "ORD-123456" in guard.detail


def test_allowed_identifier_is_not_flagged() -> None:
    checks = Checks(allowed_ids=("ORD-000055",))
    results = run_checks(checks, "Your order ORD-000055 is on its way.", [])
    assert all(r.passed for r in results)


def test_typographic_variation_does_not_fail_a_correct_answer() -> None:
    """Models emit non-breaking hyphens; that must not read as a wrong answer."""
    checks = Checks(must_contain=("ORD-000055", "1234.56"))
    results = run_checks(checks, "Order ORD‑000055 came to $1,234.56.", [])
    assert all(r.passed for r in results), [r.name for r in results if not r.passed]


def test_regex_checks_tolerate_curly_apostrophes() -> None:
    """Patterns are written with ', models emit ’ — that is typography, not error."""
    checks = Checks(must_match=(r"couldn't find",))
    results = run_checks(checks, "I’m sorry, but I couldn’t find that product.", [])
    assert all(r.passed for r in results), [r.name for r in results if not r.passed]


def test_empty_answer_always_fails() -> None:
    results = run_checks(Checks(), "   ", [])
    assert any(not r.passed and r.name == "non_empty_answer" for r in results)


# ══════════════════════════════════════════════════════════════════════════
# Scoring semantics
# ══════════════════════════════════════════════════════════════════════════


def _case(case_id: str = "c1", **kwargs) -> EvalCase:
    return EvalCase(id=case_id, agent="order", query="q", checks=Checks(**kwargs))


def test_case_passes_only_when_every_check_passes() -> None:
    """Strict by design: a partially correct answer is not a pass."""
    result = CaseResult(case=_case())
    result.checks = run_checks(
        Checks(must_contain=("alpha",), must_not_contain=("beta",)), "alpha and beta", []
    )
    assert result.checks_passed == 2  # non_empty + contains
    assert not result.passed


def test_errored_case_never_passes() -> None:
    result = CaseResult(case=_case(), error="boom")
    result.checks = run_checks(Checks(), "fine", [])
    assert not result.passed


def test_skipped_cases_are_excluded_from_the_denominator() -> None:
    """A skipped case must not silently inflate or deflate the pass rate."""
    passing = CaseResult(case=_case("ok"))
    passing.checks = run_checks(Checks(), "answer", [])
    skipped = CaseResult(case=_case("skip"), skipped="mutates the database")

    report = AgentReport(agent="order", results=[passing, skipped])
    assert report.total == 1
    assert report.passed == 1
    assert report.pass_rate == 1.0
    assert len(report.skipped) == 1


@pytest.mark.parametrize(
    "error",
    [
        "RateLimitError: Error code: 429 - tokens per day (TPD): Limit 200000",
        "APIConnectionError: Connection error.",
        "APIStatusError: Error code: 413 - rate_limit_exceeded",
        "TimeoutError: request timed out",
    ],
)
def test_provider_failures_are_unscored_not_failed(error: str) -> None:
    """An exhausted quota must not read as a low agent score."""
    result = CaseResult(case=_case(), error=error)
    assert result.infrastructure_error

    report = AgentReport(agent="escalation", results=[result])
    assert report.total == 0        # excluded from the denominator
    assert report.pass_rate == 0.0  # no evidence either way
    assert len(report.unscored) == 1


def test_genuine_agent_errors_still_count_as_failures() -> None:
    """A TypeError in the agent is a real failure, not an infrastructure blip."""
    result = CaseResult(
        case=_case(), error="TypeError: run() got an unexpected keyword argument"
    )
    assert not result.infrastructure_error

    report = AgentReport(agent="fallback", results=[result])
    assert report.total == 1
    assert report.passed == 0
    assert not report.unscored


def test_unscored_cases_are_surfaced_in_reports() -> None:
    """A quota-starved run must be unmistakable, not silently flattering."""
    report = AgentReport(
        agent="escalation",
        results=[CaseResult(case=_case("e1"), error="RateLimitError: 429 quota")],
    )
    console = format_console([report])
    assert "could NOT be scored" in console
    markdown = format_markdown([report])
    assert "could not be scored" in markdown
    assert build_json_payload([report])["agents"]["escalation"][
        "unscored_provider_errors"
    ] == 1


def test_check_score_reports_partial_credit_separately() -> None:
    result = CaseResult(case=_case())
    result.checks = run_checks(Checks(must_contain=("alpha", "gamma")), "alpha only", [])
    report = AgentReport(agent="order", results=[result])
    assert report.pass_rate == 0.0          # the case failed outright
    assert 0.0 < report.check_score < 1.0   # but two of three checks passed


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════


class _StubResult:
    def __init__(self, text: str, tool_calls: list[str], error: str = "") -> None:
        self.text = text
        self.tool_calls = tool_calls
        self.error = error


class _StubAgent:
    def __init__(self, text: str = "ok", tool_calls: list[str] | None = None) -> None:
        self.text = text
        self.tool_calls = tool_calls or []
        self.seen_customer_ids: list[str | None] = []

    def run(self, messages, customer_id=None, **kwargs):
        self.seen_customer_ids.append(customer_id)
        return _StubResult(self.text, self.tool_calls)


class _ExplodingAgent:
    def run(self, messages, customer_id=None, **kwargs):
        raise RuntimeError("provider exploded")


def test_runner_passes_case_identity_to_the_agent() -> None:
    """The case's customer_id must reach the agent, or authorization cases are meaningless."""
    agent = _StubAgent()
    case = EvalCase(id="c", agent="order", query="q", checks=Checks(), customer_id="CUST-A")
    run_case(agent, case)
    assert agent.seen_customer_ids == ["CUST-A"]


def test_runner_records_a_crash_as_a_failed_case_not_a_crashed_run() -> None:
    result = run_case(_ExplodingAgent(), _case())
    assert not result.passed
    assert "provider exploded" in result.error


def test_mutating_cases_are_skipped_without_allow_writes() -> None:
    agent = _StubAgent()
    case = EvalCase(id="w", agent="order", query="q", checks=Checks(), mutates=True)
    report = run_agent_cases("order", agent, [case], allow_writes=False, progress=False)
    assert report.total == 0
    assert "allow-writes" in report.skipped[0].skipped
    assert agent.seen_customer_ids == []  # the agent was never invoked


def test_mutating_cases_run_with_allow_writes() -> None:
    agent = _StubAgent()
    case = EvalCase(id="w", agent="order", query="q", checks=Checks(), mutates=True)
    report = run_agent_cases("order", agent, [case], allow_writes=True, progress=False)
    assert report.total == 1
    assert agent.seen_customer_ids == [None]


def test_unavailable_cases_surface_as_skipped() -> None:
    report = run_agent_cases(
        "return", _StubAgent(), [], unavailable=[("c9", "no data for {OWN_RETURN}")],
        progress=False,
    )
    assert report.total == 0
    assert report.skipped[0].case.id == "c9"


# ══════════════════════════════════════════════════════════════════════════
# Fixtures and datasets
# ══════════════════════════════════════════════════════════════════════════


def test_fixture_substitution() -> None:
    fixtures = Fixtures(values={"OWN_ORDER": "ORD-000055"})
    assert fixtures.resolve("order {OWN_ORDER} please") == "order ORD-000055 please"


def test_unknown_placeholder_raises_rather_than_silently_passing_through() -> None:
    fixtures = Fixtures(values={"OWN_ORDER": "ORD-000055"})
    with pytest.raises(FixtureError, match="NOPE"):
        fixtures.resolve("{NOPE}")


def test_required_placeholders_extraction() -> None:
    assert required_placeholders("{A} and {B}") == {"A", "B"}


@pytest.mark.parametrize("agent,filename", sorted(AGENT_DATASETS.items()))
def test_every_dataset_is_valid_and_self_consistent(agent: str, filename: str) -> None:
    """Datasets are data, so they get validated like data."""
    raw_cases = load_raw_cases(DATASET_DIR / filename)
    assert raw_cases, f"{filename} has no cases"

    seen: set[str] = set()
    for raw in raw_cases:
        case_id = raw.get("id")
        assert case_id, f"{filename}: a case is missing an id"
        assert case_id not in seen, f"{filename}: duplicate case id {case_id}"
        seen.add(case_id)
        assert raw.get("agent") == agent, f"{filename}: {case_id} declares agent {raw.get('agent')}"
        assert raw.get("query"), f"{filename}: {case_id} has no query"
        assert raw.get("description"), f"{filename}: {case_id} has no description"
        # Parsing must not raise.
        EvalCase.from_dict({**raw, "customer_id": raw.get("customer_id") or None})


def test_datasets_cover_the_security_dimension() -> None:
    """The cross-account rules must stay represented in the ground truth."""
    tagged = 0
    for filename in AGENT_DATASETS.values():
        for raw in load_raw_cases(DATASET_DIR / filename):
            if "security" in (raw.get("tags") or []):
                tagged += 1
    assert tagged >= 8, f"only {tagged} security cases; the authz regressions are under-covered"


def test_every_registered_agent_has_a_dataset_file() -> None:
    for filename in AGENT_DATASETS.values():
        assert (DATASET_DIR / filename).exists(), f"missing dataset {filename}"


# ══════════════════════════════════════════════════════════════════════════
# Routing
# ══════════════════════════════════════════════════════════════════════════


class _StubRouter:
    def __init__(self, routes: list[str]) -> None:
        self.routes = routes

    def classify_multi(self, user_message: str, history: str = "") -> dict:
        return {"routes": list(self.routes), "confidences": {}, "reason": "stub"}


def test_routing_dataset_is_valid_and_covers_every_route() -> None:
    """The dataset is data, so it gets validated like data."""
    from src.agents.router.schemas import ROUTE_LABELS

    cases = load_routing_cases()
    assert cases, "routing dataset is empty"

    covered: set[str] = set()
    for case in cases:
        assert case.description, f"{case.id} has no description"
        for route in (*case.expect_all, *case.expect_any, *case.forbid):
            assert route in ROUTE_LABELS, f"{case.id} names unknown route {route}"
        covered |= set(case.expect_all)

    assert covered == set(ROUTE_LABELS), (
        f"routing dataset does not exercise every route; missing "
        f"{sorted(set(ROUTE_LABELS) - covered)}"
    )


def test_routing_dataset_includes_multi_intent_cases() -> None:
    multi = [c for c in load_routing_cases() if c.is_multi_intent]
    assert len(multi) >= 4, "too few multi-intent routing cases"


def test_routing_case_must_assert_something() -> None:
    with pytest.raises(ValueError, match="asserts nothing"):
        RoutingCase.from_dict({"id": "x", "query": "q"})


def _routing_case(**kwargs) -> RoutingCase:
    defaults = {"id": "c1", "query": "q", "description": "d"}
    return RoutingCase.from_dict({**defaults, **kwargs})


def test_single_route_scored_correctly() -> None:
    report = evaluate_routing(
        _StubRouter(["order"]), [_routing_case(expect_all=["order"])], progress=False
    )
    assert report.accuracy == 1.0


def test_wrong_route_is_a_miss() -> None:
    report = evaluate_routing(
        _StubRouter(["product"]), [_routing_case(expect_all=["order"])], progress=False
    )
    assert report.accuracy == 0.0
    assert report.scored[0].missing == ["order"]


def test_extra_route_is_tolerated_but_recorded() -> None:
    """A second route can be a fair reading, but must not go unnoticed."""
    report = evaluate_routing(
        _StubRouter(["order", "product"]), [_routing_case(expect_all=["order"])], progress=False
    )
    assert report.accuracy == 1.0
    assert report.scored[0].extra == ["product"]
    assert report.over_routing_rate == 1.0


def test_multi_intent_requires_every_route() -> None:
    cases = [_routing_case(expect_all=["return", "order"])]
    assert evaluate_routing(_StubRouter(["return", "order"]), cases, progress=False).accuracy == 1.0
    # Getting only one of the two is a miss, not a half-pass.
    partial = evaluate_routing(_StubRouter(["return"]), cases, progress=False)
    assert partial.accuracy == 0.0
    assert partial.scored[0].missing == ["order"]


def test_forbidden_route_fails_the_case() -> None:
    report = evaluate_routing(
        _StubRouter(["order", "escalation"]),
        [_routing_case(expect_all=["order"], forbid=["escalation"])],
        progress=False,
    )
    assert report.accuracy == 0.0
    assert report.scored[0].forbidden_hit == ["escalation"]


def test_expect_any_accepts_either_option() -> None:
    cases = [_routing_case(expect_any=["escalation", "fallback"])]
    assert evaluate_routing(_StubRouter(["fallback"]), cases, progress=False).accuracy == 1.0
    assert evaluate_routing(_StubRouter(["escalation"]), cases, progress=False).accuracy == 1.0
    assert evaluate_routing(_StubRouter(["product"]), cases, progress=False).accuracy == 0.0


def test_single_and_multi_intent_are_reported_separately() -> None:
    report = evaluate_routing(
        _StubRouter(["order"]),
        [
            _routing_case(id="single", expect_all=["order"]),
            _routing_case(id="multi", expect_all=["order", "product"]),
        ],
        progress=False,
    )
    assert report.single_intent == (1.0, 1)
    assert report.multi_intent == (0.0, 1)


def test_router_errors_are_tracked_and_excluded() -> None:
    class _BrokenRouter:
        def classify_multi(self, user_message: str, history: str = "") -> dict:
            raise RuntimeError("json_validate_failed")

    report = evaluate_routing(
        _BrokenRouter(), [_routing_case(expect_all=["order"])], progress=False
    )
    assert len(report.errors) == 1
    assert report.accuracy == 0.0  # no scorable cases, not a false 100%


def test_per_route_metrics_exclude_ambiguous_cases() -> None:
    """Crediting a route for a choice among acceptable answers is meaningless."""
    report = evaluate_routing(
        _StubRouter(["order"]),
        [
            _routing_case(id="gold", expect_all=["order"]),
            _routing_case(id="ambiguous", expect_any=["order", "fallback"]),
        ],
        progress=False,
    )
    assert report.per_route_metrics()["order"]["support"] == 1.0


def test_per_route_metrics_and_confusion_are_computed() -> None:
    report = evaluate_routing(
        _StubRouter(["product"]),
        [_routing_case(id="a", expect_all=["order"]), _routing_case(id="b", expect_all=["product"])],
        progress=False,
    )
    metrics = report.per_route_metrics()
    assert metrics["order"]["recall"] == pytest.approx(0.0)
    assert metrics["product"]["recall"] == pytest.approx(1.0)
    assert metrics["product"]["precision"] == pytest.approx(0.5)
    assert report.confusion()["order"]["product"] == 1


# ══════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════


def _sample_report() -> AgentReport:
    good = CaseResult(case=_case("good"), answer="ORD-000055 has 2 items", tools_used=["t"])
    good.checks = run_checks(Checks(must_contain=("ORD-000055",)), good.answer, ["t"])
    bad = CaseResult(case=_case("bad"), answer="no idea", tools_used=[])
    bad.checks = run_checks(Checks(must_contain=("ORD-000055",)), bad.answer, [])
    return AgentReport(agent="order", results=[good, bad])


def test_console_report_renders_without_error() -> None:
    text = format_console([_sample_report()])
    assert "AGENT EVALUATION" in text
    assert "order" in text
    assert "bad" in text  # the failing case is named


def test_markdown_report_renders_without_error() -> None:
    markdown = format_markdown([_sample_report()])
    assert markdown.startswith("# Agent Evaluation Report")
    assert "| order |" in markdown


def test_json_payload_is_serialisable_and_carries_scores() -> None:
    payload = build_json_payload([_sample_report()])
    encoded = json.dumps(payload)  # must not raise
    assert '"pass_rate"' in encoded
    assert payload["agents"]["order"]["cases"] == 2
    assert payload["overall"]["passed"] == 1


def test_report_directory_is_not_created_as_a_side_effect_of_formatting(tmp_path: Path) -> None:
    """Formatting is pure; only write_reports touches the filesystem."""
    format_console([_sample_report()])
    format_markdown([_sample_report()])
    assert not (tmp_path / "output").exists()
