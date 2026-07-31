"""Tests for answer reuse and the consolidated summary.

The cache exists to keep a full evaluation affordable, but a cache that returns
a stale answer for a changed question would silently report a score for a test
that was never run. These tests pin the invalidation rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.cache import AnswerCache, code_fingerprint, prompt_hash
from src.evaluation.runner import run_agent_cases, score_cached
from src.evaluation.schema import AgentReport, CaseResult, Checks, EvalCase
from src.evaluation.summary import build_summary


def _case(case_id: str = "c1", query: str = "q", **checks) -> EvalCase:
    return EvalCase(
        id=case_id,
        agent="order",
        query=query,
        checks=Checks(**checks),
        customer_id="CUST-A",
    )


# ══════════════════════════════════════════════════════════════════════════
# Invalidation
# ══════════════════════════════════════════════════════════════════════════


def test_prompt_hash_tracks_the_question_and_identity() -> None:
    base = _case(query="what did I buy")
    assert prompt_hash(base) == prompt_hash(_case(query="what did I buy"))
    assert prompt_hash(base) != prompt_hash(_case(query="something else"))

    other_identity = EvalCase(
        id="c1",
        agent="order",
        query="what did I buy",
        checks=Checks(),
        customer_id="CUST-B",
    )
    assert prompt_hash(base) != prompt_hash(other_identity)


def test_prompt_hash_ignores_the_checks() -> None:
    """Changing a rubric must re-score for free, not trigger a model call."""
    lenient = _case(must_contain=("a",))
    strict = _case(must_contain=("a", "b"), expect_refusal=True)
    assert prompt_hash(lenient) == prompt_hash(strict)


def test_changed_question_invalidates_the_entry() -> None:
    cache = AnswerCache()
    original = _case(query="original question")
    result = CaseResult(case=original, answer="an answer", tools_used=["t"])
    cache.put("order", original, result)

    assert cache.get("order", original) is not None
    assert cache.get("order", _case(query="different question")) is None


def test_unchanged_question_is_reused() -> None:
    cache = AnswerCache()
    case = _case()
    cache.put("order", case, CaseResult(case=case, answer="stored", tools_used=["t"]))

    hit = cache.get("order", case)
    assert hit is not None
    assert hit[0] == "stored"
    assert hit[1] == ["t"]


def test_failed_and_empty_results_are_not_cached() -> None:
    """A provider error is not an answer; caching it would freeze the failure."""
    cache = AnswerCache()
    case = _case()
    cache.put("order", case, CaseResult(case=case, error="RateLimitError: 429"))
    cache.put("order", _case("c2"), CaseResult(case=_case("c2"), answer="   "))
    assert len(cache) == 0


def test_skipped_results_are_not_cached() -> None:
    cache = AnswerCache()
    case = _case()
    cache.put("order", case, CaseResult(case=case, skipped="mutates the database"))
    assert len(cache) == 0


def test_code_change_is_flagged_but_still_reused() -> None:
    """Staleness is surfaced, not enforced — the operator decides."""
    cache = AnswerCache()
    case = _case()
    cache.put("order", case, CaseResult(case=case, answer="stored"))
    # Simulate the agent source changing since the answer was recorded.
    cache._entries["order::c1"]["code_fingerprint"] = "0000000000000000"

    assert cache.get("order", case) is not None
    assert case.id in cache.stale_code


def test_current_code_is_not_flagged_as_stale() -> None:
    cache = AnswerCache()
    case = _case()
    cache.put("order", case, CaseResult(case=case, answer="stored"))
    assert cache.get("order", case) is not None
    assert cache.stale_code == []


def test_fingerprint_is_stable_and_non_empty() -> None:
    assert code_fingerprint() == code_fingerprint()
    assert len(code_fingerprint()) == 16


# ══════════════════════════════════════════════════════════════════════════
# Seeding from existing reports
# ══════════════════════════════════════════════════════════════════════════


def test_existing_reports_seed_the_cache(tmp_path: Path) -> None:
    """Answers already paid for in output/evaluation/ must not be re-bought."""
    report = {
        "generated_at": "2026-07-28T16:14:03+00:00",
        "agents": {
            "order": {
                "results": [
                    {
                        "id": "c1",
                        "answer": "a recorded answer",
                        "tools_used": ["list_order_items"],
                        "latency_seconds": 3.2,
                        "skipped": "",
                        "error": "",
                    },
                    # Skipped and errored records carry no reusable answer.
                    {"id": "c2", "answer": "", "skipped": "mutates", "error": ""},
                    {"id": "c3", "answer": "", "skipped": "", "error": "429"},
                ]
            }
        },
    }
    (tmp_path / "evaluation-20260728-161403.json").write_text(json.dumps(report))

    cache = AnswerCache.load(cache_path=tmp_path / "nope.json", report_dir=tmp_path)
    assert len(cache) == 1

    hit = cache.get("order", _case("c1"))
    assert hit is not None
    assert hit[0] == "a recorded answer"
    # Legacy records did not store the question, so reuse is unverified.
    assert "c1" in cache.unverified_prompt


def test_newer_reports_win_over_older_ones(tmp_path: Path) -> None:
    for stamp, answer in [("20260726-083450", "old"), ("20260728-161403", "new")]:
        (tmp_path / f"evaluation-{stamp}.json").write_text(
            json.dumps(
                {
                    "generated_at": stamp,
                    "agents": {
                        "order": {
                            "results": [
                                {"id": "c1", "answer": answer, "tools_used": []}
                            ]
                        }
                    },
                }
            )
        )
    cache = AnswerCache.load(cache_path=tmp_path / "nope.json", report_dir=tmp_path)
    hit = cache.get("order", _case("c1"))
    assert hit is not None and hit[0] == "new"


def test_cache_round_trips_through_disk(tmp_path: Path) -> None:
    cache = AnswerCache()
    case = _case()
    cache.put("order", case, CaseResult(case=case, answer="stored", tools_used=["t"]))
    path = cache.save(tmp_path / "answer-cache.json")

    reloaded = AnswerCache.load(cache_path=path, report_dir=tmp_path / "empty")
    hit = reloaded.get("order", case)
    assert hit is not None and hit[0] == "stored"


# ══════════════════════════════════════════════════════════════════════════
# Re-scoring
# ══════════════════════════════════════════════════════════════════════════


def test_stored_answers_are_rescored_against_current_checks() -> None:
    """The saving is in the model call; the score is always computed fresh."""
    passing = score_cached(_case(must_contain=("alpha",)), "alpha beta", [])
    assert passing.passed
    assert passing.reused

    failing = score_cached(_case(must_contain=("gamma",)), "alpha beta", [])
    assert not failing.passed


class _CountingAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, messages, customer_id=None, **kwargs):
        self.calls += 1

        class _R:
            text = "fresh answer"
            tool_calls: list[str] = []
            error = ""

        return _R()


def test_runner_skips_the_model_for_cached_cases() -> None:
    agent = _CountingAgent()
    case = _case(must_contain=("stored",))
    cache = AnswerCache()
    cache.put("order", case, CaseResult(case=case, answer="stored answer"))

    report = run_agent_cases("order", agent, [case], progress=False, cache=cache)
    assert agent.calls == 0
    assert report.executed[0].answer == "stored answer"
    assert report.executed[0].reused


def test_a_reused_failure_is_re_verified_with_a_live_call() -> None:
    """A stale stored answer must not be reported as a real defect.

    Fixture-derived values in a question can move between runs, so a stored
    answer may fail a check the current system would pass.
    """
    agent = _CountingAgent()  # answers "fresh answer"
    case = _case(must_contain=("fresh",))
    cache = AnswerCache()
    cache.put("order", case, CaseResult(case=case, answer="stale answer"))

    report = run_agent_cases("order", agent, [case], progress=False, cache=cache)
    assert agent.calls == 1
    assert report.executed[0].passed
    assert report.executed[0].answer == "fresh answer"


def test_reused_passes_are_not_re_verified() -> None:
    """Only failures pay for verification; the saving on passes is the point."""
    agent = _CountingAgent()
    case = _case(must_contain=("stored",))
    cache = AnswerCache()
    cache.put("order", case, CaseResult(case=case, answer="stored answer"))

    run_agent_cases("order", agent, [case], progress=False, cache=cache)
    assert agent.calls == 0


def test_verification_can_be_disabled() -> None:
    agent = _CountingAgent()
    case = _case(must_contain=("fresh",))
    cache = AnswerCache()
    cache.put("order", case, CaseResult(case=case, answer="stale answer"))

    report = run_agent_cases(
        "order", agent, [case], progress=False, cache=cache, verify_failures=False
    )
    assert agent.calls == 0
    assert not report.executed[0].passed


def test_runner_calls_the_model_when_uncached() -> None:
    agent = _CountingAgent()
    cache = AnswerCache()
    report = run_agent_cases("order", agent, [_case()], progress=False, cache=cache)
    assert agent.calls == 1
    assert not report.executed[0].reused
    assert len(cache) == 1  # and the fresh answer is now stored


def test_mean_latency_excludes_reused_cases() -> None:
    """Reused entries carry a latency from another run; averaging them lies."""
    fresh = CaseResult(case=_case("a"), answer="x", latency_seconds=4.0)
    fresh.checks = score_cached(_case("a"), "x", []).checks
    reused = CaseResult(case=_case("b"), answer="x", latency_seconds=99.0, reused=True)
    reused.checks = fresh.checks

    report = AgentReport(agent="order", results=[fresh, reused])
    assert report.mean_latency == pytest.approx(4.0)
    assert len(report.reused) == 1


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════


def _report_with(passed: bool, tags: tuple[str, ...]) -> AgentReport:
    case = EvalCase(
        id="c1",
        agent="order",
        query="q",
        checks=Checks(must_contain=("alpha",)),
        tags=tags,
        description="a described case",
    )
    result = CaseResult(case=case, answer="alpha" if passed else "beta")
    result.checks = score_cached(case, result.answer, []).checks
    return AgentReport(agent="order", results=[result])


def test_summary_renders_headline_metrics() -> None:
    text = build_summary([_report_with(True, ("happy-path",))])
    assert "# Evaluation Summary" in text
    assert "Overall pass rate" in text
    assert "100%" in text


def test_summary_groups_cases_into_dimensions() -> None:
    text = build_summary([_report_with(True, ("security", "cross-account"))])
    assert "Access control" in text


def test_summary_lists_outstanding_failures_with_context() -> None:
    text = build_summary([_report_with(False, ("edge",))])
    assert "Outstanding failures" in text
    assert "a described case" in text


def test_summary_reports_no_failures_when_clean() -> None:
    assert "every scored case passed" in build_summary([_report_with(True, ("edge",))])


def test_summary_discloses_reuse_and_staleness() -> None:
    text = build_summary([_report_with(True, ("edge",))], reused=12, stale=5)
    assert "12" in text and "reused" in text.lower()
    assert "Caveat" in text
    assert "5" in text


def test_summary_always_states_its_limits() -> None:
    """The numbers must never be presented without their caveats."""
    text = build_summary([_report_with(True, ("happy-path",))])
    assert "Response quality is not measured" in text
    assert "regression baseline" in text
