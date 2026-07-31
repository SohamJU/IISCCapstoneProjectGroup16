"""Execute evaluation cases against the specialist agents.

Agents are run **directly**, not through the orchestrator graph, so a failure
here is attributable to the agent rather than to routing. Routing is measured
separately in :mod:`src.evaluation.routing`; between them the two answer
"did the right agent get the request" and "did that agent handle it correctly"
without confounding the two.
"""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage

from src.evaluation.checks import run_checks
from src.evaluation.schema import AgentReport, CaseResult, Checks, EvalCase
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)


def build_agents(agents: list[str], debug: bool = False) -> dict[str, object]:
    """Instantiate the requested specialists.

    Imported lazily so that loading this module — which the harness's own unit
    tests do — does not require a working LLM provider.
    """
    from src.agents.graph import build_specialists

    specialists = build_specialists(debug=debug)
    return {name: agent for name, agent in specialists.items() if name in agents}


def run_case(agent: object, case: EvalCase) -> CaseResult:
    """Run one case and score it."""
    started = time.perf_counter()
    try:
        result = agent.run(  # type: ignore[attr-defined]
            [HumanMessage(content=case.query)],
            customer_id=case.customer_id,
        )
    except Exception as exc:  # a crashed agent is a failed case, not a crashed run
        _LOGGER.error("case %s raised: %s", case.id, exc)
        return CaseResult(
            case=case,
            latency_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    latency = time.perf_counter() - started
    answer = (result.text or "").strip()
    tools_used = list(result.tool_calls or [])

    case_result = CaseResult(
        case=case,
        answer=answer,
        tools_used=tools_used,
        latency_seconds=latency,
        error=result.error or "",
    )

    # An agent that errored still gets scored: a refusal case can legitimately
    # pass on the strength of its answer text, and scoring keeps the failure
    # visible in the per-check breakdown rather than as a bare exception.
    case_result.checks = run_checks(case.checks, answer, tools_used)
    return case_result


def score_cached(case: EvalCase, answer: str, tools_used: list[str]) -> CaseResult:
    """Score a previously recorded answer without calling the model.

    Checks are applied fresh, so a tightened rubric re-scores stored answers for
    free — only a changed question requires a new call.
    """
    result = CaseResult(case=case, answer=answer, tools_used=list(tools_used), reused=True)
    result.checks = run_checks(case.checks, answer, tools_used)
    return result


def run_agent_cases(
    agent_name: str,
    agent: object,
    cases: list[EvalCase],
    allow_writes: bool = False,
    unavailable: list[tuple[str, str]] | None = None,
    progress: bool = True,
    cache: object | None = None,
    verify_failures: bool = True,
) -> AgentReport:
    """Run every case for one agent, honouring the write guard.

    When ``cache`` is supplied, a case whose answer is already recorded is
    re-scored rather than re-asked. ``verify_failures`` re-runs any reused case
    that fails, so a stale stored answer cannot be reported as a real defect.
    """
    report = AgentReport(agent=agent_name)

    # Cases the database could not supply data for are recorded as skipped so
    # they stay visible in the report instead of quietly shrinking the denominator.
    for case_id, reason in unavailable or []:
        report.results.append(
            CaseResult(
                case=EvalCase(id=case_id, agent=agent_name, query="", checks=Checks()),
                skipped=reason,
            )
        )

    for index, case in enumerate(cases, start=1):
        if case.mutates and not allow_writes:
            report.results.append(
                CaseResult(
                    case=case,
                    skipped="mutates the database; re-run with --allow-writes",
                )
            )
            continue

        if progress:
            print(f"  [{index}/{len(cases)}] {case.id} ... ", end="", flush=True)

        cached = cache.get(agent_name, case) if cache is not None else None
        if cached is not None:
            answer, tools_used, _recorded_at = cached
            result = score_cached(case, answer, tools_used)

            # A reused answer that fails is re-verified with a live call before
            # being reported. Stored answers can be stale — a fixture-derived
            # value in the question may have moved since the answer was
            # recorded — and a stale answer produces a failure for a test the
            # current system would pass. Only failures pay this cost, so the
            # saving on the passing majority is preserved.
            if not result.passed and verify_failures:
                if progress:
                    print("reused-FAIL, re-verifying ... ", end="", flush=True)
                fresh = run_case(agent, case)
                if not fresh.error:
                    result = fresh
                    if cache is not None:
                        cache.put(agent_name, case, fresh)
        else:
            result = run_case(agent, case)
            if cache is not None:
                cache.put(agent_name, case, result)

        report.results.append(result)

        if progress:
            marker = "reused" if result.reused else f"{result.latency_seconds:.1f}s"
            if result.passed:
                print(f"PASS ({marker})")
            else:
                failed = ", ".join(check.name for check in result.failures) or result.error
                print(f"FAIL ({marker}) — {failed}")

    return report
