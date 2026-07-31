"""The consolidated evaluation summary — the artifact to present.

The per-run report in :mod:`src.evaluation.report` answers "what happened in
this run". This module answers "what is the state of the system", aggregating
per-agent results, dimension breakdowns, routing metrics and coverage counts
into one document, and stating plainly what the numbers do and do not support.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from src.evaluation.routing import RoutingReport
from src.evaluation.schema import AgentReport

SUMMARY_PATH = Path("output/evaluation/SUMMARY.md")

#: Tags rolled up into presentable dimensions, in reporting order. A case may
#: carry several tags; it counts once per dimension it belongs to.
_DIMENSIONS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "Core functionality",
        ("happy-path", "items", "status", "tracking", "listing", "purchase"),
        "Ordinary requests answered correctly from real data",
    ),
    (
        "Access control",
        ("security", "cross-account", "unauthenticated", "third-party", "attribution"),
        "One customer cannot reach or be shown another's data",
    ),
    (
        "Hallucination resistance",
        ("hallucination", "grounding", "negative"),
        "No invented orders, prices, policies or identifiers",
    ),
    (
        "Edge cases",
        ("edge", "validation", "state-machine", "robustness", "policy"),
        "Malformed input, impossible constraints, expired windows",
    ),
    (
        "Scope and UX",
        ("scope", "ux", "clarification", "precision", "tone", "constraints"),
        "Stays in domain, never asks for internal identifiers",
    ),
)


def _dimension_rows(reports: list[AgentReport]) -> list[tuple[str, int, int, str]]:
    """(dimension, passed, total, description) across every agent."""
    rows: list[tuple[str, int, int, str]] = []
    for label, tags, description in _DIMENSIONS:
        wanted = set(tags)
        results = [
            result
            for report in reports
            for result in report.executed
            if wanted & set(result.case.tags)
        ]
        if not results:
            continue
        passed = sum(1 for r in results if r.passed)
        rows.append((label, passed, len(results), description))
    return rows


def _tag_counts(reports: list[AgentReport]) -> Counter:
    counts: Counter = Counter()
    for report in reports:
        for result in report.results:
            for tag in result.case.tags:
                counts[tag] += 1
    return counts


def build_summary(
    reports: list[AgentReport],
    routing: RoutingReport | None = None,
    reused: int = 0,
    stale: int = 0,
) -> str:
    """Render the consolidated summary as markdown."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    executed = sum(r.total for r in reports)
    passed = sum(r.passed for r in reports)
    overall = passed / executed if executed else 0.0
    skipped = sum(len(r.skipped) for r in reports)
    unscored = sum(len(r.unscored) for r in reports)
    authored = sum(len(r.results) for r in reports)

    lines: list[str] = [
        "# Evaluation Summary",
        "",
        f"_Generated {stamp}_",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Cases authored | {authored} |",
        f"| Cases scored | {executed} |",
        f"| Cases passed | {passed} |",
        f"| **Overall pass rate** | **{overall:.0%}** |",
    ]
    if routing is not None:
        single_rate, single_n = routing.single_intent
        multi_rate, multi_n = routing.multi_intent
        lines.append(f"| Intent-routing accuracy | {routing.accuracy:.0%} |")
        if single_n:
            lines.append(f"| — single-intent | {single_rate:.0%} |")
        if multi_n:
            lines.append(f"| — multi-intent | {multi_rate:.0%} |")
    if skipped:
        lines.append(f"| Skipped (would mutate the database) | {skipped} |")
    if unscored:
        lines.append(f"| Unscored (provider errors) | {unscored} |")
    lines.append("")

    # ── Per agent ────────────────────────────────────────────────────────
    lines += [
        "## Per-agent results",
        "",
        "| Agent | Scored | Passed | Pass rate | Check score | Mean latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in sorted(reports, key=lambda r: r.agent):
        latency = f"{report.mean_latency:.1f}s" if report.mean_latency else "—"
        lines.append(
            f"| {report.agent} | {report.total} | {report.passed} | "
            f"{report.pass_rate:.0%} | {report.check_score:.0%} | {latency} |"
        )
    lines += [
        f"| **Total** | **{executed}** | **{passed}** | **{overall:.0%}** | | |",
        "",
        "**Pass rate** counts a case only when *every* check on it passed. "
        "**Check score** is the proportion of individual checks passed, and "
        "shows how near the failures were.",
        "",
    ]

    # ── By dimension ─────────────────────────────────────────────────────
    dimension_rows = _dimension_rows(reports)
    if dimension_rows:
        lines += [
            "## By dimension",
            "",
            "| Dimension | Passed | Rate | What it verifies |",
            "|---|---:|---:|---|",
        ]
        for label, dim_passed, total, description in dimension_rows:
            rate = dim_passed / total if total else 0.0
            lines.append(
                f"| {label} | {dim_passed}/{total} | {rate:.0%} | {description} |"
            )
        lines += [
            "",
            "Dimensions are derived from case tags and overlap: a case tagged "
            "both `security` and `edge` counts in both rows.",
            "",
        ]

    # ── Routing ──────────────────────────────────────────────────────────
    if routing is not None:
        single_rate, single_n = routing.single_intent
        multi_rate, multi_n = routing.multi_intent
        lines += [
            "## Intent routing",
            "",
            "Scored against a hand-authored dataset covering all six routes, "
            "multi-intent requests, boundary cases and requests outside the "
            "agent taxonomy. Each case declares which routes must be predicted, "
            "which are merely acceptable, and which are forbidden.",
            "",
            f"- **Overall accuracy: {routing.accuracy:.0%}** (n={len(routing.scored)})",
        ]
        if single_n:
            lines.append(f"- Single-intent: {single_rate:.0%} (n={single_n})")
        if multi_n:
            lines.append(
                f"- Multi-intent: {multi_rate:.0%} (n={multi_n}) — **every** required "
                "route must appear, so this is the harder measure."
            )
        lines.append(
            f"- Over-routing: {routing.over_routing_rate:.0%} of cases received an "
            "extra route that was not asked for. Tracked because a router that "
            "always answers with two routes would otherwise pass by covering "
            "every possibility."
        )
        if routing.errors:
            lines.append(
                f"- Router errors: {len(routing.errors)} (excluded from the above)"
            )
        lines += [
            "",
            "This dataset replaced the generated `customer_queries` table, whose "
            "labels were unreliable — queries labelled `product_search` read "
            '"it is missing parts and I need to...", which is a return. Scoring '
            "against them measured label noise as much as router quality. The "
            "hand-authored set is smaller but every case is deliberate.",
            "",
            "**Caveat.** The router is not deterministic, so a small dataset "
            "moves between runs. Treat a single-case change as noise rather "
            "than as a regression.",
            "",
        ]

        metrics = routing.per_route_metrics()
        if metrics:
            lines += [
                "| Route | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
            ]
            for route in sorted(metrics):
                values = metrics[route]
                if not values["support"] and not values["precision"]:
                    continue
                lines.append(
                    f"| {route} | {values['precision']:.0%} | {values['recall']:.0%} "
                    f"| {values['f1']:.2f} | {int(values['support'])} |"
                )
            lines.append("")

    # ── Coverage ─────────────────────────────────────────────────────────
    counts = _tag_counts(reports)
    if counts:
        top = ", ".join(f"`{tag}` ({n})" for tag, n in counts.most_common(12))
        lines += ["## Coverage", "", f"Cases by tag: {top}", ""]

    # ── Failures ─────────────────────────────────────────────────────────
    failures = [
        (report.agent, result)
        for report in reports
        for result in report.executed
        if not result.passed
    ]
    lines += ["## Outstanding failures", ""]
    if not failures:
        lines.append("None — every scored case passed.")
    else:
        lines += [
            "| Agent | Case | Failed check | What it means |",
            "|---|---|---|---|",
        ]
        for agent, result in failures:
            names = ", ".join(c.name for c in result.failures) or result.error
            lines.append(
                f"| {agent} | `{result.case.id}` | {names} | "
                f"{result.case.description} |"
            )
    lines.append("")

    # ── Method and limits ────────────────────────────────────────────────
    lines += [
        "## Method",
        "",
        "Scoring is deterministic. Each case asserts checkable properties — "
        "which tools ran, which facts appear, which must not, whether a refusal "
        "occurred, and whether any cited order identifier is real. There is no "
        "model-as-judge, so the same dataset against the same code yields the "
        "same score and a change in the number reflects a change in behaviour.",
        "",
        "Agents are evaluated directly rather than through the supervisor graph, "
        "so a failure is attributable to the agent rather than to routing. "
        "Routing is measured separately against the labelled query dataset.",
        "",
    ]
    if reused:
        lines += [
            f"{reused} of the scored cases reused an answer recorded in an "
            "earlier run rather than calling the model again, to stay within the "
            "provider's quota. Checks were re-applied fresh to every stored "
            "answer; only the answer text was reused, and a changed question "
            "always forces a new call.",
            "",
        ]
        if stale:
            lines += [
                f"**Caveat:** {stale} of those reused answers were recorded "
                "before the current agent code. They are reported as measured, "
                "but a clean run (`--no-cache`) is needed to confirm them.",
                "",
            ]

    lines += [
        "## What these numbers do not show",
        "",
        "- **Response quality is not measured.** A terse but correct answer and "
        "a well-structured one score identically; only correctness, tool "
        "selection and safety are checked.",
        "- **The cases were authored alongside the implementation**, so they "
        "encode intended behaviour rather than independently specified "
        "requirements. A full pass is a regression baseline, not proof of "
        "general reliability.",
        "- **Retrieval metrics are absent.** Context precision and faithfulness "
        "for the RAG paths are not implemented.",
        "- **The routing dataset is small and hand-authored.** It covers every "
        "route deliberately, but 32 cases cannot characterise the full space of "
        "customer phrasing, and the cases reflect the authors' idea of what a "
        "correct routing decision is.",
        "- **A 100% pass rate means the suite found nothing, not that nothing "
        "is there.** Every defect fixed during development was first found by "
        "a case being added; the score reflects coverage as much as quality.",
        "",
    ]

    return "\n".join(lines)


def write_summary(
    reports: list[AgentReport],
    routing: RoutingReport | None = None,
    reused: int = 0,
    stale: int = 0,
    path: Path = SUMMARY_PATH,
) -> Path:
    """Write the consolidated summary and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_summary(reports, routing, reused, stale), encoding="utf-8")
    return path
