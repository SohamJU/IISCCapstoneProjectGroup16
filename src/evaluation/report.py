"""Render evaluation results as console text, markdown and JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evaluation.routing import RoutingReport
from src.evaluation.schema import AgentReport

REPORT_DIR = Path("output/evaluation")


def _bar(fraction: float, width: int = 20) -> str:
    filled = round(fraction * width)
    return "#" * filled + "." * (width - filled)


def format_console(
    reports: list[AgentReport],
    routing: RoutingReport | None = None,
) -> str:
    """Human-readable summary for the terminal."""
    lines: list[str] = []
    if not reports:
        # --routing-only: an empty agent table reading "OVERALL 0 0%" looks
        # like a total failure rather than a section that was not requested.
        return _format_routing_section(routing) if routing else "No results."

    lines.append("=" * 72)
    lines.append("AGENT EVALUATION")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        f"{'agent':<16}{'cases':>7}{'passed':>8}{'pass rate':>12}"
        f"{'checks':>10}{'latency':>10}"
    )
    lines.append("-" * 72)

    for report in sorted(reports, key=lambda r: r.agent):
        lines.append(
            f"{report.agent:<16}{report.total:>7}{report.passed:>8}"
            f"{report.pass_rate:>11.0%}{report.check_score:>10.0%}"
            f"{report.mean_latency:>9.1f}s"
        )

    executed = sum(r.total for r in reports)
    passed = sum(r.passed for r in reports)
    lines.append("-" * 72)
    overall = passed / executed if executed else 0.0
    lines.append(f"{'OVERALL':<16}{executed:>7}{passed:>8}{overall:>11.0%}")
    lines.append("")

    unscored = [(r.agent, u) for r in reports for u in r.unscored]
    if unscored:
        lines.append(
            f"!! {len(unscored)} case(s) could NOT be scored — provider errors "
            f"(rate limit / quota / connection)."
        )
        lines.append(
            "   These are excluded from the rates above; the scores cover only "
            "cases that reached the model."
        )
        for agent, result in unscored[:5]:
            lines.append(f"  - [{agent}] {result.case.id}: {result.error[:110]}")
        if len(unscored) > 5:
            lines.append(f"  ... and {len(unscored) - 5} more")
        lines.append("")

    skipped = [(r.agent, s) for r in reports for s in r.skipped]
    if skipped:
        lines.append(f"Skipped ({len(skipped)}):")
        for agent, result in skipped:
            lines.append(f"  - [{agent}] {result.case.id}: {result.skipped}")
        lines.append("")

    failures = [(r.agent, res) for r in reports for res in r.executed if not res.passed]
    if failures:
        lines.append(f"Failures ({len(failures)}):")
        for agent, result in failures:
            lines.append(f"  - [{agent}] {result.case.id}")
            if result.case.description:
                lines.append(f"      {result.case.description}")
            if result.error:
                lines.append(f"      ERROR: {result.error}")
            for check in result.failures:
                detail = f" — {check.detail}" if check.detail else ""
                lines.append(f"      x {check.name}{detail}")
            lines.append(f"      tools: {result.tools_used or 'none'}")
            lines.append(f"      answer: {result.answer[:180]!r}")
        lines.append("")

    if routing is not None:
        lines.append(_format_routing_section(routing))

    return "\n".join(lines)


def _format_routing_section(routing: RoutingReport) -> str:
    """Render the routing block on its own, so --routing-only reads cleanly."""
    lines: list[str] = ["=" * 72, "INTENT ROUTING", "=" * 72, ""]

    core_scored = [s for s in routing.core if not s.error]
    lines.append(
        f"Core accuracy      {routing.core_accuracy:>6.0%}  "
        f"{_bar(routing.core_accuracy)}  (n={len(core_scored)})  strict: primary intent only"
    )
    lines.append(
        f"  ... lenient      {routing.core_accuracy_lenient:>6.0%}  "
        f"{_bar(routing.core_accuracy_lenient)}  any intent labelled on the row"
    )

    ambiguous_scored = [s for s in routing.ambiguous if not s.error]
    if ambiguous_scored:
        lines.append(
            f"Ambiguous accepted {routing.ambiguous_acceptance:>6.0%}  "
            f"{_bar(routing.ambiguous_acceptance)}  (n={len(ambiguous_scored)})"
        )
        lines.append(
            "  (intents with no single correct route — billing, account, "
            "loyalty, complaints)"
        )

    if routing.errors:
        lines.append(
            f"Router errors      {len(routing.errors)} (excluded from the figures above)"
        )
        lines.append(f"  e.g. {routing.errors[0].error[:120]}")
    lines.append("")

    metrics = routing.per_route_metrics()
    if metrics:
        lines.append(f"{'route':<18}{'precision':>11}{'recall':>9}{'f1':>7}{'support':>9}")
        lines.append("-" * 54)
        for route in sorted(metrics):
            values = metrics[route]
            lines.append(
                f"{route:<18}{values['precision']:>10.0%}{values['recall']:>9.0%}"
                f"{values['f1']:>7.2f}{int(values['support']):>9}"
            )
        lines.append("")

    confusion = routing.confusion()
    if confusion:
        lines.append("Confusion (gold -> predicted):")
        for gold in sorted(confusion):
            predictions = ", ".join(
                f"{route}:{count}" for route, count in confusion[gold].most_common()
            )
            lines.append(f"  {gold:<16} {predictions}")
        lines.append("")

    return "\n".join(lines)


def format_markdown(
    reports: list[AgentReport],
    routing: RoutingReport | None = None,
) -> str:
    """Markdown report suitable for dropping into project documentation."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Agent Evaluation Report",
        "",
        f"_Generated {stamp}_",
        "",
        "Scoring is deterministic: each case asserts which tools must run, which "
        "facts must appear, which must not, and whether a refusal was required. "
        "A case passes only if every one of its checks passes.",
        "",
        "## Per-agent results",
        "",
        "| Agent | Cases | Passed | Pass rate | Check score | Mean latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for report in sorted(reports, key=lambda r: r.agent):
        lines.append(
            f"| {report.agent} | {report.total} | {report.passed} | "
            f"{report.pass_rate:.0%} | {report.check_score:.0%} | "
            f"{report.mean_latency:.1f}s |"
        )

    executed = sum(r.total for r in reports)
    passed = sum(r.passed for r in reports)
    overall = passed / executed if executed else 0.0
    lines += [
        f"| **Overall** | **{executed}** | **{passed}** | **{overall:.0%}** | | |",
        "",
    ]

    unscored = [(r.agent, u) for r in reports for u in r.unscored]
    if unscored:
        lines += [
            f"> **{len(unscored)} case(s) could not be scored** because of provider "
            f"errors (rate limit, daily quota or connection). They are excluded "
            f"from the rates above — the scores describe only cases that reached "
            f"the model.",
            "",
        ]
        for agent, result in unscored:
            lines.append(f"- `{agent}` / {result.case.id}: {result.error[:160]}")
        lines.append("")

    failures = [(r.agent, res) for r in reports for res in r.executed if not res.passed]
    if failures:
        lines += ["## Failures", ""]
        for agent, result in failures:
            lines.append(f"### `{agent}` — {result.case.id}")
            if result.case.description:
                lines.append(f"_{result.case.description}_")
            lines += ["", f"**Query:** {result.case.query}", ""]
            if result.error:
                lines += [f"**Error:** `{result.error}`", ""]
            for check in result.failures:
                detail = f" — {check.detail}" if check.detail else ""
                lines.append(f"- Failed `{check.name}`{detail}")
            lines += [
                "",
                f"**Tools called:** `{result.tools_used or 'none'}`",
                "",
                "**Answer:**",
                "",
                "```",
                result.answer[:800] or "(empty)",
                "```",
                "",
            ]

    if routing is not None:
        core_scored = [s for s in routing.core if not s.error]
        ambiguous_scored = [s for s in routing.ambiguous if not s.error]
        lines += [
            "## Intent routing",
            "",
            "The labelled dataset uses 17 intents; this system has 6 routes. "
            "Intents with exactly one defensible route form the core set and "
            "produce the headline figure. Intents describing work this system "
            "has no agent for (billing, account changes, loyalty, general "
            "complaints) are scored against a set of acceptable routes and "
            "reported separately.",
            "",
            f"- **Core routing accuracy (strict): {routing.core_accuracy:.0%}** "
            f"(n={len(core_scored)}) — the row's primary intent must be predicted.",
            f"- Core routing accuracy (lenient): {routing.core_accuracy_lenient:.0%} "
            f"— any intent listed in the row's `all_intents` is accepted. Most rows "
            f"are genuinely multi-intent, so the strict figure partly measures which "
            f"intent the dataset happened to nominate as primary.",
        ]
        if ambiguous_scored:
            lines.append(
                f"- Ambiguous-intent acceptance: {routing.ambiguous_acceptance:.0%} "
                f"(n={len(ambiguous_scored)})"
            )
        if routing.errors:
            lines.append(
                f"- Router errors: {len(routing.errors)} (excluded from the above)"
            )
        lines += ["", "| Route | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        metrics = routing.per_route_metrics()
        for route in sorted(metrics):
            values = metrics[route]
            lines.append(
                f"| {route} | {values['precision']:.0%} | {values['recall']:.0%} | "
                f"{values['f1']:.2f} | {int(values['support'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def build_json_payload(
    reports: list[AgentReport],
    routing: RoutingReport | None = None,
) -> dict[str, Any]:
    """Machine-readable results, for tracking scores across runs."""
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "agents": {},
    }

    for report in reports:
        payload["agents"][report.agent] = {
            "cases": report.total,
            "passed": report.passed,
            "pass_rate": round(report.pass_rate, 4),
            "check_score": round(report.check_score, 4),
            "unscored_provider_errors": len(report.unscored),
            "mean_latency_seconds": round(report.mean_latency, 3),
            "results": [
                {
                    "id": result.case.id,
                    "description": result.case.description,
                    "tags": list(result.case.tags),
                    "passed": result.passed,
                    "skipped": result.skipped,
                    "error": result.error,
                    "latency_seconds": round(result.latency_seconds, 3),
                    "tools_used": result.tools_used,
                    "answer": result.answer,
                    "checks": [asdict(check) for check in result.checks],
                }
                for result in report.results
            ],
        }

    executed = sum(r.total for r in reports)
    passed = sum(r.passed for r in reports)
    payload["overall"] = {
        "cases": executed,
        "passed": passed,
        "pass_rate": round(passed / executed, 4) if executed else 0.0,
    }

    if routing is not None:
        payload["routing"] = {
            "core_accuracy": round(routing.core_accuracy, 4),
            "core_accuracy_lenient": round(routing.core_accuracy_lenient, 4),
            "router_errors": len(routing.errors),
            "core_n": len([s for s in routing.core if not s.error]),
            "ambiguous_acceptance": round(routing.ambiguous_acceptance, 4),
            "ambiguous_n": len([s for s in routing.ambiguous if not s.error]),
            "per_route": routing.per_route_metrics(),
            "confusion": {
                gold: dict(counter) for gold, counter in routing.confusion().items()
            },
        }

    return payload


def write_reports(
    reports: list[AgentReport],
    routing: RoutingReport | None = None,
    directory: Path = REPORT_DIR,
) -> dict[str, Path]:
    """Write markdown and JSON reports, returning the paths written."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    markdown_path = directory / f"evaluation-{stamp}.md"
    json_path = directory / f"evaluation-{stamp}.json"
    latest_path = directory / "latest.md"

    markdown = format_markdown(reports, routing)
    markdown_path.write_text(markdown, encoding="utf-8")
    latest_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(build_json_payload(reports, routing), indent=2), encoding="utf-8"
    )

    return {"markdown": markdown_path, "json": json_path, "latest": latest_path}
