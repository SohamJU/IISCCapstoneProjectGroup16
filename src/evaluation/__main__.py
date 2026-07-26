"""Command-line entrypoint for the evaluation harness."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from src.evaluation.fixtures import FixtureError, build_fixtures  # noqa: E402
from src.evaluation.loader import AGENT_DATASETS, load_all_cases  # noqa: E402
from src.evaluation.report import (  # noqa: E402
    format_console,
    write_reports,
)
from src.evaluation.routing import evaluate_routing, load_labelled_queries  # noqa: E402
from src.evaluation.runner import build_agents, run_agent_cases  # noqa: E402
from src.evaluation.schema import AgentReport  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation",
        description=(
            "Score each agent against its ground-truth cases, and the router "
            "against the labelled customer_queries dataset."
        ),
    )
    parser.add_argument(
        "--agents",
        nargs="*",
        choices=sorted(AGENT_DATASETS),
        help="Agents to evaluate (default: all).",
    )
    parser.add_argument(
        "--no-routing",
        action="store_true",
        help="Skip the routing pass (saves one LLM call per sampled query).",
    )
    parser.add_argument(
        "--routing-only",
        action="store_true",
        help="Run only the routing evaluation.",
    )
    parser.add_argument(
        "--routing-per-intent",
        type=int,
        default=3,
        help="Labelled queries sampled per intent (default: 3, i.e. ~51 queries).",
    )
    parser.add_argument(
        "--allow-writes",
        action="store_true",
        help=(
            "Run cases marked as mutating the database. Off by default so a "
            "routine evaluation cannot cancel orders or file returns."
        ),
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Only run cases carrying at least one of these tags (e.g. security).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case progress output.",
    )
    parser.add_argument(
        "--no-write-report",
        action="store_true",
        help="Print to the console without writing files to output/evaluation/.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help=(
            "Exit non-zero if the overall pass rate is below this fraction "
            "(e.g. 0.8). Use in CI."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    progress = not args.quiet

    try:
        fixtures = build_fixtures()
    except FixtureError as exc:
        print(f"Cannot build evaluation fixtures: {exc}", file=sys.stderr)
        print("Is the database reachable and seeded?", file=sys.stderr)
        return 2

    reports: list[AgentReport] = []

    if not args.routing_only:
        selected = args.agents or list(AGENT_DATASETS)
        cases_by_agent, unavailable = load_all_cases(fixtures, selected)

        if args.tags:
            wanted = set(args.tags)
            cases_by_agent = {
                agent: [case for case in cases if wanted & set(case.tags)]
                for agent, cases in cases_by_agent.items()
            }

        agents = build_agents(selected)
        missing = [name for name in selected if name not in agents]
        if missing:
            print(
                f"Could not build agent(s): {', '.join(missing)}. "
                "Check the LLM provider configuration.",
                file=sys.stderr,
            )

        for name in selected:
            agent = agents.get(name)
            cases = cases_by_agent.get(name, [])
            if agent is None or not cases:
                continue
            if progress:
                print(f"\n{name} agent — {len(cases)} case(s)")
            reports.append(
                run_agent_cases(
                    name,
                    agent,
                    cases,
                    allow_writes=args.allow_writes,
                    unavailable=unavailable.get(name),
                    progress=progress,
                )
            )

    routing_report = None
    if not args.no_routing:
        from src.agents.router import RouterAgent

        router = RouterAgent()
        if router.init_error:
            print(
                f"Router unavailable, skipping routing evaluation: {router.init_error}",
                file=sys.stderr,
            )
        else:
            samples = load_labelled_queries(limit_per_intent=args.routing_per_intent)
            if progress:
                print(f"\nrouting — {len(samples)} labelled quer(ies)")
            routing_report = evaluate_routing(router, samples, progress=progress)

    print()
    print(format_console(reports, routing_report))

    if not args.no_write_report and (reports or routing_report):
        paths = write_reports(reports, routing_report)
        print(f"Report written to {paths['markdown']}")
        print(f"          JSON to {paths['json']}")

    executed = sum(report.total for report in reports)
    passed = sum(report.passed for report in reports)
    unscored = sum(len(report.unscored) for report in reports)
    pass_rate = passed / executed if executed else 0.0

    if unscored:
        print(
            f"\nWARNING: {unscored} case(s) could not be scored (provider "
            f"rate limit / quota / connection). Re-run them before treating "
            f"this as a complete evaluation.",
            file=sys.stderr,
        )

    if args.fail_under is not None:
        # A run where most cases never reached the model can produce a high
        # pass rate over a handful of survivors. Treat that as a failed gate
        # rather than a pass, or CI goes green on an evaluation that mostly
        # did not happen.
        attempted = executed + unscored
        if attempted and unscored / attempted > 0.2:
            print(
                f"FAIL: {unscored}/{attempted} cases were unscored; "
                f"the run is too incomplete to gate on.",
                file=sys.stderr,
            )
            return 1
        if executed and pass_rate < args.fail_under:
            print(
                f"FAIL: pass rate {pass_rate:.0%} is below the "
                f"--fail-under threshold of {args.fail_under:.0%}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
