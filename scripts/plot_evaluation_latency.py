"""Per-agent response-latency histograms from the evaluation reports.

Produces slide-ready figures from data already on disk — it runs no agents and
makes no LLM or database calls, so it is free and repeatable.

    uv run python scripts/plot_evaluation_latency.py

Outputs to ``output/evaluation/figures/``:

    latency_histograms.png    one histogram per agent, shared axes
    latency_distribution.png  box plot across agents, ordered by median
    latency_summary.csv       the numbers behind both figures

Where the data comes from
-------------------------
Every ``output/evaluation/evaluation-*.json`` report records a
``latency_seconds`` per case. A case whose answer was **reused from the cache**
records ``0.0``, because no model call happened — those are excluded, otherwise
the distribution would be dominated by a spike at zero that measures the cache
rather than the system.

What remains is one measurement per (run, agent, case). The same case measured
in a later run is a separate, legitimate sample: verified across the current
reports, repeat measurements of a case genuinely differ (``order-status-own``
was 39.9s, 9.3s and 23.7s in three runs), so they are real observations rather
than a value carried forward.

``answer-cache.json`` is folded in as a secondary source for measurements whose
originating report is no longer present, de-duplicated against the reports on
(agent, case, latency).

Read the caveats printed at the end of the run before quoting these numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "evaluation"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "evaluation" / "figures"

#: Below this many samples an agent's histogram shape is not meaningful. It is
#: still plotted — hiding it would misrepresent coverage — but labelled.
MIN_MEANINGFUL_SAMPLES = 5

#: Stable slide colours, one per agent. Anything unrecognised falls back to grey.
AGENT_COLOURS = {
    "order": "#2E6F9E",
    "return": "#C25E3A",
    "product": "#4C8C56",
    "recommendation": "#8A6BAF",
    "escalation": "#B5484C",
    "fallback": "#7A7A7A",
}
FALLBACK_COLOUR = "#999999"


def display_path(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    ``Path.relative_to`` raises for anything outside the project, which a
    ``--output-dir`` pointing at /tmp legitimately is.
    """
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class Measurement:
    """One timed agent invocation."""

    agent: str
    case_id: str
    latency: float
    source: str

    @property
    def dedup_key(self) -> tuple[str, str, float]:
        return (self.agent, self.case_id, round(self.latency, 3))


# ── Loading ───────────────────────────────────────────────────────────────


def load_from_reports(input_dir: Path) -> list[Measurement]:
    """Read every ``evaluation-*.json`` report in ``input_dir``."""
    measurements: list[Measurement] = []
    for path in sorted(input_dir.glob("evaluation-*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! skipping unreadable report {path.name}: {exc}", file=sys.stderr)
            continue

        agents = payload.get("agents")
        if not isinstance(agents, dict):
            continue

        for agent, report in agents.items():
            for result in report.get("results") or []:
                latency = result.get("latency_seconds") or 0.0
                # 0.0 means the answer was reused — no model call was timed.
                if latency <= 0:
                    continue
                measurements.append(
                    Measurement(
                        agent=str(agent),
                        case_id=str(result.get("id", "?")),
                        latency=float(latency),
                        source=path.name,
                    )
                )
    return measurements


def load_from_cache(input_dir: Path) -> list[Measurement]:
    """Read timed entries from ``answer-cache.json``."""
    path = input_dir / "answer-cache.json"
    if not path.exists():
        return []

    try:
        entries = json.loads(path.read_text()).get("entries") or {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ! skipping unreadable cache: {exc}", file=sys.stderr)
        return []

    measurements: list[Measurement] = []
    for entry in entries.values():
        latency = entry.get("latency_seconds") or 0.0
        if latency <= 0:
            continue
        measurements.append(
            Measurement(
                agent=str(entry.get("agent", "?")),
                case_id=str(entry.get("case_id", "?")),
                latency=float(latency),
                source="answer-cache.json",
            )
        )
    return measurements


def collect(input_dir: Path, use_cache: bool = True) -> list[Measurement]:
    """Load all measurements, reports first, de-duplicated."""
    measurements = load_from_reports(input_dir)
    seen = {m.dedup_key for m in measurements}
    print(f"  {len(measurements)} timed cases from evaluation reports")

    if use_cache:
        added = 0
        for m in load_from_cache(input_dir):
            if m.dedup_key not in seen:
                seen.add(m.dedup_key)
                measurements.append(m)
                added += 1
        print(f"  {added} further timed cases from the answer cache")

    return measurements


def group_by_agent(measurements: list[Measurement]) -> dict[str, list[float]]:
    """Latencies per agent, each list sorted ascending."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for m in measurements:
        grouped[m.agent].append(m.latency)
    return {agent: sorted(values) for agent, values in grouped.items()}


# ── Statistics ────────────────────────────────────────────────────────────


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


@dataclass(frozen=True)
class AgentSummary:
    """Latency statistics for one agent."""

    agent: str
    n: int
    minimum: float
    median: float
    mean: float
    p90: float
    maximum: float
    stdev: float

    @property
    def too_few(self) -> bool:
        """True when the sample is too small for the shape to mean anything."""
        return self.n < MIN_MEANINGFUL_SAMPLES

    def as_row(self) -> dict[str, object]:
        """Flat mapping for CSV output."""
        return {
            "agent": self.agent,
            "n": self.n,
            "min": round(self.minimum, 2),
            "median": round(self.median, 2),
            "mean": round(self.mean, 2),
            "p90": round(self.p90, 2),
            "max": round(self.maximum, 2),
            "stdev": round(self.stdev, 2),
        }


CSV_FIELDS = ["agent", "n", "min", "median", "mean", "p90", "max", "stdev"]


def summarise(grouped: dict[str, list[float]]) -> list[AgentSummary]:
    """Per-agent summaries, ordered by median latency descending."""
    summaries = [
        AgentSummary(
            agent=agent,
            n=len(values),
            minimum=values[0],
            median=statistics.median(values),
            mean=statistics.fmean(values),
            p90=percentile(values, 0.90),
            maximum=values[-1],
            # Sample stdev is undefined for n=1.
            stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        )
        for agent, values in grouped.items()
    ]
    summaries.sort(key=lambda s: s.median, reverse=True)
    return summaries


def print_summary(summaries: list[AgentSummary]) -> None:
    """Print the summary table to stdout."""
    header = (
        f"{'agent':<16}{'n':>5}{'min':>9}{'median':>9}"
        f"{'mean':>9}{'p90':>9}{'max':>9}{'stdev':>9}"
    )
    print()
    print(header)
    print("-" * len(header))
    for s in summaries:
        flag = "  (n too small)" if s.too_few else ""
        print(
            f"{s.agent:<16}{s.n:>5}"
            f"{s.minimum:>8.1f}s{s.median:>8.1f}s"
            f"{s.mean:>8.1f}s{s.p90:>8.1f}s"
            f"{s.maximum:>8.1f}s{s.stdev:>8.1f}s{flag}"
        )


def write_csv(summaries: list[AgentSummary], path: Path) -> None:
    """Write the summary table beside the figures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.as_row())


# ── Figures ───────────────────────────────────────────────────────────────


def _style() -> None:
    """Slide-friendly defaults: large type, light grid, no chart junk."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#666666",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def _bin_edges(all_values: list[float], bins: int, log_x: bool) -> list[float]:
    """Shared bin edges so every panel is directly comparable."""
    low = min(all_values)
    high = max(all_values)
    if log_x:
        # Latencies can be sub-second; clamp so log10 stays finite.
        low = max(low, 0.1)
        return [
            10 ** (math.log10(low) + i * (math.log10(high) - math.log10(low)) / bins)
            for i in range(bins + 1)
        ]
    return [low + i * (high - low) / bins for i in range(bins + 1)]


def plot_histograms(
    grouped: dict[str, list[float]],
    summaries: list[AgentSummary],
    output: Path,
    bins: int,
    log_x: bool,
) -> None:
    """Small-multiples grid, one histogram per agent, shared axes."""
    import matplotlib.pyplot as plt

    _style()
    order = [s.agent for s in summaries]
    all_values = [v for values in grouped.values() for v in values]
    edges = _bin_edges(all_values, bins, log_x)

    columns = 3
    grid_rows = math.ceil(len(order) / columns)
    fig, axes = plt.subplots(
        grid_rows,
        columns,
        figsize=(15, 4.2 * grid_rows),
        sharex=True,
        sharey=True,
    )
    flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for index, agent in enumerate(order):
        ax = flat[index]
        values = grouped[agent]
        colour = AGENT_COLOURS.get(agent, FALLBACK_COLOUR)

        ax.hist(values, bins=edges, color=colour, edgecolor="white", linewidth=0.8)

        median = statistics.median(values)
        ax.axvline(
            median,
            color="#222222",
            linestyle="--",
            linewidth=1.6,
            label=f"median {median:.1f}s",
        )

        thin = len(values) < MIN_MEANINGFUL_SAMPLES
        suffix = "  — too few to characterise" if thin else ""
        ax.set_title(
            f"{agent}  (n={len(values)}){suffix}", loc="left", fontweight="bold"
        )
        ax.legend(frameon=False, fontsize=10, loc="upper right")
        if log_x:
            ax.set_xscale("log")

    for ax in flat[len(order) :]:
        ax.set_visible(False)

    # Label only the lowest visible panel in each column. `sharex` hides tick
    # labels on the panels above, and an axis label with no ticks under it reads
    # as though the scale differs between rows.
    for column in range(columns):
        lowest = max(
            (i for i in range(len(order)) if i % columns == column), default=None
        )
        if lowest is not None:
            flat[lowest].set_xlabel("response latency (seconds)")
            flat[lowest].tick_params(labelbottom=True)
    for row_index in range(grid_rows):
        flat[row_index * columns].set_ylabel("number of evaluation cases")

    fig.suptitle(
        "Response latency by agent",
        fontsize=18,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.02,
        0.005,
        f"{len(all_values)} timed evaluation cases, pooled across runs. "
        "Cached (reused) answers excluded — no model call was made.",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"  wrote {display_path(output)}")


def plot_distribution(
    grouped: dict[str, list[float]],
    summaries: list[AgentSummary],
    output: Path,
) -> None:
    """Box plot across agents, ordered by median, with the raw points shown.

    The points matter: with 6-48 samples per agent a box plot alone implies more
    data than exists, and one agent here has too few to characterise at all.
    """
    import matplotlib.pyplot as plt

    _style()
    order = [s.agent for s in summaries][::-1]  # highest median at top
    data = [grouped[agent] for agent in order]

    fig, ax = plt.subplots(figsize=(13, 0.95 * len(order) + 2.5))

    box = ax.boxplot(
        data,
        orientation="horizontal",  # renamed from `vert` in matplotlib 3.11
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#111111", "linewidth": 2},
        whiskerprops={"color": "#666666"},
        capprops={"color": "#666666"},
        flierprops={"marker": ""},  # outliers are drawn as raw points instead
    )
    for patch, agent in zip(box["boxes"], order):
        patch.set_facecolor(AGENT_COLOURS.get(agent, FALLBACK_COLOUR))
        patch.set_alpha(0.35)
        patch.set_edgecolor(AGENT_COLOURS.get(agent, FALLBACK_COLOUR))

    rng_offsets = (-0.16, -0.08, 0.0, 0.08, 0.16)
    for position, (agent, values) in enumerate(zip(order, data), start=1):
        colour = AGENT_COLOURS.get(agent, FALLBACK_COLOUR)
        # Deterministic jitter — no RNG, so the figure is reproducible.
        offsets = [rng_offsets[i % len(rng_offsets)] for i in range(len(values))]
        ax.scatter(
            values,
            [position + offset for offset in offsets],
            s=26,
            color=colour,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([f"{a}\n(n={len(grouped[a])})" for a in order])
    ax.set_xlabel("response latency (seconds)")
    ax.set_title(
        "Latency distribution by agent", loc="left", fontsize=17, fontweight="bold"
    )
    fig.text(
        0.01,
        0.01,
        "Box = interquartile range, line = median, whiskers = 1.5×IQR. "
        "Every timed case is drawn as a point.",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"  wrote {display_path(output)}")


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot per-agent latency histograms from the evaluation reports.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bins", type=int, default=14, help="histogram bins (default 14)"
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="log-scale the x axis; latency is right-skewed",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="use the evaluation reports only"
    )
    parser.add_argument(
        "--show", action="store_true", help="open the figures interactively"
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"No such directory: {args.input_dir}", file=sys.stderr)
        return 1

    print(f"Reading {display_path(args.input_dir)}")
    measurements = collect(args.input_dir, use_cache=not args.no_cache)
    if not measurements:
        print(
            "No timed cases found. Every stored case was reused from the cache, "
            "so no latency was ever recorded. Run:\n"
            "    uv run python -m src.evaluation --no-cache\n"
            "to produce fresh measurements (this spends API quota).",
            file=sys.stderr,
        )
        return 1

    grouped = group_by_agent(measurements)
    summaries = summarise(grouped)
    print_summary(summaries)

    print()
    csv_path = args.output_dir / "latency_summary.csv"
    write_csv(summaries, csv_path)
    print(f"  wrote {display_path(csv_path)}")

    plot_histograms(
        grouped,
        summaries,
        args.output_dir / "latency_histograms.png",
        args.bins,
        args.log_x,
    )
    plot_distribution(grouped, summaries, args.output_dir / "latency_distribution.png")

    thin = [s.agent for s in summaries if s.too_few]
    print()
    print("Read before quoting these numbers:")
    print(
        "  - Samples pool every evaluation run on disk, spanning code changes and\n"
        "    provider conditions. They describe latency as observed, not a controlled\n"
        "    benchmark of the current code."
    )
    print(
        "  - Latency is dominated by the number of ReAct steps an agent takes, which\n"
        "    varies per question. Spread within an agent is expected, not noise."
    )
    print(
        "  - Groq rate limiting and retries are included in these timings, so the\n"
        "    long tail partly measures the provider rather than the system."
    )
    if thin:
        print(
            f"  - Too few samples to characterise: {', '.join(thin)}. "
            "Do not present a shape for these."
        )

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
