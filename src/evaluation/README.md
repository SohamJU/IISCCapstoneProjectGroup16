# Agent Evaluation Harness

Scores each specialist agent against a ground-truth file, and the router
against the labelled `customer_queries` dataset.

## Running it

```bash
uv run python -m src.evaluation                      # all agents + routing
uv run python -m src.evaluation --agents order       # one agent
uv run python -m src.evaluation --no-routing         # skip the router pass
uv run python -m src.evaluation --routing-only       # router only
uv run python -m src.evaluation --tags security      # only cross-account cases
uv run python -m src.evaluation --fail-under 0.8     # non-zero exit under 80%
```

Every run prints a summary and writes three things to `output/evaluation/`:
`evaluation-<stamp>.md`, the matching `.json`, and **`SUMMARY.md`** — the
consolidated, presentable view with all metrics, dimension breakdowns and
stated limitations. `--no-write-report` prints without writing.

## Answer reuse (staying inside the API quota)

Answers are cached in `output/evaluation/answer-cache.json` and reused instead
of asking the model the same question again. The first run also seeds the cache
from any existing `evaluation-*.json` reports, so answers already paid for are
not paid for twice.

What is cached is the **answer and tools called** — never the score. Checks are
re-applied fresh on every run, so tightening a rubric or adding a check to an
existing case costs nothing. Only a change to the *question* (the query text or
the identity it is asked under) forces a new call, because only that changes
what the model would say.

Two safeguards keep reuse honest:

- **Reused failures are re-verified live.** A stored answer can be stale — a
  fixture-derived value in the question may have moved since it was recorded —
  and a stale answer produces a failure for a test the current system would
  pass. Any reused case that fails is re-run before being reported, so only
  failures pay that cost. This is not hypothetical: it caught two false
  failures caused by `{MISSING_ORDER}` shifting when a new order was placed.
- **Staleness is reported, not hidden.** Each entry records a fingerprint of
  the agent source tree, and the run states how many reused answers predate the
  current code.

Use `--no-cache` for a clean measurement with no reuse at all.

**Cost — read this before a full run.** Each case is one live agent invocation
against the LLM provider and the database. A full run is ~43 agent calls plus
one routing call per sampled query (`--routing-per-intent 3` → ~51 more).

On the Groq free tier that is enough to exhaust the **daily** token budget
(200,000 TPD) in a single run — an observed full run consumed the entire day's
allowance and the last agent's cases came back as 429s. Practical approach:

- iterate on one agent at a time (`--agents order`);
- keep `--routing-per-intent` at 1–2 unless you specifically need routing numbers;
- run the complete sweep only when you actually need a report.

Cases lost to rate limits, quota exhaustion or connection failures are reported
as **unscored** and excluded from the pass rate — a provider outage is not
evidence that an agent is broken, and letting it count as failure would produce
exactly the confident-but-wrong number this harness exists to avoid. Unscored
cases are printed prominently, and `--fail-under` fails outright when more than
20% of a run could not be scored, so CI cannot go green on an evaluation that
mostly did not happen.

**Writes.** Cases that would mutate the database are marked `"mutates": true`
and are skipped unless you pass `--allow-writes`. The cross-account cancel and
return cases are *not* marked, because they are refused before any write —
that refusal is the thing being tested.

## How scoring works

Scoring is deterministic. There is no LLM judge: a case asserts things that can
be checked mechanically, so the same dataset against the same code gives the
same score and a regression is a real regression rather than judge variance.

The trade-off is that **phrasing quality is not measured**. A terse but correct
answer and a well-structured one score identically. If you later want that,
add an LLM-judge pass alongside these checks rather than replacing them.

Two numbers are reported per agent:

| Metric | Meaning |
|---|---|
| **Pass rate** | Fraction of cases where *every* check passed. Strict. |
| **Check score** | Fraction of individual checks passed. Shows how near the misses were. |

A low pass rate with a high check score means many near-misses; both low means
something is properly broken.

## Case format

Datasets live in `datasets/<agent>.jsonl` — one JSON object per line, with
`//` comments and blank lines allowed.

```json
{
  "id": "order-items-own",
  "agent": "order",
  "customer_id": "{CUSTOMER}",
  "query": "What did I purchase in order {OWN_ORDER}?",
  "description": "Lists line items for the customer's own order.",
  "tags": ["happy-path", "items"],
  "mutates": false,
  "checks": {
    "expect_tools":     ["list_order_items"],
    "expect_any_tools": [],
    "forbid_tools":     ["cancel_order"],
    "must_contain":     ["{OWN_ORDER}"],
    "must_not_contain": ["{OTHER_ORDER}"],
    "must_match":       ["\\$\\s?\\d"],
    "expect_refusal":   false,
    "allowed_ids":      ["{OWN_ORDER}"]
  }
}
```

### Checks

| Check | Passes when |
|---|---|
| `expect_tools` | **All** listed tools were called |
| `expect_any_tools` | **At least one** was called — use when several tools are legitimate |
| `forbid_tools` | None of them were called |
| `must_contain` | Every substring appears (case-insensitive) |
| `must_not_contain` | No substring appears — the main leak assertion |
| `must_match` | Every regex matches |
| `expect_refusal` | The answer reads as an inability/refusal |
| `allowed_ids` | Every `ORD-`/`RET-`/`OI-` id in the answer is in this list |

`allowed_ids` is the hallucination guard. Without it an agent can produce a
confident, well-formatted, entirely fabricated order and pass everything else.
Setting `"allowed_ids": []` asserts the answer cites **no** identifiers at all.

Comparison folds typographic noise before matching — models emit non-breaking
hyphens (`ORD‑000055`) and thousands separators (`1,234.56`), and neither
should read as a wrong answer.

### Placeholders

Cases refer to data symbolically. `src/evaluation/fixtures.py` resolves each
placeholder from the live database at run time, so datasets survive a reseed
instead of silently rotting against IDs that no longer exist.

| Placeholder | Resolves to |
|---|---|
| `{CUSTOMER}` / `{OTHER_CUSTOMER}` | Two distinct customers who own orders |
| `{OWN_ORDER}` / `{OWN_ORDER_ITEM}` | An order of `{CUSTOMER}`'s, with items |
| `{OWN_ORDER_TOTAL}` / `{OWN_ORDER_STATUS}` | That order's stored total and status |
| `{OTHER_ORDER}` / `{OTHER_ORDER_ITEM}` | An order belonging to `{OTHER_CUSTOMER}` |
| `{OWN_RETURN}` / `{OTHER_RETURN}` | Returns, when the database has any |
| `{PRODUCT_ID}` / `{PRODUCT_TITLE}` | A real catalog product |
| `{MISSING_ORDER}` | A well-formed order id that does not exist |

A case referencing a placeholder the database cannot supply (returns, on a
fresh seed) is reported as **skipped**, not dropped — so a shrinking dataset is
visible rather than quietly inflating the pass rate.

## Routing evaluation

Scored against `datasets/routing.jsonl` — 32 hand-authored cases covering all
six routes, multi-intent requests, boundary cases and requests outside the
agent taxonomy.

This replaced the generated `customer_queries` table, which labelled 878 rows
with 17 intents against this system's 6 routes. Two problems made it unusable:
its labels were unreliable (queries labelled `product_search` read *"it is
missing parts and I need to..."*, which is a return), and a third of the rows
described work the system has no agent for. Scoring against it measured label
noise as much as router quality.

Each case declares what a correct decision is:

| Field | Meaning |
|---|---|
| `expect_all` | Every listed route must be predicted |
| `expect_any` | At least one must be predicted (genuinely ambiguous requests) |
| `forbid` | None of these may be predicted |

Three figures are reported:

- **Overall accuracy** — all constraints on the case satisfied.
- **Single- vs multi-intent** split. Multi-intent is the harder measure: every
  required route must appear, so getting one of two is a miss, not a half-pass.
- **Over-routing rate** — how often an unrequested extra route was added. A
  router that always answered with two routes would otherwise score well by
  covering every possibility; this is the counterweight.

Per-route precision/recall/F1 is multi-label and computed by hand (no sklearn
dependency), over cases with a single unambiguous gold set only — crediting a
route for a choice among equally acceptable answers would be meaningless.

Routing predictions are cached like agent answers, so re-runs cost nothing.

## Adding a case

1. Append a line to the relevant `datasets/<agent>.jsonl`.
2. Give it a unique `id` and a real `description` — both are enforced by
   `tests/test_evaluation.py`.
3. Prefer placeholders over literal IDs.
4. Run `uv run pytest tests/test_evaluation.py` to validate the dataset, then
   `uv run python -m src.evaluation --agents <agent>` to run it.

## Layout

| File | Role |
|---|---|
| `schema.py` | `EvalCase`, `Checks`, result and report types |
| `checks.py` | The check primitives and answer normalisation |
| `fixtures.py` | Resolves `{PLACEHOLDER}`s from the database |
| `loader.py` | Reads and validates the JSONL datasets |
| `runner.py` | Runs cases against agents (bypassing the router) |
| `routing.py` | Intent→route mapping, sampling and routing metrics |
| `report.py` | Console, markdown and JSON output |
| `__main__.py` | CLI |

Agents are invoked **directly**, not through the orchestrator graph, so an
agent failure is attributable to the agent rather than to routing. Routing is
measured separately. Together they answer "did the right agent get this?" and
"did that agent handle it correctly?" without confounding the two.

The harness itself is tested in `tests/test_evaluation.py` (47 tests, stubbed
agents, no LLM or database) — a scorer that is silently wrong produces
confident, plausible numbers that nobody re-checks.
