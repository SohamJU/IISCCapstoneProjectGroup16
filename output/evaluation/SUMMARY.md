# Evaluation Summary

_Generated 2026-07-31 11:34 UTC_

## Headline

| Metric | Value |
|---|---:|
| Cases authored | 64 |
| Cases scored | 62 |
| Cases passed | 62 |
| **Overall pass rate** | **100%** |
| Intent-routing accuracy | 94% |
| — single-intent | 100% |
| — multi-intent | 60% |
| Skipped (would mutate the database) | 1 |
| Unscored (provider errors) | 1 |

## Per-agent results

| Agent | Scored | Passed | Pass rate | Check score | Mean latency |
|---|---:|---:|---:|---:|---:|
| escalation | 7 | 7 | 100% | 100% | — |
| fallback | 6 | 6 | 100% | 100% | — |
| order | 21 | 21 | 100% | 100% | — |
| product | 9 | 9 | 100% | 100% | — |
| recommendation | 7 | 7 | 100% | 100% | — |
| return | 12 | 12 | 100% | 100% | — |
| **Total** | **62** | **62** | **100%** | | |

**Pass rate** counts a case only when *every* check on it passed. **Check score** is the proportion of individual checks passed, and shows how near the failures were.

## By dimension

| Dimension | Passed | Rate | What it verifies |
|---|---:|---:|---|
| Core functionality | 19/19 | 100% | Ordinary requests answered correctly from real data |
| Access control | 21/21 | 100% | One customer cannot reach or be shown another's data |
| Hallucination resistance | 11/11 | 100% | No invented orders, prices, policies or identifiers |
| Edge cases | 15/15 | 100% | Malformed input, impossible constraints, expired windows |
| Scope and UX | 10/10 | 100% | Stays in domain, never asks for internal identifiers |

Dimensions are derived from case tags and overlap: a case tagged both `security` and `edge` counts in both rows.

## Intent routing

Scored against a hand-authored dataset covering all six routes, multi-intent requests, boundary cases and requests outside the agent taxonomy. Each case declares which routes must be predicted, which are merely acceptable, and which are forbidden.

- **Overall accuracy: 94%** (n=32)
- Single-intent: 100% (n=27)
- Multi-intent: 60% (n=5) — **every** required route must appear, so this is the harder measure.
- Over-routing: 0% of cases received an extra route that was not asked for. Tracked because a router that always answers with two routes would otherwise pass by covering every possibility.

This dataset replaced the generated `customer_queries` table, whose labels were unreliable — queries labelled `product_search` read "it is missing parts and I need to...", which is a return. Scoring against them measured label noise as much as router quality. The hand-authored set is smaller but every case is deliberate.

**Caveat.** The router is not deterministic, so a small dataset moves between runs. Treat a single-case change as noise rather than as a regression.

| Route | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| escalation | 100% | 100% | 1.00 | 4 |
| fallback | 100% | 100% | 1.00 | 3 |
| order | 100% | 90% | 0.95 | 10 |
| product | 100% | 83% | 0.91 | 6 |
| recommendation | 100% | 100% | 1.00 | 4 |
| return | 100% | 100% | 1.00 | 6 |

## Coverage

Cases by tag: `security` (20), `happy-path` (18), `edge` (15), `hallucination` (8), `cross-account` (7), `ux` (5), `third-party` (4), `grounding` (3), `unauthenticated` (3), `attribution` (3), `sql` (3), `negative` (2)

## Outstanding failures

None — every scored case passed.

## Method

Scoring is deterministic. Each case asserts checkable properties — which tools ran, which facts appear, which must not, whether a refusal occurred, and whether any cited order identifier is real. There is no model-as-judge, so the same dataset against the same code yields the same score and a change in the number reflects a change in behaviour.

Agents are evaluated directly rather than through the supervisor graph, so a failure is attributable to the agent rather than to routing. Routing is measured separately against the labelled query dataset.

94 of the scored cases reused an answer recorded in an earlier run rather than calling the model again, to stay within the provider's quota. Checks were re-applied fresh to every stored answer; only the answer text was reused, and a changed question always forces a new call.

## What these numbers do not show

- **Response quality is not measured.** A terse but correct answer and a well-structured one score identically; only correctness, tool selection and safety are checked.
- **The cases were authored alongside the implementation**, so they encode intended behaviour rather than independently specified requirements. A full pass is a regression baseline, not proof of general reliability.
- **Retrieval metrics are absent.** Context precision and faithfulness for the RAG paths are not implemented.
- **The routing dataset is small and hand-authored.** It covers every route deliberately, but 32 cases cannot characterise the full space of customer phrasing, and the cases reflect the authors' idea of what a correct routing decision is.
- **A 100% pass rate means the suite found nothing, not that nothing is there.** Every defect fixed during development was first found by a case being added; the score reflects coverage as much as quality.
