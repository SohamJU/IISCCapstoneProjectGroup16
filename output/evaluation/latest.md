# Agent Evaluation Report

_Generated 2026-07-31 13:16 UTC_

Scoring is deterministic: each case asserts which tools must run, which facts must appear, which must not, and whether a refusal was required. A case passes only if every one of its checks passes.

## Per-agent results

| Agent | Cases | Passed | Pass rate | Check score | Mean latency |
|---|---:|---:|---:|---:|---:|
| **Overall** | **0** | **0** | **0%** | | |

## Intent routing

Scored against a hand-authored dataset covering all six routes, single- and multi-intent requests, boundary cases and requests outside the agent taxonomy. Each case states exactly which routes must be predicted, which are merely acceptable, and which are forbidden.

- **Overall accuracy: 94%** (n=32)
- Single-intent: 100% (n=27)
- Multi-intent: 60% (n=5) — every required route must appear in the prediction.
- Over-routing: 0% of cases were given an extra route the case did not ask for.

| Route | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| escalation | 100% | 100% | 1.00 | 4 |
| fallback | 100% | 100% | 1.00 | 3 |
| order | 100% | 90% | 0.95 | 10 |
| product | 100% | 83% | 0.91 | 6 |
| recommendation | 100% | 100% | 1.00 | 4 |
| return | 100% | 100% | 1.00 | 6 |
