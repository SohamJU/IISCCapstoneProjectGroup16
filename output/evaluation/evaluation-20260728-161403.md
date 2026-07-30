# Agent Evaluation Report

_Generated 2026-07-28 16:14 UTC_

Scoring is deterministic: each case asserts which tools must run, which facts must appear, which must not, and whether a refusal was required. A case passes only if every one of its checks passes.

## Per-agent results

| Agent | Cases | Passed | Pass rate | Check score | Mean latency |
|---|---:|---:|---:|---:|---:|
| escalation | 5 | 5 | 100% | 100% | 9.0s |
| fallback | 4 | 4 | 100% | 100% | 0.0s |
| order | 13 | 13 | 100% | 100% | 14.8s |
| product | 6 | 6 | 100% | 100% | 37.6s |
| recommendation | 5 | 4 | 80% | 89% | 32.1s |
| return | 9 | 9 | 100% | 100% | 38.2s |
| **Overall** | **42** | **41** | **98%** | | |

> **1 case(s) could not be scored** because of provider errors (rate limit, daily quota or connection). They are excluded from the rates above — the scores describe only cases that reached the model.

- `product` / product-no-order-tools: APIStatusError: Error code: 413 - {'error': {'message': 'Request too large for model `openai/gpt-oss-120b` in organization `org_01km9w1qj9ecyayxd749ezmw6h` serv

## Failures

### `recommendation` — rec-personalised
_Grounds suggestions in the signed-in customer's own history._

**Query:** Based on what I've bought before, what should I get next?

**Error:** `BadRequestError: Error code: 400 - {'error': {'message': 'Tool call validation failed: tool call validation failed: parameters for tool recommend_for_customer did not match schema: errors: [`/budget`: expected number, but got null]', 'type': 'invalid_request_error', 'code': 'tool_use_failed', 'failed_generation': '{"name": "recommend_for_customer", "arguments": {\n  "budget": null,\n  "limit": 5\n}}'}}`

- Failed `non_empty_answer` — agent produced no answer
- Failed `calls_any:recommend_for_customer|get_customer_order_history|get_customer_profile` — tools actually called: none

**Tools called:** `none`

**Answer:**

```
(empty)
```

## Intent routing

The labelled dataset uses 17 intents; this system has 6 routes. Intents with exactly one defensible route form the core set and produce the headline figure. Intents describing work this system has no agent for (billing, account changes, loyalty, general complaints) are scored against a set of acceptable routes and reported separately.

- **Core routing accuracy (strict): 64%** (n=33) — the row's primary intent must be predicted.
- Core routing accuracy (lenient): 70% — any intent listed in the row's `all_intents` is accepted. Most rows are genuinely multi-intent, so the strict figure partly measures which intent the dataset happened to nominate as primary.
- Ambiguous-intent acceptance: 22% (n=18)

| Route | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| escalation | 0% | 0% | 0.00 | 0 |
| order | 73% | 53% | 0.62 | 15 |
| product | 100% | 33% | 0.50 | 6 |
| recommendation | 67% | 67% | 0.67 | 3 |
| return | 50% | 100% | 0.67 | 9 |
