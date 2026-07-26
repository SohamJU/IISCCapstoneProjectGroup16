# Customer Support Chatbot — Rebuild Plan & Change Log

**Date:** 2026-07-25
**Branch:** `agents_evaluation_richa`
**Scope:** Fix answer quality, port orchestration to LangGraph, make failures visible.

---

## 0. The headline finding

The brief was *"this code is not in LangGraph orchestration and doesn't give very
good answers."* Half of that was already true, and the causal link was not.

- The **specialist agents were already LangGraph** — every one was
  `create_react_agent` with a `MemorySaver`. What wasn't LangGraph was the
  *supervisor*, a hand-written `for route in routes:` loop.
- **Porting that loop to a `StateGraph` fixes none of the bad answers on its
  own.** The loop was functionally equivalent to a supervisor graph.

The bad answers came from ~10 specific, individually-diagnosable defects. This
plan fixes those first, then does the LangGraph port for the things a graph
genuinely buys (shared state, one checkpointer, streaming-ready).

Four defects were severe enough that they alone explain most of the symptom:

| # | Defect | Effect |
|---|---|---|
| 1 | Router keyword-scanned the **conversation history** | One escalation permanently hijacked the session |
| 2 | Router parsed LLM output with `if label in text` | One question fanned out to 3-4 agents |
| 3 | `ON CONFLICT (session_id)` matched no constraint | **Conversation persistence never worked, at all** |
| 4 | Product prompt asserted "there is no category column" | False premise; the column exists. Drove 700 words of broken SQL heuristics |

---

## 1. Baseline before any change

```
5 failing tests (already broken on the branch)
PINECONE_API_KEY unset  -> review search returned an error string to the model
customer_sessions       -> every save silently failed
product_catalog         -> 194,318 rows, zero indexes
```

---

## 2. Changes, in the order they were made

### 2.1 `src/utils/logger.py` — implement the logger *(was a one-line stub)*

The file contained only a docstring. Multiple `except Exception: pass` blocks
hid real misconfiguration behind canned replies, so a broken API key looked
like a model-quality problem.

- Added `get_logger()` with a single configured stderr handler.
- `SUPPORT_LOG_LEVEL=DEBUG` for verbose agent tracing.
- Silenced `httpx`/`httpcore` noise.

### 2.2 `src/agents/llm/llm_provider.py` — model configuration

| Change | Before | After | Why |
|---|---|---|---|
| `max_tokens` | 1024 | 4096 | Agents had to emit a long SQL query *and* a full answer inside one completion; 1024 truncated both |
| Rate limiter | New instance per `get_llm()` call | One process-wide singleton | Each agent independently burned the full quota, so the "limit" was never an application-wide ceiling |
| Requests/sec | 0.5 (= 2s per call) | 2.0, via `SUPPORT_LLM_RPS` | A 3-route turn cost 15+ seconds of pure rate-limiting |
| `reasoning_effort` | not set | `medium` for agents, `low` for routing | See measurement below |

**Reasoning-effort measurement** on *"show me wireless headphones under $250"*:

| Setting | Time | Tool calls | Result quality |
|---|---|---|---|
| default | 51s | 3 | good |
| `low` | 41s | 2 | **bad** — picked wrong filter args, surfaced no-name products |
| `medium` | **30s** | **1** | good — premium, correctly ranked |

`medium` wins on both axes: better reasoning picks the right tool arguments
first time, so the ReAct loop terminates in one round-trip instead of three.
Cutting effort below this is a false economy.

Added `get_router_llm()` and `get_synthesis_llm()` so routing and synthesis
cost can be tuned independently of the agents.

### 2.3 `src/agents/router/` — the single highest-impact fix

**New file `schemas.py`** — Pydantic `RouteDecision` with a `Literal` route type.

**`agent.py` rewritten.** Two bugs removed:

**(a) Sticky escalation.** The orchestrator passed a history-enriched blob to
`classify_multi`, which ran the escalation regex over *all of it*. Once any
turn contained `human`, `escalate`, `angry` — **including the assistant's own
reply** — every subsequent turn hard-routed to escalation at 0.98 confidence.
A question about product pricing returned a human-handoff summary.

Verified before the fix:

```
poisoned history + "what is the price of the sony headphones"
  -> {'routes': ['escalation'], 'confidences': {'escalation': 0.98}}
```

Fix: keyword safety checks run on the **current user message only**. History is
passed separately and used solely for reference resolution ("cancel *it*").
After the fix the same input returns `['product']` at 0.99.

The escalation regex was also narrowed. It previously fired on `complaint` and
`angry`, which are ordinary support tone, not a request for a human.

**(b) Substring parsing.** `if label in text` over the raw model reply matched
three labels from one prose sentence ("this is a product question, not an order
or return issue"). Three routes → three agents → three partial answers stitched
into a wall of text.

Fix: `with_structured_output(RouteDecision)`. No text parsing. The model also
now reports **genuine confidence** instead of a hardcoded `0.7`.

**Also added:**
- `MAX_ROUTES_PER_TURN = 2` hard cap.
- `_normalise_routes()` — validates labels, de-duplicates, promotes escalation
  to first, drops `fallback` when a real specialist was also chosen.
- Explicit **product vs recommendation** disambiguation in the prompt. A stated
  budget or category is a *search* (product); only history/profile-dependent
  requests are *recommendation*. This was misrouting "show me headphones under
  $250" to the recommendation agent, which then demanded a customer ID.
- Keyword fast paths for unambiguous inputs (`ORD-000123` + a tracking verb),
  which skip the router LLM call entirely.
- `init_error` is now recorded and surfaced instead of silently disabling LLM
  routing.

### 2.4 `src/rag/policy_store.py` — working policy retrieval *(new)*

`PINECONE_API_KEY` is unset, so every call into `src/rag/retriever.py` raised.
The return agent's policy tool silently fell through to a naive line scan: it
OR-matched any query token over two characters against every line and returned
up to eight orphan bullet lines with no headings. The agent got fragments and
either hedged or invented policy terms.

The knowledge base is only ~390 lines across four files, so a local index is
both practical and better:

- Documents split on `##` headings → a match returns a **coherent section**
  ("## Return Window" with all its bullets), not one line.
- TF-IDF scoring with heading boosts and per-document topic boosts.
- Pinecone still preferred when configured; this is the fallback.

Verified — correct document selected for every probe:

```
"how many days do I have to return an item" -> return_policy.md > Return Process
"is shipping free"                          -> shipping_policy.md > Shipping Methods & Costs
"what does the warranty cover"              -> warranty_policy.md > Warranty Claim Process
"can I pay with EMI"                        -> payment_policy.md > Buy Now, Pay Later (BNPL)
```

### 2.5 `src/agents/shared_tools.py` — `lookup_support_policy` *(new)*

Policy lookup existed **only** on the Return Agent, yet every agent's prompt
said "never invent policy terms" and "use tools for all policy details". The
Order Agent had no way to answer *"what's your cancellation policy?"* — it
either refused or made something up.

One retrieval tool now covers returns, shipping, warranty and payments, and is
bound to the Order, Product and Return agents.

### 2.6 `src/agents/product_agent/` — the false-premise fix

The prompt stated:

> *"Since there is no category column, you must enforce this strictly through
> advanced text matching and numeric heuristics"*

**This is false.** `product_catalog` has `main_category` (All Electronics,
Computers, Camera & Photo, Home Audio & Theater, …) **and** `sub_categories`.
That mistaken belief generated ~700 words instructing the model to hand-write
`NOT ILIKE '%case%' AND NOT ILIKE '%charger%' …` chains and invent price floors
inline in SQL — the largest single source of malformed queries and empty
results.

**`tools.py`** — accessory filtering, price bounds, rating floors and result
shaping moved out of the prompt into Python:

- **`search_products`** *(new, now the default tool)* — structured parameters:
  `min_price`, `max_price`, `min_rating`, `category`, `exclude_accessories`,
  `bestsellers_only`, `sort_by`, `limit`. Accessory detection is a Python
  predicate over ~35 terms plus subordinating patterns (`" for "`,
  `"compatible with"`). Falls back gracefully — if filtering removes
  everything, the customer was probably shopping *for* an accessory, so results
  are returned with a note rather than a dead end.
- **`get_product_details`** *(new)* — single-product lookup, with description
  and features truncated so one call can't blow the context window.
- **`list_product_categories`** *(new)* — so the model can discover valid
  `category` values instead of guessing.
- **`search_products` now returns a `snippet`** (first 350 chars of
  features/description). Without it the model followed every search with N
  `get_product_details` calls just to explain *why* a product fits, tripling
  the ReAct steps.
- **`search_product_reviews`** — previously returned
  `"Review retrieval error: PINECONE_API_KEY is not set."` straight into the
  model's context, where it was interpreted as review data or retried in a
  loop. Now returns an explicit instruction not to retry and to use
  `average_rating`/`rating_count` instead.
- Row cap raised 5 → 10 (`search_products`) / 8 (`query_products`). Five rows
  is not enough to recommend from.
- `query_products` (raw SQL) retained, but demoted to an explicit escape hatch
  for analytical questions.

Verified:

```
laptop, max_price=500   -> Acer Aspire 3 $439.99 (4.5), HP Chromebook $214 (4.4),
                           Surface Laptop Go $487.10 (4.5), Pixelbook Go $498 (4.4)
wireless headphones     -> Bose QC35 II $344.78 (4.7), Beats Powerbeats Pro $199.95 (4.5),
                           Sony WH-1000XM3 $229.98 (4.6)
```

Real devices, correct prices, no cases or chargers.

**`prompts.py`** rewritten — false premise removed, heuristics deleted, replaced
with a short "which tool when" section and grounding rules.

### 2.7 `src/data/session_persistence.py` — persistence was 100% broken

```
[✗] Error saving session to database:
    there is no unique or exclusion constraint matching the ON CONFLICT specification
```

The upsert targeted `ON CONFLICT (session_id)`. The live table's only unique
constraint is `UNIQUE (customer_id, session_id)`. The `CREATE TABLE IF NOT
EXISTS` DDL that would have added a `session_id UNIQUE` never ran, because the
table already existed. **Every single turn failed to save, silently.**

- Upsert changed to `ON CONFLICT (customer_id, session_id)`.
- DDL now explicitly creates the index the upsert depends on.

Result: `tests/test_session_persistence.py` went from 1 failure to **16 passed**.

### 2.8 `src/data/migrations.py` — schema drift repair *(new)*

`product_catalog` (194,318 rows) had **zero indexes**. Added idempotent
migrations for `product_id`, `price`, `main_category`, plus the
`customer_sessions` unique index. Run with `python -m src.data.migrations`.

### 2.9 `src/agents/base_agent.py` — `SpecialistAgent` *(was a one-line stub)*

The five specialists were near-identical ~95-line copies, each carrying two
defects:

**(a) Per-agent `MemorySaver`.** Combined with the orchestrator prepending a
`CONVERSATION_HISTORY:` block to *every* message, each agent saw its own
private history **plus** a flattened transcript of a conversation it had only
partly taken part in — duplicated and often contradictory.

**(b) Subtask scoping concatenated into the user turn.** `build_subtask_message`
produced `"User request: …\n\nSubtask for Product Agent: Focus only on product
facts…"` and that whole string was stored as a `HumanMessage`. By turn three
the model saw the customer repeatedly issuing meta-instructions to itself.

Both fixed. `SpecialistAgent` is **stateless** — no checkpointer, no history.
The supervisor graph owns the canonical message list and passes a window in.
Per-turn scoping arrives as a `SystemMessage`, which never enters the record.

`AgentResult` also captures which tools ran, for the UI trace.

The five agents collapsed to thin subclasses (~40 lines each, from ~95).

### 2.10 `src/agents/graph/` — the LangGraph supervisor *(new)*

```
              ┌───────────┐
    START ───▶│ guardrail │──(invalid)──────────────┐
              └─────┬─────┘                         │
                    │ (valid)                       │
                    ▼                               │
              ┌────────────┐                        │
        ┌────▶│ supervisor │──(nothing pending)────▶├──▶ synthesize ──▶ END
        │     └─────┬──────┘                        │
        │           │ (next pending route)          │
        │           ▼                               │
        │   product │ order │ return │              │
        │   recommendation │ escalation │ fallback  │
        │           │                               │
        └───────────┘
```

**`state.py` — `SupportState`.** The supervisor is re-entered after every
specialist; that loop is what lets a two-route turn share state.

`facts` is the shared scratchpad the old for-loop could not provide. Previously
"return this order and send a replacement" made the return agent resolve
`ORD-000123`, then the order agent immediately asked the customer for their
order ID again.

**Reducer bug found and fixed during testing.** Fields written by more than one
node need a reducer, otherwise the second writer replaces the first — which is
exactly how `executed_routes` lost the first of two routes (caught by
`test_graph_runs_both_routes_for_compound_request`). Convention adopted:

- update of `None` → **reset** the field (used once per turn),
- any other update → **combine**.

That distinction matters: an agent legitimately returns `facts={}` when it found
no identifiers, and that must not wipe facts an earlier agent established in the
same turn.

**`nodes.py`:**
- `guardrail_node` — validates before any model call.
- `supervisor_node` — classifies on first entry, reports remaining work on
  re-entry. Honours the router's `needs_clarification` (ask once rather than
  guess and answer wrongly).
- `make_agent_node` — builds the scope `SystemMessage`, injects customer
  identity and known facts, extracts new facts from the answer.
- `make_synthesis_node` — **replaces `_merge_responses`**, which emitted:

  > Handled your request in 2 steps: return, order.
  >
  > Step 1 - return (confidence 0.70)
  > *…900 chars, truncated mid-sentence…*

  Now: single-agent turns pass through **untouched with no extra LLM call**;
  multi-agent turns are merged by one synthesis call instructed to preserve
  every concrete fact exactly and never mention agents, steps or confidence.
- `LOW_CONFIDENCE_THRESHOLD` moved to `0.45` and is now **reachable**. At 0.60
  against a hardcoded 0.70 confidence it was dead code.

**`builder.py`** — compiles the graph. Specialists are built **once per process**,
not per session, removing the unbounded `_sessions` dict that leaked six agents,
six LLM clients and six checkpointers per session. One `MemorySaver` for the
whole graph replaces six; swap for `PostgresSaver` to survive restarts.

### 2.11 `src/agents/orchestrator/agent.py` — now a thin adapter

Retains the public `handle()` API, session close semantics, and mirroring turns
into `customer_sessions`. Everything else delegates to the graph.

**Close-chat regex.** The old pattern matched `close|end|stop|exit|quit|bye`
**anywhere** in the message:

- "When does the return window **close**?" → session wiped
- "I want to **stop** my subscription" → session wiped
- "It should arrive by the **end** of the week" → session wiped

Now anchored — it must be the whole message (`^…$`).

**Silent degradation removed.** A bad `GROQ_API_KEY` used to swap all six agents
for `DeterministicSupportAgent` canned strings with no signal anywhere, so
"bad answers" looked like a model problem rather than a config problem. The
fallback still exists but now logs loudly and sets `is_degraded` /
`degraded_reason`, which the UI renders as a banner.

**History seeding.** Stored turns are loaded into graph state **once**, when a
session is first seen — replacing the old approach of prepending
`CONVERSATION_HISTORY:` to every single message.

`OrchestratorResponse` gained `tools_used`, `warnings` and `degraded`.

### 2.12 Deleted stale shadow modules

`src/agents/{router,orchestrator,order_agent,product_agent,return_agent,recommendation_agent}.py`
each contained a single docstring line and shadowed the real package of the
same name. Removed.

### 2.13 `app/gradio_app.py` and `main.py`

- **`load_dotenv()` added to both.** Neither loaded `.env`, so running
  `python main.py` without an exported `GROQ_API_KEY` dropped the whole system
  into deterministic mode.
- **Degraded-mode banner** in the UI — impossible to miss.
- **Routing trace** now shows route, *real* confidence and the tools each agent
  actually called. The old `[routes=… confidence=0.70]` printed a constant.
- Example prompts added for the demo.
- Customer dropdown capped at 500 rows.
- `share=True` removed from `launch()` (was creating a public tunnel on every
  run).

### 2.14 `tests/test_agents.py` — rewritten

Several old tests **asserted the bugs**: that one message should fan out to
three specialists, and that the reply should be a `"Handled your request in N
steps:"` concatenation. Those now assert the corrected behaviour, each with a
comment explaining what changed.

New regression tests cover: sticky escalation, route-count capping, scope as a
`SystemMessage` (not user text), fact sharing between agents, close-chat false
positives, guardrail short-circuiting, and threshold reachability.

```
Before:  5 failed
After:  58 passed
```

### 2.15 `scripts/verify_demo.py` *(new)*

Live end-to-end check of the six scenarios that used to fail, so a regression
is visible **before** a demo rather than during one. Reports `PASS` / `FAIL` /
`BLOCKED`, where `BLOCKED` means the provider quota prevented verification —
never silently counted as a pass.

### 2.16 Provider quota handling *(found during verification)*

The verification run exhausted Groq's free-tier daily cap (200,000 tokens/day)
partway through. That surfaced a real UX defect: a `429` propagated to the
customer as the cheerful generic fallback —

> *"I can help with e-commerce support topics such as products, orders,
> returns…"*

— which hides an operational failure and invites the customer to retry into
the same wall forever.

Fixed in `src/agents/graph/nodes.py`: rate-limit/quota errors are detected at
both the router and the synthesis node and produce an honest message
(*"I'm temporarily over capacity… please try again in a few minutes"*), logged
at `ERROR`, and reported on the route as `unavailable` rather than `fallback`.

### 2.17 Shared-database safety *(triggered by a live incident)*

During this work the running Gradio app reported:

```
[✗] Error saving session to database: relation "customer_sessions" does not exist
```

`customer_sessions` had **39 rows** at the start of the session and **0** by
the time the error was investigated, and its unique constraint had moved from
`pg_constraint` to being only an index — meaning the table was dropped and
recreated by something outside this process. The prior conversation history
was lost. This database is shared (`defaultdb` / `avnadmin`, "shared group
endpoints"), so any teammate's pipeline run or manual drop does exactly this.

Three fixes:

**(a) Self-healing writes** — `save_session_to_db` now catches
`psycopg2.errors.UndefinedTable`, recreates the table once, and retries.
Previously a missing table meant every turn printed `[✗]` forever with no
recovery and no UI signal. Verified by dropping the table mid-run (with a
backup/restore around the test):

```
WARNING  customer_sessions is missing (dropped by another process?) — recreating and retrying
[✓] customer_sessions table initialized successfully
save_session_to_db returned: True    row written: ('heal-test',)
```

`print` calls in this module became proper logging.

**(b) Destructive writes now require opting in twice.**
`upload_dataframe_to_postgresql_db` defaulted to `if_exists="replace"`, which
**drops and recreates** the target table. On a shared database that made every
call — a stray import, a notebook cell, a pipeline re-run — silently
destructive.

| | Before | After |
|---|---|---|
| Default `if_exists` | `"replace"` (drops table) | `"fail"` |
| Dropping a table | Happened by default | Needs `if_exists="replace"` **and** `confirm_destructive=True` |
| Invalid value (e.g. `"skip"`) | Passed through to pandas | `ValueError` |
| CLI `--behavior replace` | Dropped everything | Refuses without `--confirm-replace` |

`"append"` was deliberately *not* chosen as the default: re-running a pipeline
would silently duplicate every row, which is harder to notice than an error.

**(c) Two latent bugs in `upload_postgres.py`:**

- `--force` with the configured `"skip"` behavior passed the literal string
  `"skip"` to `pandas.to_sql`, which only accepts `fail`/`replace`/`append` —
  so every table raised. It now resolves to the safe `"fail"`.
- The guard `if not POSTGRESQL_AIVEN_PASSWORD` tested a variable that is an
  **empty string** in this project (real credentials live in
  `POSTGRESQL_CONNECTION_STRING`). It was therefore always true, so the upload
  pipeline silently refused to run for everyone while reporting a missing
  password. Now checks the connection string actually used.

> Note: (c) was accidentally acting as a safety net against (b). Fixing it is
> only safe *because* the explicit destruction guard now exists — don't revert
> one without the other.

### 2.18 Return agent asked for a key customers do not have

Reported from a live run: starting a return, the agent asked the customer to
supply an `order_item_id` even though it had already been given the order ID.

This was not a prompt problem. The Return Agent had **no tool that maps an
order to its items** — `check_return_eligibility` and `create_return_request`
both *require* `order_item_id`, and nothing could discover it. The agent had
no option but to ask for an internal surrogate key the customer has never seen
and cannot look up. The old prompt even codified it: *"If order_id or
order_item_id is missing, ask for it directly."*

It is also unnecessary in the majority of cases: **6,318 of 10,000 orders
(63%) contain exactly one item**, so there is nothing to disambiguate.

Fix:

- **New `list_order_items(order_id)` tool** — returns each item with its
  product title, quantity, price and status, plus a `returnable_count` and
  explicit guidance for the model.
- **Prompt rewritten** with a dedicated "Never ask for an order_item_id"
  section: resolve it via the tool, proceed automatically when there is one
  returnable item, and otherwise ask **by product name**.
- Tool ordering in the agent puts `list_order_items` first.

Verified end to end:

```
single-item order ORD-000299
  tools: list_order_items, check_return_eligibility, lookup_return_policy, ...
  -> named the product ("Zip Ties"), cited return_policy.md > Return Window,
     gave the real delivery date. Never asked for an ID.

multi-item order ORD-005158
  tools: list_order_items
  -> listed 3 products by name and asked which one.
     No order_item_id or OI- key leaked to the customer.
```

### 2.19 Demo data + a return bug it uncovered

Almost every synthetic order is years old and the return window is 30 days, so
*every* return correctly came back "outside the window" — the agent behaving
properly, but no way to demo a successful return. Only 5 delivered orders were
in-window, and each is consumed the moment you demo against it.

**`scripts/prepare_demo_data.py`** *(new)* shifts a handful of delivered orders
into the window, choosing a mix of single-item and multi-item orders so both
return flows can be shown. Because this writes to a shared database it is
dry-run by default, snapshots original values into `demo_date_backup` before
any update, supports `--revert`, and only ever touches three date columns.

**The bug it uncovered.** With a genuinely returnable order, the return flow
crashed:

```
ERROR [return] agent invocation failed: 'StructuredTool' object is not callable
```

`create_return_request` called `check_return_eligibility(...)` directly, but
`@tool` turns a function into a `StructuredTool`, which is not callable.
**`create_return_request` had therefore never worked** — no return could ever
be created. It was invisible because every order failed the window check
first, so execution never reached that line.

Fix: eligibility logic extracted into a plain `_check_return_eligibility`
helper that both the tool wrapper and `create_return_request` call. An AST scan
across `src/` confirmed no other tool calls another tool directly.

Verified end to end:

```
"I want to return my order ORD-000002, the item is faulty"
  tools: list_order_items, check_return_eligibility, create_return_request
  -> "I've started a return for the Addlink S20 256GB 3-Pack SSD from order
      ORD-000002... Your return request (ID RET-002001) is now pending."
  -> row confirmed in `returns` (refund_amount 44.81, status pending)
```

The test return was deleted and the item reset afterwards, so `ORD-000002`
is fresh for the demo.

### 2.20 Cross-customer conversation leak

Reported from the UI: switching the customer dropdown while the Session ID box
stayed at `gradio-session-1` meant two different customers shared one
conversation history.

This was not cosmetic. The graph checkpointer is keyed on `thread_id`, which
was the raw `session_id`. Database persistence was already keyed on
`(customer_id, session_id)`, so the two layers disagreed: the DB correctly
started a new row per customer while the **in-memory graph resumed the previous
customer's messages**. Customer B could see customer A's conversation, and
agents would reason over the wrong person's orders.

Fixed at two levels:

**(a) Server-side (the real fix).** `SupportOrchestrator._thread_id()`
namespaces the graph thread by customer — `"{customer_id}::{session_id}"` —
applied consistently in `handle`, `reset_session` and `_seed_history_if_new`.
Isolation now holds regardless of what a caller passes as `session_id`, so the
CLI and any future API get it for free.

**(b) UI.** The Session ID box now starts as `{customer}-{random}` and
auto-regenerates when the dropdown changes, clearing the visible transcript at
the same time. A "Start new session" button does the same on demand.

Verified end to end — same session ID, two customers:

```
A says "My secret codeword is PLATYPUS. Remember it."

customer A thread  AEFKF6R2GUSK2AWPSWRR4ZO36JVQ::shared-session-id  2 messages
customer B thread  AHCPZDDPHJE3G7M6ST5WGRPLXHOA::shared-session-id  0 messages
LEAK of A conversation into B?  False
```

Three regression tests added (61 passing).

---

## 3. How to run the demo

```bash
# one-time: repair schema drift (idempotent)
.venv/bin/python -m src.data.migrations

# make a few orders returnable so the happy-path return demo works
.venv/bin/python scripts/prepare_demo_data.py            # preview
.venv/bin/python scripts/prepare_demo_data.py --apply    # write
.venv/bin/python scripts/prepare_demo_data.py --revert   # undo

# verify the stack end to end against the live LLM + DB
.venv/bin/python scripts/verify_demo.py

# unit tests
.venv/bin/python -m pytest tests/

# CLI
.venv/bin/python main.py

# UI
.venv/bin/python app/gradio_app.py
```

**If the UI shows a DEGRADED MODE banner, stop and fix the cause first** — every
answer will be a canned string.

### Groq free-tier quota — why you kept hitting 429

Three independent limits, enforced **per model**. Verified from live
`x-ratelimit-*` headers on this account:

| Model | TPM | RPD | TPD |
|---|---|---|---|
| `openai/gpt-oss-120b` | 8,000 | 1,000 | 200,000 |
| `openai/gpt-oss-20b` | 8,000 | 1,000 | — |
| `llama-3.3-70b-versatile` | 12,000 | 1,000 | — |
| `llama-3.1-8b-instant` | 6,000 | 14,400 | — |

**Root cause: Groq reserves `prompt_tokens + max_tokens` against your bucket
for the duration of each request, not the tokens actually generated.**

Measured directly — same trivial prompt, varying `max_tokens`:

```
max_tokens=16     TPM bucket delta =    15     actual usage = 88
max_tokens=3000   TPM bucket delta = 3,032     actual usage = 104
```

A 104-token request consumed 3,032 tokens of headroom. Confirmed again in a
429 payload: prompt ~2,821 + `max_tokens` 4,096 → `"Requested 7278"`.

This made `max_tokens` a *pre-charge*, not a safety cap. At 4,096 with an
8,000 TPM ceiling, barely one request could be in flight — so consecutive
ReAct steps within a single turn collided and 429'd mid-answer.

**Fixes applied:**

| Change | Effect |
|---|---|
| `max_tokens` 4096 → 2048 (agents), 512 → 384 (router), 1536 → 1024 (synthesis) | −2,048 reserved tokens per agent call |
| Router + synthesis moved to `openai/gpt-oss-20b` | Uses a **separate quota bucket**; frees the 120b bucket for agents only |
| `search_product_reviews` only bound when `PINECONE_API_KEY` is set | −203 tokens/step, and stops the model calling a tool that cannot work |
| Schema block trimmed (`features`, `description`, `sub_categories` sample values dropped; remaining samples capped at 60 chars) | −503 tokens/step |

Measured result:

```
prompt per call    2,821 -> 2,115 tokens
quota per call     6,917 -> 4,163 tokens
turns/day (120b)      ~9 ->    ~16      (plus router/synthesis now free of this bucket)
```

**Further levers if you still run short:**

1. **Upgrade to Groq Dev tier.** The single highest-leverage change; the free
   tier is not sized for iterative development plus a demo.
2. **Move the agents to `llama-3.3-70b-versatile`** — 12,000 TPM vs 8,000, and
   as a non-reasoning model it emits no chain-of-thought tokens. Costs some
   tool-selection quality; worth A/B testing via `SUPPORT_LLM_MODEL`.
3. **Cut `search_products` default `limit` 10 → 5** and `snippet` 350 → 200
   chars. Tool results are re-sent on the next ReAct step, so this saves
   roughly 500 tokens per turn.
4. **Shrink `DEFAULT_HISTORY_WINDOW`** (`src/agents/base_agent.py`, currently 8)
   to 4. Halves resent conversation context on long chats.
5. **Don't run the test suite before demoing** — see below.

### Demo-day warning: token budget

Groq's free tier is **200,000 tokens/day** on `gpt-oss-120b`, and this account
hit that ceiling during verification. After the fixes above, one product turn
costs ~12k tokens of that bucket, giving roughly **16 turns/day**.

Before demoing:

1. Check remaining quota (`x-ratelimit-remaining-tokens` header), or upgrade to
   Groq's Dev tier.
2. Do **not** run the full test suite immediately beforehand.
3. If you see *"I'm temporarily over capacity"*, that is the quota, not a bug.

To inspect live quota at any time:

```bash
curl -sD- -o/dev/null https://api.groq.com/openai/v1/chat/completions \
  -H "Authorization: Bearer $GROQ_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
  | grep -i ratelimit
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Without it the system runs degraded |
| `SUPPORT_LLM_RPS` | `2.0` | Global request ceiling. Lower on 429s |
| `SUPPORT_AGENT_REASONING_EFFORT` | `medium` | Do not lower — see §2.2 |
| `SUPPORT_ROUTER_REASONING_EFFORT` | `low` | Routing is a simple classification |
| `SUPPORT_LLM_MAX_TOKENS` | `4096` | Agent completion budget |
| `SUPPORT_LOG_LEVEL` | `INFO` | `DEBUG` for full agent tracing |
| `SUPPORT_DETERMINISTIC_MODE` | `false` | Force LLM-free canned replies |
| `PINECONE_API_KEY` | unset | Optional. Local retrieval is used when absent |

---

## 4. Verification status — read this before demoing

### Fully verified

| Check | Evidence |
|---|---|
| Unit + integration suite | **58 passed** (was 5 failed) |
| Session persistence | 16/16 persistence tests pass; upsert now writes |
| Router: sticky escalation | Poisoned history + product question → `['product']` @ 0.99 (was `['escalation']` @ 0.98) |
| Router: product vs recommendation | 6/6 probe queries route correctly |
| Router: fan-out cap | `_normalise_routes` capped at 2, unit-tested |
| Policy retrieval | Correct document selected for all 4 topic probes |
| Product search tool | Real in-budget devices, no accessories, for laptop/headphone/monitor queries |
| **Live: policy grounding** | `['return']` @ 0.99 via `lookup_support_policy`; quoted 30/90-day windows correctly with source |
| **Live: product search** | `['product']` @ 0.99 via `search_products`; 5 real headphones, all under $250 |
| **Live: conversational memory** | Follow-up "which of those has the best rating?" resolved with **0 tool calls** |
| Latency | 51s → ~30s per product turn (measured) |
| Rate-limit handling | Honest "over capacity" message, route `unavailable` |

### NOT yet verified end-to-end — blocked by provider quota

The Groq free tier allows **200,000 tokens/day** and the verification run
exhausted it. These three scenarios returned HTTP 429, **not** a logic failure,
so they remain unconfirmed against the live model:

| Scenario | Status |
|---|---|
| Sticky escalation, full multi-turn through the orchestrator | **BLOCKED** — unit-tested at the router level and passing |
| "when does the return window close?" not closing the session | **BLOCKED** — regex fix unit-tested and passing |
| Multi-intent synthesis producing one merged reply | **BLOCKED** — graph fan-out and fact sharing unit-tested and passing |

**Re-run `python scripts/verify_demo.py` once the daily quota resets** (or on a
paid tier) to confirm these three. Each is covered by a passing unit test, but
a unit test with stub agents is not the same as a live end-to-end run.

### Before-and-after summary

| Scenario | Before | After |
|---|---|---|
| Product question after any escalation | Re-escalated forever | Routes to `product` |
| "when does the return window close?" | **Session terminated** | Answered from policy KB |
| "show me headphones under $250" | Demanded a customer ID | 5 real products, all in budget |
| "which of those has the best rating?" | No cross-turn memory | Resolved with **0 tool calls** |
| Compound return + reorder | 2 disjoint blocks, `Step 1 / Step 2` | One merged reply, order ID shared |
| Every conversation save | Silently failed | Persisted |
| Bad API key | Silent canned replies | Loud banner + logs |
| Provider 429 | Generic "I can help with…" reply | Honest "over capacity, retry shortly" |
| Product turn latency | 51s | ~30s |

---

## 5. Not done — recommended next

1. **`PostgresSaver` instead of `MemorySaver`.** Graph state is currently
   in-process; a restart loses live conversation state (stored turns are
   re-seeded, so this is a degradation, not data loss). Needs
   `langgraph-checkpoint-postgres`, which is not installed.
2. **Streaming to Gradio** via `graph.stream(..., stream_mode="messages")` —
   the biggest remaining *perceived* latency win.
3. **Index the review corpus in Pinecone**, or drop `search_product_reviews`.
   It is currently a permanently-unavailable tool.
4. **Trigram index** (`pg_trgm`) on `product_catalog.title` — `ILIKE '%…%'`
   still can't use a B-tree.
5. **Ground-truth evaluation set** (item 1 in `ITEMS TO-DO-capstone.txt`).
   `src/evaluation/` is scaffolded but empty; `scripts/verify_demo.py` is a
   smoke test, not a benchmark.
6. **`src/agents/deterministic_agent.py`** could return a clearer
   "system is misconfigured" message rather than pretending to be a support
   agent.

---

## 6. File-by-file index

**New**

| File | Purpose |
|---|---|
| `src/agents/graph/state.py` | `SupportState`, reducers, per-turn reset |
| `src/agents/graph/nodes.py` | guardrail, supervisor, agent, synthesis nodes |
| `src/agents/graph/builder.py` | graph construction and compilation |
| `src/agents/router/schemas.py` | `RouteDecision` structured output |
| `src/agents/shared_tools.py` | `lookup_support_policy` |
| `src/rag/policy_store.py` | section-aware local policy retrieval |
| `src/data/migrations.py` | idempotent schema repair |
| `scripts/verify_demo.py` | live end-to-end verification |

**Substantially rewritten**

| File | Change |
|---|---|
| `src/agents/router/agent.py` | structured output, history no longer keyword-scanned |
| `src/agents/orchestrator/agent.py` | thin adapter over the graph |
| `src/agents/base_agent.py` | `SpecialistAgent` (was a stub) |
| `src/agents/product_agent/tools.py` | `search_products` + 2 more tools |
| `src/agents/product_agent/prompts.py` | false premise removed |
| `src/agents/llm/llm_provider.py` | tokens, shared limiter, reasoning effort |
| `src/utils/logger.py` | implemented (was a stub) |
| `tests/test_agents.py` | rewritten against corrected behaviour |
| `app/gradio_app.py` | trace, banner, dotenv |

**Targeted fixes**

| File | Change |
|---|---|
| `src/data/session_persistence.py` | upsert conflict target |
| `src/agents/recommendation_agent/prompts.py` | stop demanding a customer ID |
| `src/agents/{5}/agent.py` | collapsed onto `SpecialistAgent` |
| `main.py` | `load_dotenv`, degraded warning, trace |

**Deleted** — 6 one-line shadow modules under `src/agents/`.
