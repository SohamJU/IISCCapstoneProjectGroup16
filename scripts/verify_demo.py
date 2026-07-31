"""End-to-end demo verification against the live stack.

Exercises the specific failure modes that made the previous build give bad
answers, so a regression is visible before a demo rather than during one.

Run::

    python scripts/verify_demo.py
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv()

from src.agents.orchestrator import SupportOrchestrator

CUSTOMER_ID = "AEFKF6R2GUSK2AWPSWRR4ZO36JVQ"  # Robert, has ORD-000002/3
ORDER_ID = "ORD-000002"


def is_rate_limited(result) -> bool:
    """True when the provider quota, not the code, produced this outcome."""
    blob = " ".join([result.response, *result.warnings]).lower()
    return "over capacity" in blob or "rate limit" in blob or "429" in blob


def show(label: str, result, expect_routes: list[str] | None = None) -> str:
    """Print one scenario's outcome. Returns "PASS", "FAIL" or "BLOCKED"."""
    if is_rate_limited(result):
        # Never report a quota failure as a pass. The earlier version of this
        # script defaulted ok=True whenever expect_routes was None, so two
        # scenarios reported PASS while actually returning a 429 fallback.
        status = "BLOCKED"
        ok = False
    elif expect_routes is not None:
        ok = result.routes == expect_routes
        status = "PASS" if ok else "FAIL"
    else:
        # No route expectation: still require a real specialist answer rather
        # than the generic out-of-scope fallback.
        ok = bool(result.response.strip()) and result.routes != ["fallback"]
        status = "PASS" if ok else "FAIL"

    print(f"\n{'=' * 78}")
    print(f"[{status}] {label}")
    print(f"  routes={result.routes} confidence={result.confidence:.2f}")
    if expect_routes is not None and not ok:
        print(f"  EXPECTED routes={expect_routes}")
    if result.tools_used:
        print(f"  tools={list(dict.fromkeys(result.tools_used))}")
    if result.warnings:
        print(f"  warnings={result.warnings}")
    print(f"{'-' * 78}")
    print(result.response[:1400])
    return status


def main() -> int:
    orchestrator = SupportOrchestrator()

    if orchestrator.is_degraded:
        print(f"ABORT: system is degraded — {orchestrator.degraded_reason}")
        return 1

    results: list[tuple[str, str]] = []
    started = time.time()

    # 1. Policy question — must be grounded in the knowledge base.
    results.append(
        (
            "policy grounding",
            show(
                "Return policy question (expect return route, policy tool)",
                orchestrator.handle(
                    "How many days do I have to return something?",
                    session_id="verify-policy",
                ),
                expect_routes=["return"],
            ),
        )
    )

    # 2. Product search — must return real, in-budget products.
    results.append(
        (
            "product search",
            show(
                "Product search with budget (expect product route)",
                orchestrator.handle(
                    "Show me wireless headphones under $250",
                    session_id="verify-product",
                ),
                expect_routes=["product"],
            ),
        )
    )

    # 3. Sticky escalation regression — the single worst bug in the old build.
    print(f"\n{'#' * 78}\n# STICKY ESCALATION REGRESSION\n{'#' * 78}")
    orchestrator.handle(
        "This is unacceptable, I want to speak to a human!",
        session_id="verify-sticky",
        customer_id=CUSTOMER_ID,
    )
    results.append(
        (
            "sticky escalation",
            show(
                "Normal product question AFTER an escalation (must NOT re-escalate)",
                orchestrator.handle(
                    "Anyway, what 4K monitors do you sell?",
                    session_id="verify-sticky",
                    customer_id=CUSTOMER_ID,
                ),
                expect_routes=["product"],
            ),
        )
    )

    # 4. Close-chat false positive.
    close_result = orchestrator.handle(
        "when does the return window close?", session_id="verify-close"
    )
    results.append(
        (
            "close-chat false positive",
            show(
                "'when does the return window close?' must NOT end the session",
                close_result,
                expect_routes=["return"],
            ),
        )
    )

    # 5. Multi-intent — two agents, one synthesised reply.
    multi = orchestrator.handle(
        f"I want to return order {ORDER_ID} and also find a cheaper replacement",
        session_id="verify-multi",
        customer_id=CUSTOMER_ID,
    )
    multi_status = show(
        "Compound request (expect 2 routes, ONE merged reply with no 'Step N')",
        multi,
    )
    if multi_status == "PASS" and not (
        len(multi.routes) >= 2 and "Step 1" not in multi.response
    ):
        multi_status = "FAIL"
        print("  -> expected >=2 routes and no 'Step N' scaffolding")
    results.append(("multi-intent synthesis", multi_status))

    # 6. Conversational memory across turns.
    orchestrator.handle(
        "Show me wireless headphones under $250", session_id="verify-memory"
    )
    results.append(
        (
            "conversational memory",
            show(
                "Follow-up 'which of those is cheapest?' (must resolve the reference)",
                orchestrator.handle(
                    "which of those is cheapest?", session_id="verify-memory"
                ),
            ),
        )
    )

    elapsed = time.time() - started
    print(f"\n{'=' * 78}\nSUMMARY ({elapsed:.0f}s total)")

    passed = sum(1 for _, status in results if status == "PASS")
    blocked = sum(1 for _, status in results if status == "BLOCKED")

    for name, status in results:
        print(f"  [{status}] {name}")

    print(f"\n{passed}/{len(results)} scenarios passed")

    if blocked:
        print(
            f"\n!! {blocked} scenario(s) BLOCKED by the provider rate limit — "
            "these were NOT verified.\n"
            "!! Groq's free tier is 200,000 tokens/day. Wait for the daily "
            "reset or upgrade, then re-run."
        )
        return 2

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
