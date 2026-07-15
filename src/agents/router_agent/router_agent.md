# Router agent

The router agent is a lightweight intent classifier that decides which specialist
agent should handle a user request.

## What it does

- Routes product-related questions to the product agent.
- Routes order and shipment questions to the order agent.
- Routes return, refund, and replacement questions to the return agent.
- Supports future expansion by registering additional agents through the
  public registration API.

## How to use it

```python
from src.agents.router_agent import RouterAgent

router = RouterAgent()
decision = router.route("Where is my order?")
print(decision.target_agent)
print(decision.confidence)
```

## Routing strategy

The router first uses deterministic keyword and phrase scoring for fast,
interpretable routing. If an LLM is supplied and enabled, it can optionally use
that as a fallback classifier.

## Extending it

You can add new routes later with:

```python
router.register_agent(
    name="escalation",
    description="Escalate complex issues to a human support specialist.",
    keywords=(("human", 0.5), ("escalate", 0.6)),
)
```
