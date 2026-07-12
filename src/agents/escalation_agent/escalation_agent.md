# Escalation Agent

## Purpose
The Escalation Agent is a specialized support layer that monitors customer conversations for high-risk or out-of-scope issues, then prepares a concise handoff to a human agent.

## Core Responsibilities
- **Sentiment and complexity monitoring**: detect extreme frustration, profanity, repeated requests, and issues that exceed the primary AI agents' capabilities.
- **Contextual handoff summarization**: create a short, clear ticket summary for the human agent.
- **Pipeline routing**: classify escalated issues into the proper department: Billing, Technical Support, or Management.
- **Maintaining continuity**: preserve session context so customers do not need to repeat their issue when a human takes over.
- **RAG-aware support**: optionally incorporate relevant policy or knowledge-base evidence when deciding whether escalation is needed.

## Implementation
The escalation agent is implemented in `src/agents/escalation_agent` and includes:

- `agent.py` — `EscalationAgent` class that creates a LangGraph ReAct agent with conversation memory.
- `config.py` — runtime and RAG-related constants.
- `prompts.py` — system prompt templates and handoff summary renderers.
- `tools.py` — escalation-specific tools such as knowledge retrieval and department classification.

## Behavior and flow
1. A new message arrives from the customer.
2. The Escalation Agent evaluates whether the request is in-scope for AI support.
3. If escalation is required, it generates:
   - original customer intent
   - prior actions taken by AI agents
   - customer pain points and tone
   - relevant retrieval evidence or policy guidance
   - destination department and reason for escalation
4. If escalation is not required, it returns an AI resolution or clarification prompt.

## Handoff Summary Requirements
When a handoff is created, the summary should include:
- **Original customer intent**
- **Steps already taken**
- **Identified pain points**
- **Relevant RAG or policy evidence**
- **Recommended escalation destination**
- **Reason for escalation**

## Escalation Routing Logic
- **Billing**: refunds, charges, billing disputes, payment issues.
- **Technical Support**: product malfunctions, login/connectivity/error issues, inaccessible services.
- **Management**: complaints, policy exceptions, requests for a manager, severe dissatisfaction.
- **General Support**: all other unresolved questions that still require human review.

## Integration Notes
- Use `EscalationAgent.chat(user_message)` to evaluate each incoming message.
- Call `EscalationAgent.reset_memory()` when the session ends or when a clean slate is needed.
- The `retrieve_knowledge` tool is currently a placeholder and should be connected to the repository's RAG retriever if available.

## Example Usage
```python
from src.agents.escalation_agent import EscalationAgent

agent = EscalationAgent(session_id="session_123")
response = agent.chat(
    "I want a refund and I have been waiting for 3 weeks. This is unacceptable."
)
print(response)
```

## Flowchart
```mermaid
flowchart TD
    A[Customer message arrives] --> B{Should escalate?}
    B -- No --> C[AI resolves or asks clarification]
    B -- Yes --> D[Analyze sentiment, frustration, out-of-scope criteria]
    D --> E[Retrieve relevant policy/knowledge evidence]
    E --> F[Classify escalation destination]
    F --> G[Generate concise handoff summary]
    G --> H[Route ticket to human agent dashboard]
    H --> I[Continue session context for human handoff]
```

## Future Work
- Connect `retrieve_knowledge` to an actual RAG retriever.
- Add explicit escalation scoring and threshold handling.
- Extend departmental routing rules with more fine-grained categories.
- Add automated tests that verify escalation summaries and routing logic.
