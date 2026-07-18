# Build Plan - Agentic Customer Support

## Locked decisions
- Framework: LangGraph ReAct agents.
- LLM: Groq through src/agents/llm/llm_provider.py.
- Vector store: Pinecone.
- Embeddings: local sentence-transformers.
- Guardrails: Guardrails-AI + SQL/tool-level safety rules.
- Routing: router-centric orchestration, no direct agent-to-agent communication.

## Scope split
- Product Agent: product facts (specs, compare, warranty, reviews), no personalization.
- Recommendation Agent: personalized recommendations from profile + order history.

## Build phases
1. Foundation: shared helpers, dependencies, package structure.
2. Retrieval: embeddings + Pinecone ingestion/retrieval tools.
3. Agents: product/order/return/recommendation/escalation/fallback.
4. Router + orchestrator.
5. Guardrails wiring.
6. Local CLI pipeline.
7. Gradio UI.
8. Evaluation and hardening.

## Acceptance checks
- Order tool can place/track/cancel safely.
- Return tool can check eligibility and create return requests.
- Recommendation tool uses customer profile + purchase history.
- Router dispatches to one specialist agent per turn.
- main.py and app/gradio_app.py run locally.
