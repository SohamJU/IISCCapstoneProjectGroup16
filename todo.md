# Project TODO

## Data available
- [x] Amazon product & reviews data (Structured)
- [x] Twitter data (Unstructured data)

## Data created
- [x] Policy documents
- [x] Synthetic data

## Environment dependencies
- [x] CI is already testing compatibility
- [x] Python 3.12
- [x] Pyproject.toml

## Data hosting
- [x] Structured data: PostgreSQL
- [ ] Unstructured data: Pinecone

## EDA
- Check the Project Doc

## Agents
- Product agent - WIP
- Order agent
- Recommendation agent
- Router agent
- Escalation agent
- Return / cancellation / refund agent (together or separate)

## Guardrails
- User query should not change or delete records
- Change system prompt if the question is out of context; provide a standard message
- Input
- Output
- Intermediate
- Prompt injection

## Evaluation
- Edge cases should be part of evaluation
- Basic evaluation dataset creation
- Types of evaluation:
  - LLM as a Judge
  - Deterministic
  - Human in the loop
  - Constitutional AI

## Production / Deploy
- Docker
- CI/CD
  - Run evaluation metrics as part of CI for deterministic metrics
  - For LLM-as-a-judge metrics, run locally

## UI
- Streamlit / Gradio

## Edge cases
- TBD

## Documentation
- Include progress todos in the running progress
- Place everything in the Doc folder
- How to run the pipeline? -> Developer
  - Architectural overview / flow of information
- Details about the complete project
  - Architectural overview / flow of information
- User documentation

## Memory management
- Do we want to manage long-term conversations? - MongoDB?
- Chat history?
- Agent-level session memory
