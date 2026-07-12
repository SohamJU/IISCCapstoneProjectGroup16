# Customer Support Agent

## Overview
Multi-agent customer support system with:
- Router Agent
- Product Agent
- Order Agent
- Return Agent
- Recommendation Agent
- Retrieval-Augmented Generation (RAG)
- Conversation Memory
- Tool Calling
- Evaluation Framework
- Optional Gradio UI

## High-Level Architecture

User
-> Router Agent
-> Specialized Agent
-> RAG Retriever
-> Vector Store
-> LLM
-> Response

## Repository Structure

This repository is organized into:
- src/agents: agent implementations
- src/rag: retrieval and generation pipeline
- src/embeddings: embeddings and vector stores
- src/memory: conversation memory
- src/tools: external tools and integrations
- src/evaluation: benchmarking and evaluation
- app: optional user interface
- notebooks: experimentation and demos

## Preprocessing

The repository now includes a preprocessing workflow for:
- structured Amazon ratings data
- Twitter customer support conversations
- product catalog text documents

See [PREPROCESSING.md](PREPROCESSING.md) for setup, outputs, and run commands.

## Synthetic Data Pipeline

To build a robust, perfectly-relational database for agent testing, the project includes a massive 13-step synthetic data generation pipeline (`pipelines/02_run_synthetic_data_pipeline.py`).

This pipeline handles:
1. **Downloading** real Amazon 2023 product and review data.
2. **Generating** thousands of highly realistic synthetic customers, relational orders, and product returns using Faker and logic constraints.
3. **Validating** data integrity (foreign keys, logic checks).
4. **Publishing** all tables directly to our shared Aiven Cloud PostgreSQL database.

For a full breakdown of the architecture, CLI commands, and environment variables, see [docs/01_data_pipeline.md](docs/01_data_pipeline.md).

## Memory

The repository also includes a lightweight runtime memory layer for live chat sessions.

See [MEMORY.md](MEMORY.md) for usage examples, expected output, and how `SessionManager` and `ConversationMemory` fit into downstream chat flows.

## Status
Project scaffold created. Implementation pending.
