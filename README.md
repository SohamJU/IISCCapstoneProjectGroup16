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

## Memory

The repository also includes a lightweight runtime memory layer for live chat sessions.

See [MEMORY.md](MEMORY.md) for usage examples, expected output, and how `SessionManager` and `ConversationMemory` fit into downstream chat flows.

## Status
Project scaffold created. Implementation pending.
