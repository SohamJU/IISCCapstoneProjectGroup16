"""Centralized runtime configuration for agents and retrieval."""

from __future__ import annotations

import os


# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "customer-support-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Embeddings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# Retrieval defaults
DEFAULT_RETRIEVAL_TOP_K = int(os.getenv("DEFAULT_RETRIEVAL_TOP_K", "5"))
MAX_RETRIEVAL_TOP_K = int(os.getenv("MAX_RETRIEVAL_TOP_K", "10"))

# Debugging
DEBUG = os.getenv("SUPPORT_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}

