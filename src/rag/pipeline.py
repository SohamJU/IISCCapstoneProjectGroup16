"""End-to-end RAG indexing pipeline for policies and product reviews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.data import KNOWLEDGE_BASE_DIR, REVIEWS_PROCESSED_PATH
from src.config.settings import EMBEDDING_MODEL_NAME
from src.embeddings.embedder import SentenceTransformerEmbedder
from src.embeddings.vector_store import PineconeVectorStore, VectorDocument
from src.rag.retriever import get_retriever


@dataclass
class ChunkedRecord:
	"""Chunk record before vectorization."""

	doc_id: str
	text: str
	metadata: dict[str, Any]


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> list[str]:
	"""Chunk text using simple character windows with overlap."""
	cleaned = " ".join(text.split())
	if not cleaned:
		return []
	if len(cleaned) <= chunk_size:
		return [cleaned]

	chunks: list[str] = []
	step = max(1, chunk_size - overlap)
	for start in range(0, len(cleaned), step):
		chunk = cleaned[start : start + chunk_size]
		if chunk:
			chunks.append(chunk)
		if start + chunk_size >= len(cleaned):
			break
	return chunks


def _read_policy_chunks() -> list[ChunkedRecord]:
	records: list[ChunkedRecord] = []
	if not KNOWLEDGE_BASE_DIR.exists():
		return records

	for policy_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
		text = policy_path.read_text(encoding="utf-8")
		chunks = chunk_text(text)
		for i, chunk in enumerate(chunks, start=1):
			records.append(
				ChunkedRecord(
					doc_id=f"policy-{policy_path.stem}-{i}",
					text=chunk,
					metadata={
						"source": policy_path.name,
						"doc_type": "policy",
					},
				)
			)
	return records


def _resolve_review_text(row: pd.Series) -> str:
	candidates = [
		"review_text",
		"text",
		"review",
		"review_body",
		"content",
		"summary",
	]
	for col in candidates:
		if col in row and pd.notna(row[col]):
			value = str(row[col]).strip()
			if value:
				return value
	return ""


def _read_review_chunks(limit: int = 5000) -> list[ChunkedRecord]:
	records: list[ChunkedRecord] = []
	path = Path(REVIEWS_PROCESSED_PATH)
	if not path.exists():
		return records

	df = pd.read_csv(path)
	if limit > 0:
		df = df.head(limit)

	for idx, row in df.iterrows():
		text = _resolve_review_text(row)
		if not text:
			continue

		product_id = str(row.get("product_id", "")).strip()
		source = str(row.get("source", "reviews.csv"))
		chunks = chunk_text(text, chunk_size=450, overlap=80)

		for i, chunk in enumerate(chunks, start=1):
			records.append(
				ChunkedRecord(
					doc_id=f"review-{idx}-{i}",
					text=chunk,
					metadata={
						"source": source,
						"doc_type": "review",
						"product_id": product_id,
					},
				)
			)
	return records


def build_vector_indexes(review_limit: int = 5000) -> dict[str, int]:
	"""Build/refresh Pinecone namespaces for policies and reviews."""
	retriever = get_retriever()
	embedder = SentenceTransformerEmbedder(EMBEDDING_MODEL_NAME)
	store: PineconeVectorStore = retriever.vector_store

	policy_records = _read_policy_chunks()
	review_records = _read_review_chunks(limit=review_limit)

	policy_count = 0
	if policy_records:
		policy_vectors = embedder.embed_documents([r.text for r in policy_records])
		policy_docs = [VectorDocument(r.doc_id, r.text, r.metadata) for r in policy_records]
		policy_count = store.upsert(
			namespace="policies",
			documents=policy_docs,
			vectors=policy_vectors,
		)

	review_count = 0
	if review_records:
		review_vectors = embedder.embed_documents([r.text for r in review_records])
		review_docs = [VectorDocument(r.doc_id, r.text, r.metadata) for r in review_records]
		review_count = store.upsert(
			namespace="reviews",
			documents=review_docs,
			vectors=review_vectors,
		)

	return {
		"policies_indexed": policy_count,
		"reviews_indexed": review_count,
	}

