"""Retriever implementation for Pinecone-backed RAG."""

from __future__ import annotations

from typing import Any

from src.config.settings import (
	DEFAULT_RETRIEVAL_TOP_K,
	EMBEDDING_DIMENSION,
	EMBEDDING_MODEL_NAME,
	MAX_RETRIEVAL_TOP_K,
	PINECONE_API_KEY,
	PINECONE_CLOUD,
	PINECONE_INDEX_NAME,
	PINECONE_REGION,
)
from src.embeddings.embedder import SentenceTransformerEmbedder
from src.embeddings.vector_store import PineconeVectorStore


class SupportRetriever:
	"""Retrieves policy and review context from Pinecone namespaces."""

	def __init__(self) -> None:
		self.embedder = SentenceTransformerEmbedder(EMBEDDING_MODEL_NAME)
		self.vector_store = PineconeVectorStore(
			api_key=PINECONE_API_KEY,
			index_name=PINECONE_INDEX_NAME,
			cloud=PINECONE_CLOUD,
			region=PINECONE_REGION,
			dimension=EMBEDDING_DIMENSION,
		)

	@staticmethod
	def _safe_top_k(top_k: int) -> int:
		return max(1, min(top_k, MAX_RETRIEVAL_TOP_K))

	def search_policies(self, query: str, top_k: int = DEFAULT_RETRIEVAL_TOP_K) -> list[dict[str, Any]]:
		"""Search policy documents in Pinecone namespace `policies`."""
		vector = self.embedder.embed_query(query)
		return self.vector_store.query(
			namespace="policies",
			query_vector=vector,
			top_k=self._safe_top_k(top_k),
		)

	def search_reviews(
		self,
		query: str,
		*,
		product_id: str | None = None,
		top_k: int = DEFAULT_RETRIEVAL_TOP_K,
	) -> list[dict[str, Any]]:
		"""Search review snippets in Pinecone namespace `reviews`."""
		vector = self.embedder.embed_query(query)
		metadata_filter = {"product_id": product_id} if product_id else None
		return self.vector_store.query(
			namespace="reviews",
			query_vector=vector,
			top_k=self._safe_top_k(top_k),
			metadata_filter=metadata_filter,
		)


_RETRIEVER: SupportRetriever | None = None


def get_retriever() -> SupportRetriever:
	"""Return process-wide retriever singleton."""
	global _RETRIEVER
	if _RETRIEVER is None:
		_RETRIEVER = SupportRetriever()
	return _RETRIEVER


def format_matches(matches: list[dict[str, Any]]) -> str:
	"""Format retrieval matches into concise JSON-like text for agent prompts."""
	if not matches:
		return "No relevant matches found."

	lines: list[str] = []
	for i, match in enumerate(matches, start=1):
		metadata = match.get("metadata", {})
		text = str(metadata.get("text", "")).strip().replace("\n", " ")
		if len(text) > 500:
			text = text[:497] + "..."
		source = metadata.get("source", "unknown")
		score = match.get("score", 0.0)
		lines.append(f"{i}. score={score:.4f} source={source} text={text}")
	return "\n".join(lines)

