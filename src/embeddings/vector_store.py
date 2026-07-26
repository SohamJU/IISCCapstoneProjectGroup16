"""Pinecone vector database operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone, ServerlessSpec


@dataclass
class VectorDocument:
	"""Document payload for vector upsert."""

	doc_id: str
	text: str
	metadata: dict[str, Any]


class PineconeVectorStore:
	"""Pinecone index wrapper with convenience upsert/query helpers."""

	def __init__(
		self,
		*,
		api_key: str,
		index_name: str,
		cloud: str,
		region: str,
		dimension: int,
	) -> None:
		if not api_key:
			raise ValueError("PINECONE_API_KEY is not set.")

		self.index_name = index_name
		self.dimension = dimension
		self._client = Pinecone(api_key=api_key)
		self._ensure_index(cloud=cloud, region=region)
		self._index = self._client.Index(index_name)

	def _ensure_index(self, *, cloud: str, region: str) -> None:
		"""Create index if it does not exist."""
		existing = {idx["name"] for idx in self._client.list_indexes()}
		if self.index_name in existing:
			return

		self._client.create_index(
			name=self.index_name,
			dimension=self.dimension,
			metric="cosine",
			spec=ServerlessSpec(cloud=cloud, region=region),
		)

	def upsert(
		self,
		*,
		namespace: str,
		documents: list[VectorDocument],
		vectors: list[list[float]],
	) -> int:
		"""Upsert vectorized documents into namespace and return count."""
		if not documents:
			return 0

		payload = []
		for doc, vector in zip(documents, vectors, strict=False):
			metadata = {**doc.metadata, "text": doc.text}
			payload.append({"id": doc.doc_id, "values": vector, "metadata": metadata})

		self._index.upsert(vectors=payload, namespace=namespace)
		return len(payload)

	def query(
		self,
		*,
		namespace: str,
		query_vector: list[float],
		top_k: int,
		metadata_filter: dict[str, Any] | None = None,
	) -> list[dict[str, Any]]:
		"""Query nearest neighbors and return simplified match payloads."""
		result = self._index.query(
			vector=query_vector,
			top_k=top_k,
			namespace=namespace,
			include_metadata=True,
			filter=metadata_filter,
		)

		matches: list[dict[str, Any]] = []
		for match in result.get("matches", []):
			matches.append(
				{
					"id": match.get("id"),
					"score": match.get("score"),
					"metadata": match.get("metadata", {}),
				}
			)
		return matches

