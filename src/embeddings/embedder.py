"""Sentence-transformers embedding wrapper."""

from __future__ import annotations

from typing import Sequence

from src.config.settings import EMBEDDING_MODEL_NAME


class SentenceTransformerEmbedder:
	"""Thin wrapper around sentence-transformers for consistent embeddings."""

	def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
		self.model_name = model_name
		self._model = None

	@property
	def model(self):
		"""Lazy-load and return underlying sentence-transformers model."""
		if self._model is None:
			from sentence_transformers import SentenceTransformer

			self._model = SentenceTransformer(self.model_name)
		return self._model

	def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
		"""Embed multiple documents into vectors."""
		if not texts:
			return []
		vectors = self.model.encode(list(texts), normalize_embeddings=True)
		return [v.tolist() for v in vectors]

	def embed_query(self, text: str) -> list[float]:
		"""Embed a single query string into a vector."""
		vectors = self.embed_documents([text])
		return vectors[0] if vectors else []

