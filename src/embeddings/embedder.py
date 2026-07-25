"""Embedding model wrapper using BAAI/bge-small-en-v1.5.

Provides a lightweight CPU-friendly embedding model that runs fully locally
(no API key required). The model is downloaded from HuggingFace on first use
and cached in the default ``~/.cache/huggingface`` directory.

Model: ``BAAI/bge-small-en-v1.5``
- Parameters: ~33M
- Embedding dimensions: 384
- Metric: Cosine similarity
- CPU performance: ~2–4 sec / 256-doc batch on a standard laptop

**Important BGE convention:**
- *Documents* (to be indexed): passed as-is.
- *Queries* (at search time): must be prefixed with
  ``"Represent this sentence for searching relevant passages: "``
  This is handled automatically by ``embed_query()``.

Usage::

    from src.embeddings.embedder import ProductEmbedder

    embedder = ProductEmbedder()

    # Embed a search query
    q_vec = embedder.embed_query("quiet washing machine under $800")

    # Embed a batch of documents for indexing
    vecs = embedder.embed_texts(["Product title 1", "Product title 2"])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_DEFAULT_BATCH_SIZE = 256


class ProductEmbedder:
    """Thin wrapper around ``sentence-transformers`` for product embeddings.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Defaults to ``BAAI/bge-small-en-v1.5``.
    batch_size : int
        Number of texts embedded per forward pass.
    show_progress : bool
        Show a ``tqdm`` progress bar during batch embedding.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
        use_cpu: bool = False,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._show_progress = show_progress
        self._use_cpu = use_cpu
        self._model: SentenceTransformer | None = None  # lazy load

    # ── Public API ─────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query (applies BGE query prefix)."""
        prefixed = f"{_QUERY_PREFIX}{query}"
        vec = self._model_instance.encode(
            prefixed,
            normalize_embeddings=True,
        )
        return vec.tolist()

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool | None = None,
    ) -> np.ndarray:
        """Embed a list of document texts (no prefix — for indexing)."""
        progress = show_progress if show_progress is not None else self._show_progress
        vecs: np.ndarray = self._model_instance.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=progress,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)
        
    def embed_texts_multiprocess(self, texts: list[str]) -> np.ndarray:
        """Embed a large list of texts using two CPU processes."""
        model = self._model_instance
        target_devices = ["cpu", "cpu"] if self._use_cpu else None
        logger.info("Starting multi-process pool for %d texts with %d workers...", len(texts), len(target_devices or [0]))
        pool = model.start_multi_process_pool(target_devices=target_devices)
        try:
            # Use a smaller chunk size so worker processes receive work
            # in reasonably sized packages and make observable progress.
            vecs = model.encode_multi_process(
                texts,
                pool,
                batch_size=self._batch_size,
                chunk_size=500,
                normalize_embeddings=True,
            )
            return np.array(vecs, dtype=np.float32)
        finally:
            model.stop_multi_process_pool(pool)
            logger.info("Multi-process pool stopped.")

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (384 for bge-small-en-v1.5)."""
        return self._model_instance.get_sentence_embedding_dimension()  # type: ignore[return-value]

    # ── Internal ───────────────────────────────────────────────────────────

    @property
    def _model_instance(self) -> "SentenceTransformer":
        """Lazy-load the SentenceTransformer model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            
            device = "cpu" if self._use_cpu else None
            self._model = SentenceTransformer(self._model_name, device=device)
            
            logger.info(
                "Embedding model '%s' loaded on %s. Dimension: %d",
                self._model_name,
                self._model.device,
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model


# ── Module-level singleton ────────────────────────────────────────────────────
_EMBEDDER: ProductEmbedder | None = None


def get_embedder() -> ProductEmbedder:
    """Return (and cache) the module-level embedder singleton."""
    global _EMBEDDER  # noqa: PLW0603
    if _EMBEDDER is None:
        _EMBEDDER = ProductEmbedder(show_progress=False)
    return _EMBEDDER
