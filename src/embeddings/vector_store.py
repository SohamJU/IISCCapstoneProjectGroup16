"""Pinecone vector store + local embedding cache for product embeddings.

Provides two classes:

``PineconeVectorStore``
    Wraps the Pinecone SDK to upsert and query 384-dim product embeddings.
    Requires ``PINECONE_API_KEY`` and ``PINECONE_INDEX_NAME`` in ``.env``.

``LocalEmbeddingCache``
    Reads and writes a local Parquet file (``embeddings_cache.parquet``)
    so the indexing pipeline can resume without re-embedding already-done rows.

Usage::

    # Pinecone query
    from src.embeddings.vector_store import PineconeVectorStore
    store = PineconeVectorStore()
    results = store.query(embedding=[...], top_k=10)

    # Local cache
    from src.embeddings.vector_store import LocalEmbeddingCache
    cache = LocalEmbeddingCache()
    cached_ids = cache.get_cached_ids()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_PATH = _PROJECT_ROOT / "data" / "processed" / "embeddings_cache.parquet"

_PINECONE_BATCH_SIZE = 100  # vectors per upsert call


# ═══════════════════════════════════════════════════════════════════════════
# Pinecone Vector Store
# ═══════════════════════════════════════════════════════════════════════════

class PineconeVectorStore:
    """Pinecone-backed vector store for product embeddings.

    Parameters
    ----------
    api_key : str | None
        Pinecone API key. If ``None``, reads from ``PINECONE_API_KEY`` env var.
    index_name : str | None
        Pinecone index name. If ``None``, reads from ``PINECONE_INDEX_NAME``
        env var (default: ``"product-catalog"``).

    Notes
    -----
    The Pinecone index must be created beforehand with:
    - Dimensions: 384
    - Metric: cosine
    Create it at https://app.pinecone.io/.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("PINECONE_API_KEY", "")
        self._index_name = index_name or os.getenv(
            "PINECONE_INDEX_NAME", "product-catalog"
        )
        self._index = None  # lazy init

    # ── Public API ─────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if a valid API key is configured."""
        return bool(self._api_key) and self._api_key != "your-pinecone-api-key-here"

    def upsert(
        self,
        product_ids: list[str],
        embeddings: np.ndarray,
        metadata_list: list[dict[str, Any]],
    ) -> int:
        """Upsert product vectors into the Pinecone index.

        Parameters
        ----------
        product_ids : list[str]
            Unique identifiers (ASIN) for each product.
        embeddings : np.ndarray
            Shape ``(N, 384)`` float32 embedding matrix.
        metadata_list : list[dict]
            Per-product metadata dicts stored alongside the vector.
            Include ``title``, ``price``, ``average_rating``, ``main_category``.

        Returns
        -------
        int
            Number of vectors successfully upserted.
        """
        if not self.is_available():
            logger.warning("Pinecone API key not set — skipping upsert.")
            return 0

        index = self._get_index()
        total = 0

        for start in range(0, len(product_ids), _PINECONE_BATCH_SIZE):
            end = start + _PINECONE_BATCH_SIZE
            batch_ids = product_ids[start:end]
            batch_vecs = embeddings[start:end].tolist()
            batch_meta = metadata_list[start:end]

            vectors = [
                {"id": pid, "values": vec, "metadata": meta}
                for pid, vec, meta in zip(batch_ids, batch_vecs, batch_meta)
            ]
            index.upsert(vectors=vectors)
            total += len(vectors)
            logger.debug("Upserted batch %d–%d to Pinecone.", start, end)

        logger.info("Pinecone upsert complete: %d vectors.", total)
        return total

    def query(
        self,
        embedding: list[float],
        top_k: int = 15,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the Pinecone index for nearest neighbours.

        Parameters
        ----------
        embedding : list[float]
            384-dim query vector (unit-normalised).
        top_k : int
            Number of neighbours to retrieve.
        filter_dict : dict | None
            Pinecone metadata filter (e.g., ``{"main_category": "Electronics"}``).

        Returns
        -------
        list[dict]
            List of matches, each with keys: ``product_id``, ``score``,
            and all metadata fields stored at upsert time.
        """
        if not self.is_available():
            logger.warning("Pinecone not available — returning empty results.")
            return []

        try:
            index = self._get_index()
            kwargs: dict[str, Any] = {
                "vector": embedding,
                "top_k": top_k,
                "include_metadata": True,
            }
            if filter_dict:
                kwargs["filter"] = filter_dict

            response = index.query(**kwargs)
            results = []
            for match in response.get("matches", []):
                row = {"product_id": match["id"], "_vector_score": match["score"]}
                row.update(match.get("metadata", {}))
                results.append(row)
            return results
        except Exception as exc:
            logger.error("Pinecone query failed: %s", exc)
            return []

    def index_stats(self) -> dict[str, Any]:
        """Return Pinecone index statistics (total vector count, etc.)."""
        if not self.is_available():
            return {"status": "unavailable — API key not set"}
        try:
            return self._get_index().describe_index_stats()
        except Exception as exc:
            return {"error": str(exc)}

    # ── Internal ───────────────────────────────────────────────────────────

    def _get_index(self):
        """Lazy-init the Pinecone index connection."""
        if self._index is None:
            from pinecone import Pinecone  # noqa: PLC0415

            pc = Pinecone(api_key=self._api_key)
            self._index = pc.Index(self._index_name)
            logger.info("Connected to Pinecone index '%s'.", self._index_name)
        return self._index


# ═══════════════════════════════════════════════════════════════════════════
# Local Embedding Cache (Parquet)
# ═══════════════════════════════════════════════════════════════════════════

class LocalEmbeddingCache:
    """Parquet-backed local cache for product embeddings.

    Stores ``(product_id, embedding_bytes)`` rows so the offline indexing
    pipeline can resume without re-embedding already processed products.

    The embedding vector is serialised as raw bytes (``np.ndarray.tobytes()``)
    and deserialised with ``np.frombuffer(..., dtype=np.float32)``.

    Parameters
    ----------
    path : Path | None
        Path to the Parquet file. Defaults to
        ``data/processed/embeddings_cache.parquet``.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _CACHE_PATH

    def get_cached_ids(self) -> set[str]:
        """Return the set of product_ids already in the cache."""
        if not self._path.exists():
            return set()
        df = pd.read_parquet(self._path, columns=["product_id"])
        return set(df["product_id"].tolist())

    def load(self) -> tuple[list[str], np.ndarray]:
        """Load all cached (product_id, embedding) pairs.

        Returns
        -------
        tuple[list[str], np.ndarray]
            ``(product_ids, embeddings)`` where embeddings has shape (N, 384).
        """
        if not self._path.exists():
            return [], np.empty((0, 384), dtype=np.float32)

        df = pd.read_parquet(self._path)
        product_ids = df["product_id"].tolist()
        embeddings = np.stack(
            [np.frombuffer(b, dtype=np.float32) for b in df["embedding_bytes"]]
        )
        return product_ids, embeddings

    def append(self, product_ids: list[str], embeddings: np.ndarray) -> None:
        """Append new rows to the cache file.

        Parameters
        ----------
        product_ids : list[str]
            Product IDs to cache.
        embeddings : np.ndarray
            Shape ``(N, 384)`` float32 embedding matrix.
        """
        new_rows = {
            "product_id": product_ids,
            "embedding_bytes": [row.astype(np.float32).tobytes() for row in embeddings],
        }
        new_df = pd.DataFrame(new_rows)

        if self._path.exists():
            existing = pd.read_parquet(self._path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            combined = new_df

        combined.to_parquet(self._path, index=False)
        logger.debug("Cache updated: %d total rows.", len(combined))

    @property
    def path(self) -> Path:
        return self._path
