"""BM25 keyword index over the product catalog.

Builds an in-memory BM25 index (using ``rank_bm25``) over the concatenated
text of ``title``, ``description``, and ``features`` columns.  The index is
loaded lazily on first use from PostgreSQL (or falls back to the processed CSV
if the DB is unavailable).

Usage::

    from src.search.bm25_index import BM25ProductIndex

    index = BM25ProductIndex()
    results = index.search("gaming headset with noise cancellation", top_k=10)
    for r in results:
        print(r["title"], r["price"], r["average_rating"])
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CATALOG_CSV = _PROJECT_ROOT / "data" / "processed" / "product_catalog.csv"

# ── Text helpers ──────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _safe_list_str(raw: Any) -> str:
    """Convert a Python-list-as-string or JSON array into a flat string."""
    if not raw or (isinstance(raw, float)):
        return ""
    s = str(raw).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return " ".join(str(x) for x in parsed)
    except Exception:
        pass
    # Strip surrounding brackets/quotes as a fallback
    return re.sub(r"[\[\]'\"]", " ", s)


def _build_text(row: pd.Series) -> str:
    """Concatenate the searchable text fields for one product row."""
    title = str(row.get("title", "") or "")
    description = _safe_list_str(row.get("description", ""))
    features = _safe_list_str(row.get("features", ""))
    category = str(row.get("main_category", "") or "")
    store = str(row.get("store", "") or "")
    return f"{title} {category} {store} {description} {features}"


def _tokenize(text: str) -> list[str]:
    """Lower-case word tokenisation suitable for BM25."""
    return _TOKEN_RE.findall(text.lower())


# ── Main class ────────────────────────────────────────────────────────────────

class BM25ProductIndex:
    """In-memory BM25 index over the product catalog.

    Parameters
    ----------
    source : {"postgres", "csv"}, default "postgres"
        Where to load the catalog from. Falls back to CSV automatically
        if the PostgreSQL connection fails.
    max_rows : int | None
        Cap on rows loaded (useful for testing).  ``None`` loads all rows.
    """

    def __init__(
        self,
        source: str = "postgres",
        max_rows: int | None = None,
    ) -> None:
        self._records: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._source = source
        self._max_rows = max_rows
        self._build()

    # ── Public API ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        price_min: float | None = None,
        price_max: float | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-K most relevant products for ``query``.

        Applies optional hard filters (price range, category) *after* BM25
        scoring to preserve ranking quality.

        Parameters
        ----------
        query : str
            Natural-language product search query.
        top_k : int
            Number of results to return.
        price_min : float | None
            Minimum price filter (inclusive).
        price_max : float | None
            Maximum price filter (inclusive).
        category : str | None
            Case-insensitive substring match against ``main_category``.

        Returns
        -------
        list[dict]
            Ranked list of product dicts, highest relevance first.
        """
        if self._bm25 is None or not self._records:
            logger.warning("BM25 index is empty — returning no results.")
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Pair every record with its score, sort descending
        scored = sorted(
            zip(self._records, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        for record, score in scored:
            if score <= 0:
                break  # BM25 score of 0 means no term overlap

            # Hard filters
            price = record.get("price")
            if price_min is not None and (price is None or price < price_min):
                continue
            if price_max is not None and (price is None or price > price_max):
                continue
            if category is not None:
                cat = str(record.get("main_category", "") or "").lower()
                if category.lower() not in cat:
                    continue

            results.append({**record, "_bm25_score": round(float(score), 4)})
            if len(results) >= top_k:
                break

        return results

    def __len__(self) -> int:
        return len(self._records)

    # ── Internal ───────────────────────────────────────────────────────────

    def _build(self) -> None:
        """Load catalog and build BM25 index."""
        t0 = time.perf_counter()
        df = self._load_catalog()
        if df.empty:
            logger.error("Product catalog is empty — BM25 index not built.")
            return

        if self._max_rows:
            df = df.head(self._max_rows)

        self._records = df.to_dict(orient="records")
        corpus = [_tokenize(_build_text(row)) for _, row in df.iterrows()]
        self._bm25 = BM25Okapi(corpus)
        elapsed = time.perf_counter() - t0
        logger.info(
            "BM25 index built: %d products indexed in %.1fs",
            len(self._records),
            elapsed,
        )

    def _load_catalog(self) -> pd.DataFrame:
        """Try PostgreSQL first, then fall back to CSV."""
        if self._source == "postgres":
            try:
                return self._load_from_postgres()
            except Exception as exc:
                logger.warning(
                    "PostgreSQL load failed (%s) — falling back to CSV.", exc
                )
        return self._load_from_csv()

    def _load_from_postgres(self) -> pd.DataFrame:
        from src.data.postgresql import get_db_engine

        engine = get_db_engine()
        cols = "product_id, title, main_category, price, average_rating, rating_count, description, features, store, is_bestseller"
        query = f"SELECT {cols} FROM product_catalog"  # noqa: S608
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        logger.info("Loaded %d rows from PostgreSQL.", len(df))
        return df

    def _load_from_csv(self) -> pd.DataFrame:
        if not _CATALOG_CSV.exists():
            logger.error("Catalog CSV not found at %s", _CATALOG_CSV)
            return pd.DataFrame()
        df = pd.read_csv(
            _CATALOG_CSV,
            usecols=[
                "product_id", "title", "main_category", "price",
                "average_rating", "rating_count", "description",
                "features", "store", "is_bestseller",
            ],
            low_memory=False,
        )
        logger.info("Loaded %d rows from CSV.", len(df))
        return df


# ── Module-level singleton (lazy) ─────────────────────────────────────────────
_INDEX: BM25ProductIndex | None = None


def get_bm25_index() -> BM25ProductIndex:
    """Return (and cache) the module-level BM25 index singleton.

    The index is built on first call and reused for subsequent calls,
    avoiding repeated I/O and indexing overhead during the agent's lifetime.
    """
    global _INDEX  # noqa: PLW0603
    if _INDEX is None:
        logger.info("Initialising BM25ProductIndex singleton…")
        _INDEX = BM25ProductIndex()
    return _INDEX
