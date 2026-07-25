"""BM25 keyword index over the product catalog.

Builds an in-memory BM25 index (using ``rank_bm25``) over the concatenated
text of ``title``, ``description``, and ``features`` columns.  The index is
loaded lazily on first use from PostgreSQL (or falls back to the processed CSV
if the DB is unavailable).

**Local persistence:** The built index is automatically saved to disk as a
pickle file (default: ``data/processed/bm25_index.pkl``). On subsequent
starts, the index is loaded from the pickle (~1s) instead of being rebuilt
from the database (~30s for 194k products).

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
import pickle
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
    cache_path : Path | None
        Path to a pickle file for persisting the index. If the file exists,
        the index is loaded from it instead of being rebuilt. If ``None``,
        no caching is used (in-memory only, rebuilt every time).
    """

    def __init__(
        self,
        source: str = "postgres",
        max_rows: int | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self._records: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._source = source
        self._max_rows = max_rows
        self._cache_path = cache_path
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

    # ── Persistence ────────────────────────────────────────────────────────

    def save_to_disk(self, path: Path | None = None) -> None:
        """Save the index (records + BM25 model) to a pickle file.

        Parameters
        ----------
        path : Path | None
            Destination file. Uses ``self._cache_path`` if not specified.
        """
        save_path = path or self._cache_path
        if save_path is None:
            logger.debug("No cache path configured — skipping save.")
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": self._records,
            "bm25": self._bm25,
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "BM25 index saved to disk: %s (%d products).",
            save_path,
            len(self._records),
        )

    @classmethod
    def load_from_disk(cls, path: Path) -> BM25ProductIndex:
        """Load a previously saved BM25 index from a pickle file.

        Parameters
        ----------
        path : Path
            Path to the pickle file.

        Returns
        -------
        BM25ProductIndex
            A fully initialised instance with the loaded index.

        Raises
        ------
        FileNotFoundError
            If the pickle file does not exist.
        """
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        instance = cls.__new__(cls)
        instance._records = data["records"]
        instance._bm25 = data["bm25"]
        instance._source = "cache"
        instance._max_rows = None
        instance._cache_path = path

        elapsed = time.perf_counter() - t0
        logger.info(
            "BM25 index loaded from cache: %d products in %.2fs.",
            len(instance._records),
            elapsed,
        )
        return instance

    # ── Internal ───────────────────────────────────────────────────────────

    def _build(self) -> None:
        """Load catalog and build BM25 index.

        If a cache file exists at ``self._cache_path``, loads from it instead
        of rebuilding from the database. After building fresh, auto-saves to
        the cache path for next time.
        """
        # Try loading from disk cache first
        if self._cache_path and self._cache_path.exists():
            try:
                cached = BM25ProductIndex.load_from_disk(self._cache_path)
                self._records = cached._records
                self._bm25 = cached._bm25
                return
            except Exception as exc:
                logger.warning(
                    "Failed to load BM25 cache (%s) — rebuilding from source.",
                    exc,
                )

        # Build fresh from database/CSV
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

        # Auto-save to disk for next time
        self.save_to_disk()

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

    Uses ``BM25_CACHE_PATH`` from config to persist the index on disk.
    On first run, builds from DB/CSV and saves the pickle.
    On subsequent runs, loads from the pickle (~1s vs ~30s rebuild).
    """
    global _INDEX  # noqa: PLW0603
    if _INDEX is None:
        from src.agents.product_agent.config import BM25_CACHE_PATH  # noqa: PLC0415

        logger.info("Initialising BM25ProductIndex singleton...")
        _INDEX = BM25ProductIndex(cache_path=BM25_CACHE_PATH)
    return _INDEX
