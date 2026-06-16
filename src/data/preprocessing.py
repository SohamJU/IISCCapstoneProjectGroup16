"""Data preprocessing utilities for structured and unstructured datasets."""

from __future__ import annotations

import html
import math
import re
from pathlib import Path

import pandas as pd

from src.config.data import (
    AMAZON_PROCESSED_DATA_DIR,
    AMAZON_PROCESSED_DATA_FILENAME,
    PRODUCT_CATALOG_PROCESSED_FILENAME,
    TWITTER_SUPPORT_HISTORY_FILENAME,
    TWITTER_SUPPORT_PROCESSED_FILENAME,
)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")
LEADING_MENTIONS_PATTERN = re.compile(r"^(?:@\w+\s+)+")

PRODUCT_ID_CANDIDATES = ["product_id", "asin", "sku", "id", "item_id"]
PRODUCT_TITLE_CANDIDATES = ["title", "name", "product_name"]
PRODUCT_TEXT_CANDIDATES = [
    "description",
    "about",
    "specifications",
    "features",
    "review_text",
    "review",
    "summary",
]


def preprocess_ratings(
    df: pd.DataFrame,
    min_user_interactions: int = 5,
    min_product_interactions: int = 5,
) -> pd.DataFrame:
    """Preprocess Amazon Electronics ratings dataset."""

    del min_user_interactions
    del min_product_interactions

    print(f"Initial shape: {df.shape}")

    df = df.drop_duplicates()
    df = df.dropna()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df[(df["rating"] >= 1.0) & (df["rating"] <= 5.0)]

    print(f"Final shape: {df.shape}")

    return df


def clean_text(
    text: object,
    *,
    remove_urls: bool = True,
    remove_leading_mentions: bool = False,
) -> str:
    """Normalize raw text while preserving useful conversational meaning."""

    if text is None or pd.isna(text):
        return ""

    cleaned_text = html.unescape(str(text))
    cleaned_text = cleaned_text.replace("\r", " ").replace("\n", " ")

    if remove_urls:
        cleaned_text = URL_PATTERN.sub("", cleaned_text)

    if remove_leading_mentions:
        cleaned_text = LEADING_MENTIONS_PATTERN.sub("", cleaned_text)

    cleaned_text = WHITESPACE_PATTERN.sub(" ", cleaned_text).strip()
    return cleaned_text


def preprocess_customer_support_conversations(
    df: pd.DataFrame,
    *,
    customer_only: bool = False,
    remove_leading_mentions: bool = True,
) -> pd.DataFrame:
    """Clean and organize customer support conversations for downstream RAG."""

    required_columns = {
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required conversation columns: {missing}")

    conversations = df.copy()

    conversations["tweet_id"] = conversations["tweet_id"].apply(_normalize_identifier)
    conversations["author_id"] = conversations["author_id"].astype(str)
    conversations["response_tweet_id"] = conversations["response_tweet_id"].apply(
        _normalize_identifier
    )
    conversations["in_response_to_tweet_id"] = conversations["in_response_to_tweet_id"].apply(
        _normalize_identifier
    )
    conversations["inbound"] = conversations["inbound"].astype(str).str.lower().eq("true")
    conversations["speaker_type"] = conversations["inbound"].map(
        {True: "customer", False: "support"}
    )
    conversations["created_at"] = pd.to_datetime(
        conversations["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
        utc=True,
    )
    conversations["text"] = conversations["text"].apply(
        lambda value: clean_text(
            value,
            remove_urls=True,
            remove_leading_mentions=remove_leading_mentions,
        )
    )

    conversations = conversations.dropna(subset=["created_at"])
    conversations = conversations[conversations["text"].astype(bool)]
    conversations = conversations[conversations["tweet_id"].astype(bool)]
    conversations = conversations.drop_duplicates(subset=["tweet_id"])

    if customer_only:
        conversations = conversations[conversations["inbound"]]

    conversations["conversation_id"] = _resolve_conversation_ids(conversations)
    conversations = conversations.sort_values(
        ["conversation_id", "created_at", "tweet_id"]
    ).reset_index(drop=True)
    conversations["turn_index"] = conversations.groupby("conversation_id").cumcount() + 1
    conversations["source"] = "twitter_support"
    conversations["doc_id"] = conversations["tweet_id"]

    return conversations[
        [
            "doc_id",
            "conversation_id",
            "turn_index",
            "tweet_id",
            "author_id",
            "speaker_type",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
            "source",
        ]
    ]


def preprocess_product_catalog(
    df: pd.DataFrame,
    *,
    id_column: str | None = None,
    title_column: str | None = None,
    text_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize product catalog records into a retrieval-friendly document table."""

    products = df.copy()

    if id_column is None:
        id_column = _first_existing_column(products, PRODUCT_ID_CANDIDATES)
    if title_column is None:
        title_column = _first_existing_column(products, PRODUCT_TITLE_CANDIDATES)
    if text_columns is None:
        text_columns = [column for column in PRODUCT_TEXT_CANDIDATES if column in products]

    if not text_columns and title_column is None:
        raise ValueError(
            "Could not infer product text columns. Provide title_column or text_columns."
        )

    if id_column is None:
        products["product_id"] = [f"product_{index}" for index in range(len(products))]
        id_column = "product_id"
    else:
        products[id_column] = products[id_column].astype(str)

    if title_column is None:
        products["title"] = ""
        title_column = "title"

    for column in [title_column, *text_columns]:
        products[column] = products[column].fillna("").astype(str)

    text_source_columns = [title_column, *text_columns]
    products["text"] = products[text_source_columns].apply(
        lambda row: clean_text(" ".join(value for value in row if value.strip())),
        axis=1,
    )

    products = products[products["text"].astype(bool)].copy()
    products = products.drop_duplicates(subset=[id_column, "text"]).reset_index(drop=True)

    products["doc_id"] = products[id_column].map(lambda value: f"product::{value}")
    products["product_id"] = products[id_column]
    products["title"] = products[title_column].apply(clean_text)
    products["source"] = "product_catalog"

    selected_columns = ["doc_id", "product_id", "title", "text", "source"]
    metadata_columns = [
        column
        for column in products.columns
        if column not in selected_columns and column not in text_source_columns
    ]
    output_columns = selected_columns + metadata_columns

    return products[output_columns]


def build_conversation_history(
    conversations: pd.DataFrame,
    *,
    max_turns: int | None = None,
) -> pd.DataFrame:
    """Aggregate ordered conversation threads into compact history records."""

    if "conversation_id" not in conversations.columns or "text" not in conversations.columns:
        raise ValueError("Conversations dataframe must include conversation_id and text.")

    ordered = conversations.sort_values(
        ["conversation_id", "turn_index", "created_at", "tweet_id"],
        na_position="last",
    ).copy()

    if max_turns is not None:
        ordered = ordered.groupby("conversation_id").tail(max_turns)

    history_rows = []

    for conversation_id, group in ordered.groupby("conversation_id", sort=False):
        turns = [
            f"{row.speaker_type}: {row.text}"
            for row in group.itertuples(index=False)
            if row.text
        ]
        latest_customer_turns = group[group["speaker_type"] == "customer"]["text"].tolist()

        history_rows.append(
            {
                "conversation_id": conversation_id,
                "turn_count": len(group),
                "customer_turn_count": int((group["speaker_type"] == "customer").sum()),
                "support_turn_count": int((group["speaker_type"] == "support").sum()),
                "conversation_text": "\n".join(turns),
                "latest_customer_text": latest_customer_turns[-1] if latest_customer_turns else "",
                "started_at": group["created_at"].min(),
                "ended_at": group["created_at"].max(),
                "source": "twitter_support",
            }
        )

    return pd.DataFrame(history_rows)


def save_processed_data(df: pd.DataFrame):
    """Save the structured ratings dataframe."""

    AMAZON_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AMAZON_PROCESSED_DATA_DIR / AMAZON_PROCESSED_DATA_FILENAME
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")


def save_unstructured_data(
    df: pd.DataFrame,
    *,
    filename: str,
    output_dir: Path | None = None,
) -> Path:
    """Persist an unstructured dataset artifact to the processed data directory."""

    if output_dir is None:
        output_dir = AMAZON_PROCESSED_DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    return output_path


def save_customer_support_conversations(df: pd.DataFrame) -> Path:
    """Save processed customer support conversations."""

    return save_unstructured_data(df, filename=TWITTER_SUPPORT_PROCESSED_FILENAME)


def save_conversation_history(df: pd.DataFrame) -> Path:
    """Save aggregated conversation history."""

    return save_unstructured_data(df, filename=TWITTER_SUPPORT_HISTORY_FILENAME)


def save_product_catalog(df: pd.DataFrame) -> Path:
    """Save processed product catalog documents."""

    return save_unstructured_data(df, filename=PRODUCT_CATALOG_PROCESSED_FILENAME)


def _resolve_conversation_ids(conversations: pd.DataFrame) -> pd.Series:
    """Assign a stable conversation id by walking reply chains to the root tweet."""

    parent_by_tweet = (
        conversations.set_index("tweet_id")["in_response_to_tweet_id"].fillna("").to_dict()
    )
    resolved_ids: dict[str, str] = {}

    def resolve_root(tweet_id: str) -> str:
        if tweet_id in resolved_ids:
            return resolved_ids[tweet_id]

        visited = set()
        current_id = tweet_id

        while True:
            parent_id = parent_by_tweet.get(current_id, "")
            if not parent_id or parent_id == "nan" or parent_id in visited:
                root_id = current_id
                break
            visited.add(parent_id)
            if parent_id not in parent_by_tweet:
                root_id = parent_id
                break
            current_id = parent_id

        resolved_ids[tweet_id] = root_id
        return root_id

    return conversations["tweet_id"].map(resolve_root)


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column from a candidate list."""

    for column in candidates:
        if column in df.columns:
            return column
    return None


def _normalize_identifier(value: object) -> str:
    """Normalize ids so `123`, `123.0`, and numeric strings resolve the same way."""

    if value is None:
        return ""

    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return str(value).strip()

    if isinstance(value, int):
        return str(value)

    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    if text.endswith(".0"):
        integer_part = text[:-2]
        if integer_part.isdigit():
            return integer_part

    return text
