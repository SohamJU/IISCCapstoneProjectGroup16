import pandas as pd

from src.data.twitter_data_pipeline.unstructured_preprocessing import (
    build_conversation_history,
    preprocess_customer_support_conversations,
    preprocess_product_catalog,
)


def test_preprocess_customer_support_conversations_builds_threads():
    raw_df = pd.DataFrame(
        [
            {
                "tweet_id": "2",
                "author_id": "brand",
                "inbound": False,
                "created_at": "Wed Oct 11 13:25:49 +0000 2017",
                "text": "@105835 Please DM us your order number. https://t.co/example",
                "response_tweet_id": "",
                "in_response_to_tweet_id": "1",
            },
            {
                "tweet_id": "1",
                "author_id": "customer",
                "inbound": True,
                "created_at": "Wed Oct 11 13:00:09 +0000 2017",
                "text": "@Brand my package is delayed &amp; I need help",
                "response_tweet_id": "2",
                "in_response_to_tweet_id": "",
            },
            {
                "tweet_id": "3",
                "author_id": "customer",
                "inbound": True,
                "created_at": "Wed Oct 11 13:40:00 +0000 2017",
                "text": "@Brand It is order 12345",
                "response_tweet_id": "",
                "in_response_to_tweet_id": "2",
            },
        ]
    )

    processed_df = preprocess_customer_support_conversations(raw_df)

    assert list(processed_df["turn_index"]) == [1, 2, 3]
    assert processed_df["conversation_id"].nunique() == 1
    assert processed_df.iloc[0]["text"] == "my package is delayed & I need help"
    assert "https://t.co/example" not in processed_df.iloc[1]["text"]
    assert list(processed_df["conversation_id"]) == ["1", "1", "1"]


def test_preprocess_customer_support_conversations_normalizes_float_reply_ids():
    raw_df = pd.DataFrame(
        [
            {
                "tweet_id": 119239,
                "author_id": "customer",
                "inbound": True,
                "created_at": "Wed Oct 11 13:00:09 +0000 2017",
                "text": "@Brand Need help with my card",
                "response_tweet_id": "119238",
                "in_response_to_tweet_id": float("nan"),
            },
            {
                "tweet_id": 119238,
                "author_id": "ChaseSupport",
                "inbound": False,
                "created_at": "Wed Oct 11 13:25:49 +0000 2017",
                "text": "@105835 Please DM us more details",
                "response_tweet_id": "",
                "in_response_to_tweet_id": 119239.0,
            },
        ]
    )

    processed_df = preprocess_customer_support_conversations(raw_df)

    assert list(processed_df["conversation_id"]) == ["119239", "119239"]
    assert list(processed_df["turn_index"]) == [1, 2]


def test_build_conversation_history_aggregates_context():
    conversations_df = pd.DataFrame(
        [
            {
                "conversation_id": "10",
                "turn_index": 1,
                "created_at": pd.Timestamp("2024-01-01T10:00:00Z"),
                "tweet_id": "10",
                "speaker_type": "customer",
                "text": "My headphones stopped working.",
            },
            {
                "conversation_id": "10",
                "turn_index": 2,
                "created_at": pd.Timestamp("2024-01-01T10:05:00Z"),
                "tweet_id": "11",
                "speaker_type": "support",
                "text": "Can you share the model number?",
            },
        ]
    )

    history_df = build_conversation_history(conversations_df)

    assert history_df.iloc[0]["turn_count"] == 2
    assert "customer: My headphones stopped working." in history_df.iloc[0]["conversation_text"]
    assert history_df.iloc[0]["latest_customer_text"] == "My headphones stopped working."


def test_preprocess_product_catalog_infers_text_fields():
    raw_products = pd.DataFrame(
        [
            {
                "asin": "A1",
                "title": "Bluetooth Speaker",
                "description": "Portable waterproof speaker",
                "features": "12 hour battery",
                "category": "audio",
            }
        ]
    )

    processed_products = preprocess_product_catalog(raw_products)

    assert processed_products.iloc[0]["doc_id"] == "product::A1"
    assert "Portable waterproof speaker" in processed_products.iloc[0]["text"]
    assert processed_products.iloc[0]["source"] == "product_catalog"
