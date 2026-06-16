# Preprocessing Guide

This document currently covers the unstructured data preprocessing flow.

The structured preprocessing section can be added later in the same file without changing the unstructured instructions below.

## Unstructured Data Preprocessing

### Dataset Used

The current unstructured preprocessing flow uses the Kaggle dataset:
- `Customer Support on Twitter`

### How To Download And Place The Dataset

1. Download the Kaggle dataset `Customer Support on Twitter`.
2. Extract the downloaded archive.
3. Copy the extracted dataset into the repository `archive/` directory.

Expected layout inside the repository:

```text
archive/
├── sample.csv
└── twcs/
    └── twcs.csv
```

Notes:
- `sample.csv` is useful for quick testing and debugging on a small subset.
- `twcs/twcs.csv` is the full dataset file used by default when you run the Twitter preprocessing flow without a custom path.

### What The Preprocessing Does

The current implementation:
- loads the Twitter support dataset
- cleans tweet text
- removes URLs
- decodes HTML entities such as `&amp;`
- optionally removes leading Twitter mentions
- identifies whether each tweet is from a customer or support account
- reconstructs reply chains into ordered conversations
- assigns a `conversation_id` and `turn_index`
- builds conversation-history records for later RAG and memory usage

### Files Created

Running the Twitter preprocessing flow creates:
- `data/processed/twitter_support_processed.csv`
- `data/processed/twitter_support_conversation_history.csv`

## Output File Schemas

### `twitter_support_processed.csv`

This is the cleaned row-level conversation file.
Each row represents one tweet/message.

Columns:
- `doc_id`: unique document identifier for the processed row
- `conversation_id`: root conversation/thread identifier shared by all tweets in the same thread
- `turn_index`: order of the message inside the conversation
- `tweet_id`: original tweet id from the Kaggle dataset
- `author_id`: original author id or support handle
- `speaker_type`: normalized role, either `customer` or `support`
- `inbound`: boolean flag from the dataset indicating whether the message is inbound from the customer
- `created_at`: parsed tweet timestamp
- `text`: cleaned tweet text after preprocessing
- `response_tweet_id`: original response tweet id if available
- `in_response_to_tweet_id`: original parent tweet id if available
- `source`: source label, currently always `twitter_support`

### `twitter_support_conversation_history.csv`

This is the conversation-level history file.
Each row represents one complete conversation/thread.

Columns:
- `conversation_id`: root thread identifier
- `turn_count`: total number of messages in the conversation
- `customer_turn_count`: number of customer messages in the conversation
- `support_turn_count`: number of support messages in the conversation
- `conversation_text`: stitched conversation text in chronological order using `customer:` and `support:` prefixes
- `latest_customer_text`: most recent customer-side message in the thread
- `started_at`: timestamp of the first message in the conversation
- `ended_at`: timestamp of the last message in the conversation
- `source`: source label, currently always `twitter_support`

## Files Added Or Updated For This Flow

- `src/data/preprocessing.py`
- `src/data/loaders.py`
- `src/config/data.py`
- `src/memory/conversation_memory.py`
- `src/memory/session_manager.py`
- `scripts/run_preprocessing.py`

## How To Run

Run from the repository root.

### Default Twitter preprocessing

This uses the full dataset from `archive/twcs/twcs.csv`.

```bash
python3 scripts/run_preprocessing.py --dataset twitter
```

### Run on the sample file

This is useful for quick validation.

```bash
python3 scripts/run_preprocessing.py --dataset twitter --twitter-path archive/sample.csv
```

### Keep only customer-side tweets

```bash
python3 scripts/run_preprocessing.py --dataset twitter --customer-only
```

## Python Usage

```python
from src.data.loaders import load_twitter_support_conversations
from src.data.preprocessing import (
    build_conversation_history,
    preprocess_customer_support_conversations,
    save_conversation_history,
    save_customer_support_conversations,
)

df = load_twitter_support_conversations()
processed_df = preprocess_customer_support_conversations(df)
history_df = build_conversation_history(processed_df)

save_customer_support_conversations(processed_df)
save_conversation_history(history_df)
```

## Notes

- Tweets are already short, so text chunking is not required yet.
- Conversation history is intentionally lightweight so it can later be reused by the RAG pipeline and memory components.
- The preprocessing code also contains helper support for product catalog text, but this README section is intentionally focused on the current unstructured Twitter conversation flow.

## Placeholder For Structured Preprocessing

This section is intentionally left open for future updates to the structured data preprocessing flow.
