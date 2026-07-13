# Pipeline Setup and Execution Guide

This document outlines the steps required to set up the project environment and run the data pipeline.

## 1. Project Setup

### Prerequisites
- **Python:** Ensure you have Python 3.12 installed on your system.
- **uv Package Manager:** We use `uv` for lightning-fast dependency management. You can install it using pip if you haven't already:
  ```bash
  pip install uv
  ```

### Environment Setup Steps
1. **Clone the repository and navigate into it:**
   ```bash
   cd IISCCapstoneProjectGroup16
   ```

2. **Create a virtual environment using Python 3.12:**
   ```bash
   uv venv --python 3.12
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies from `pyproject.toml`:**
   Using `uv sync` will automatically read the `pyproject.toml` and `uv.lock` files to install the exact required versions:
   ```bash
   uv sync
   ```

5. **Configure Environment Variables:**
   - Ensure a `.env` file exists in the repository root.
   - The app loads this file automatically via `python-dotenv` in `src/config/data.py`.
   - The key required by the pipeline is:
     * `POSTGRESQL_AIVEN_PASSWORD` — required for any PostgreSQL upload or query operations.
   - Optional overrides with defaults are [For now, Do not set them]:
     * `POSTGRESQL_HOST` — defaults to the shared Aiven host.
     * `POSTGRESQL_PORT` — defaults to the shared Aiven port.
     * `POSTGRESQL_USER` — defaults to `avnadmin`.
     * `POSTGRESQL_DB` — defaults to `defaultdb`.
   - Example `.env` entries:
     ```env
     POSTGRESQL_AIVEN_PASSWORD=your_password_here
     # POSTGRESQL_HOST=pg-20bad560-myproject123-456.e.aivencloud.com
     # POSTGRESQL_PORT=12548
     # POSTGRESQL_USER=avnadmin
     # POSTGRESQL_DB=defaultdb

 - More Info:
    Add Extenstion for Postgressql (Microsoft) - Elephant Icon
    Go to Parameters Tab and Add the details above. 
    Go to Advanced and Don't forget to Add Port there
     ```

### VS Code Configuration
The workspace includes `.vscode/settings.json` and `.vscode/launch.json` to help enable linting and debugging in VS Code.

- `.vscode/settings.json`
  - `python.linting.enabled`: turns Python linting on in VS Code.
  - `python.linting.ruffEnabled`: enables the Ruff linter specifically.
  - `python.linting.ruffPath`: points VS Code to the Ruff executable in the local virtual environment.
  - `python.linting.ruffArgs`: passes the project config file `.ruff.toml` to Ruff.
  - `python.envFile`: loads environment variables from the repository `.env` file.

- `.vscode/launch.json`
  - `name`: the debug configuration label shown in VS Code.
  - `type`: uses `debugpy` for Python debugging.
  - `request`: `launch` means VS Code starts a new debug session.
  - `program`: `${file}` tells VS Code to run the currently-open file.
  - `console`: `integratedTerminal` runs output in the built-in terminal.
  - `env`: sets `PYTHONPATH` so imports resolve from the workspace root.

To enable linting and debugging in VS Code:
1. Open the workspace in VS Code.
2. Confirm `.vscode/settings.json` exists with the lint settings above.
3. Confirm `.vscode/launch.json` exists with the debug configuration above.
4. Open any Python file and save it to trigger Ruff linting automatically.

---
The Twitter processing pipeline prepares customer support conversation data for downstream tasks, including conversation history creation and repository-ready CSV output.
The details are available in the file 'docs/individual_pipeline_docs/01_twitter_data_pipeline.md'

## 2. Run the Twitter preprocessing script
From the repository root, execute:
```bash
python src/data/twitter_data_pipeline/run_twitter_preprocessing.py --dataset twitter --twitter-path archive/twcs/twcs.csv
```

### Run the wrapper script in `pipelines/`
If you prefer the pipeline wrapper, use:
```bash
python pipelines/01_run_twitter_data_pipeline.py
```

### Notes
- The script supports other preprocessing targets:
  - `--dataset ratings` for the structured ratings dataset
  - `--dataset product` for product catalog preprocessing
  - `--dataset all` to run all preprocessing flows
- The default `--twitter-path` location is `archive/sample.csv` when the file path is omitted.

---


## 3. Running the Synthetic Data Pipeline

The synthetic data pipeline (`pipelines/01_run_synthetic_data_pipeline.py`) manages the ETL process, generates synthetic records (customers, orders, returns, queries), and uploads them to a cloud PostgreSQL database.

The pipeline consists of 13 discrete steps. By default, it detects already completed steps and safely skips them.
The details are available in the file 'docs/individual_pipeline_docs/01_amazon_data_pipeline.md'

### Run the Full Pipeline
To execute the entire pipeline end-to-end:
```bash
python notebooks/02_run_synthetic_data_pipeline.py
If above fails due to src pathing, try python -m pipelines.02_run_synthetic_data_pipeline
```

### Notes on query generation
Step 8 now runs both the generic query generator and the enhanced query generator, then merges their outputs into a single `data/synthetic/customer_queries.csv` file.

### Force a Complete Re-run
CAUTION : To IGNORE completed steps, OVERWRITE all existing data, and start fresh from Step 1:
```bash
python pipelines/02_run_synthetic_data_pipeline.py --force
```

### Run a Specific Step
If you want to execute a single step (e.g., just Step 11 for Schema Generation, or Step 13 for Database Testing):
```bash
# Run only Step 11, forcing it to overwrite existing schema files
python pipelines/02_run_synthetic_data_pipeline.py --step 11 --force

# Run only Step 13
python pipelines/02_run_synthetic_data_pipeline.py --step 13
```

### Custom Scaling
You can modify the volume of synthetic data generated (customers, orders, etc.) on the fly by passing flags:
```bash
python pipelines/02_run_synthetic_data_pipeline.py --num-customers 5000 --num-orders 20000 --total-queries 1000
```

### Modify Database Upload Behavior
By default, the pipeline upload behavior is driven by your configuration (e.g., `skip`, `replace`, or `append`). You can override this behavior dynamically for Step 12:
```bash
# Force the pipeline to append to existing cloud tables
python pipelines/02_run_synthetic_data_pipeline.py --step 12 --postgres-behavior append

# Force the pipeline to drop and replace tables
python pipelines/02_run_synthetic_data_pipeline.py --step 12 --postgres-behavior replace
```

Note on the default `skip` behavior
----------------------------------
The project configuration (`src/config/data.py`) defines `POSTGRESQL_UPLOAD_BEHAVIOR = "skip"` by default. When this value is set to `skip`, Step 12 (the Postgres upload step) will be a no-op: the pipeline will detect the setting and intentionally skip attempting any uploads to the cloud database. This is useful for local development or when you do not want the pipeline to modify the shared cloud database.

If you want the pipeline to actually perform uploads, either:
- Pass `--postgres-behavior append` or `--postgres-behavior replace` to the Step 12 invocation (examples above), or
- Change the `POSTGRESQL_UPLOAD_BEHAVIOR` value in `src/config/data.py` to `append` or `replace` before running the pipeline.

When using `append`, existing tables are retained and new rows are appended. When using `replace`, existing tables will be dropped and re-created from the pipeline output.

### PostgreSQL tables created by the Amazon pipeline
When the Amazon synthetic pipeline uploads datasets to PostgreSQL, it uses the table mappings defined in `src/config/data.py`.

The uploaded tables are:
- `product_catalog`  — generated from `data/processed/product_catalog.csv`
- `reviews`          — generated from `data/processed/reviews.csv`
- `customers`        — generated from `data/synthetic/customers.csv`
- `orders`           — generated from `data/synthetic/orders.csv`
- `order_items`      — generated from `data/synthetic/order_items.csv`
- `returns`          — generated from `data/synthetic/returns.csv`
- `customer_queries` — generated from `data/synthetic/customer_queries.csv`

#### Querying the tables
Use normal SQL against the target PostgreSQL database. Example queries:
```sql
SELECT * FROM customers LIMIT 10;
SELECT COUNT(*) FROM orders;
SELECT order_id, customer_id, total_amount FROM orders WHERE total_amount > 100;
SELECT * FROM order_items WHERE order_id = 'ORD-000123';
SELECT * FROM customer_queries WHERE query_type = 'returns';
```

#### Querying from Python using `execute_sql_query`
The helper in `src/data/postgresql.py` can execute SQL directly against the configured PostgreSQL database. Example usage:
```python
from src.data.postgresql import execute_sql_query

result = execute_sql_query("SELECT * FROM customers LIMIT 10;")
print(result)
```

This returns a list of dictionaries for `SELECT` queries, where each row maps column names to values.

#### Where to find table schemas
Schema files are generated by Step 11 and saved next to the source CSVs with the `.schema.json` suffix.

The schema files are typically located at:
- `data/processed/product_catalog.schema.json`
- `data/processed/reviews.schema.json`
- `data/synthetic/customers.schema.json`
- `data/synthetic/orders.schema.json`
- `data/synthetic/order_items.schema.json`
- `data/synthetic/returns.schema.json`
- `data/synthetic/customer_queries.schema.json`

Run schema generation explicitly with:
```bash
python pipelines/02_run_synthetic_data_pipeline.py --step 11
```

For more detailed information regarding the individual 13 steps and the variables defined in `src/config/data.py`, refer to `docs/individual pipeline docs/01_data_pipeline.md`.

---


