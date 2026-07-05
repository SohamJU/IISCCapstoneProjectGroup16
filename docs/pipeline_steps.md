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
   - Ensure a `.env` file exists in the root directory.
   - It should contain necessary keys such as database connection credentials (e.g., `POSTGRESQL_AIVEN_PASSWORD`, `POSTGRESQL_HOST`, etc.).

---

## 2. Running the Synthetic Data Pipeline

The synthetic data pipeline (`pipelines/01_run_synthetic_data_pipeline.py`) manages the ETL process, generates synthetic records (customers, orders, returns, queries), and uploads them to a cloud PostgreSQL database.

The pipeline consists of 13 discrete steps. By default, it detects already completed steps and safely skips them.

### Run the Full Pipeline
To execute the entire pipeline end-to-end:
```bash
python notebooks/02_run_synthetic_data_pipeline.py
```

### Force a Complete Re-run
To ignore completed steps, overwrite all existing data, and start fresh from Step 1:
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

For more detailed information regarding the individual 13 steps and the variables defined in `src/config/data.py`, refer to `docs/individual pipeline docs/01_data_pipeline.md`.
