# 🌊 ArgoPipeline – Float Data Processing & RAG Pipeline

This repository processes Argo float data, validates outputs, loads into DuckDB & ChromaDB, and enables semantic + SQL queries via an LLM RAG pipeline.

---

## 📂 Repository Structure

```
ArgoPipeline/
│
├── chroma_store/        # Local storage for Chroma vector DB (ignored in git)
├── data/                # Raw input data (CSV/NetCDF) organized by year/month (ignored in git)
├── notebooks/           # Jupyter notebooks for exploration & experiments
├── output/              # Processed CSV outputs, logs, summaries (ignored in git)
├── parquet/             # Parquet versions of processed data (used for DuckDB + Chroma ingest)
├── pipeline/            # Core Python pipeline (RAG, validators, query engine, etc.)
├── pipeline_audits/     # Logs & audits of pipeline runs (ignored in git)
├── pipeline_outputs/    # JSON outputs from RAG queries (kept for inspection)
├── pipeline_tests/      # Test scripts & regression checks
├── venv/                # Python virtual environment (ignored in git)
│
├── .env                 # Environment variables (Azure, Chroma paths, etc.) (ignored in git)
├── .gitignore           # Ignores: venv, .env, chroma_store, data, output, pipeline_audits
├── argo.duckdb          # DuckDB database storing float data
├── requirements.txt     # Python dependencies
├── viewer.ipynb         # Notebook to visualize/query processed data
└── ReadMe.md            # This file
```

---

## 🛠️ Setup

1. **Clone the repo**

   ```bash
   git clone <repo-url>
   cd ArgoPipeline
   ```

2. **Create & activate virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   Copy `.env.example` → `.env` and add required keys (Azure OpenAI, Chroma paths, etc.).

---

## 🚀 CLI Commands

### 1. **Batch Processing (monthly)**

Process a single month’s raw data into output + logs.

```bash
python batch_runner.py -i ../data/2025/09 -o ../output/2025/09 --log ../output/2025/09/process_log.csv
```

### 2. **Yearly Runner**

Process all months in a year.

```bash
python year_runner.py -y 2025 -d ../data -o ../output
```

### 3. **Convert CSV → Parquet**

Convert processed CSVs into Parquet.

```bash
python csv_to_parquet.py
python csv_to_parquet.py -y 2025 -o ../output -p ../parquet
```

### 4. **Validate Parquet**

Check parquet integrity & schema consistency.

```bash
python validate_parquet.py -y 2025 -o ../output -p ../parquet
```

### 5. **Load into DuckDB**

Populate DuckDB from Parquet.

```bash
python duckdb_loader.py -y 2025 -p ../parquet -d ../argo.duckdb
```

### 6. **Validate DuckDB**

Check tables & queries against DuckDB.

```bash
python pipeline/validate_duckdb.py
```

### 7. **Ingest into ChromaDB**

Embed and store parquet data into Chroma.

```bash
python vector_ingest.py -y 2025 -p ../parquet
```

### 8. **Validate ChromaDB**

Check vector store integrity.

```bash
python validate_chroma.py
```

### 9. **Run RAG Pipeline Query**

Query via semantic retrieval + DuckDB + Azure OpenAI.

```bash
python pipeline/rag_pipeline.py -q "Show me salinity profiles near the equator in March 2025" -y 2025 -m 03 --top_k 30 --out ./pipeline_outputs/rag_equator_mar2025.json
```

### 10. **Validators**

* **Query validator** (sanity-check user input & prompts)

  ```bash
  python pipeline/query_validator.py
  ```
* **Data validator** (strictly check retrieved data vs. context)

  ```bash
  python pipeline/data_validator.py
  ```

### 11. **Test Suite**

Run all regression & edge-case tests.

```bash
python pipeline/rag_test_suite.py
```

---

## 📦 Ignored in Git

These folders are `.gitignore`d (to avoid huge or sensitive files):

* `venv/` – local Python environment
* `.env` – secrets & keys
* `data/` – raw input datasets
* `output/` – processed CSV outputs/logs
* `chroma_store/` – local vector DB storage
* `pipeline_audits/` – run logs/audits

⚡ **How to restore locally:**
Run the CLI commands above (`batch_runner.py`, `csv_to_parquet.py`, `duckdb_loader.py`, `vector_ingest.py`) to regenerate `output/`, `parquet/`, `argo.duckdb`, and `chroma_store/` from raw `data/`.

---

## 🧪 Testing

* Validate **each stage** with its respective validator script (`validate_parquet`, `validate_duckdb`, `validate_chroma`).
* Run **end-to-end check** with `rag_test_suite.py`.

---

## By Team Breaking Code 💥
@ Surendhiran – Argo Pipeline & RAG