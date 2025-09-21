### Welcome to the Argo Float Data Ingestion & RAG Pipeline ⛵️

This document serves as a complete, **developer-grade guide** to the entire data processing and RAG pipeline I’ve implemented, from raw NetCDF files to a production-ready, chat-enabled visualization system. It's designed to be a definitive reference for anyone looking to understand or validate this project.

-----

### 1\. High-Level Overview (TL;DR)

I've built a robust, production-quality ingestion, indexing, and RAG (Retrieval-Augmented Generation) pipeline for Argo float data.

The process flows like this:

  * **Ingestion:** Raw Argo **NetCDF** files (`.nc`) are batch-processed and converted into daily **CSVs**.
  * **Consolidation:** These CSVs are then consolidated, normalized, and converted into highly-efficient, per-month **Parquet** files.
  * **Indexing:** The Parquet data is loaded into two separate data stores for optimal performance:
      * A **DuckDB** SQL database for structured, numeric data.
      * A **Chroma** vector store for semantic metadata embeddings.
  * **RAG Pipeline:** The core RAG pipeline (`rag_pipeline.py`) orchestrates a multi-step process:
    1.  A user's natural language query is parsed.
    2.  **Semantic retrieval** is performed on the Chroma vector store.
    3.  Candidate profile IDs are extracted.
    4.  **Numeric data is fetched** from the DuckDB database.
    5.  A **deterministic canvas** of visual suggestions (e.g., charts, maps) is built.
    6.  An **LLM call** to Azure OpenAI generates a human-readable summary.
    7.  The final output is a structured JSON response, including an audit trail.

I have also integrated a series of validators and test suites (`query_validator.py`, `data_validator.py`, `rag_test_suite.py`) to ensure the pipeline's correctness and reproducibility.

The key outputs of this pipeline are: `argo.duckdb` (the structured database), `chroma_store/` (the persistent vector store), `monthly parquet/` files, and per-query JSON outputs and audit files.

-----

### 2\. Project File Inventory

This section provides a detailed breakdown of the project's folder and file structure, mirroring the format of a standard `README.md`.

#### Top-level folders

  * `data/`: Stores the original, raw `.nc` files.
  * `output/`: Contains the per-day CSV extraction outputs.
  * `parquet/`: Stores the per-month, per-layer Parquet outputs created by `csv_to_parquet.py`.
  * `pipeline/`: Contains all of the core pipeline scripts.
  * `chroma_store/`: The persistent on-disk Chroma vector store.
  * `argo.duckdb`: The DuckDB database file.
  * `pipeline_audits/` & `pipeline_outputs/`: Store audit and RAG JSON outputs for each run.

#### Important `pipeline/` files and their purpose

  * `year_runner.py`: The main driver script that orchestrates `batch_runner.py` for each month in a given year, used for large-scale ingestion.
  * `batch_runner.py`: Processes all `.nc` files in a folder by calling per-file extractors.
  * `extract_*.py` (`extract_metadata.py`, `extract_location.py`, `extract_core.py`, `extract_qc.py`, `extract_history.py`): These scripts contain the per-file extraction logic, writing CSV layer outputs to `output/<year>/<month>/<file>/`.
  * `utils.py`: A collection of helper utilities, including functions for atomic writes, date parsing, and type conversions.
  * `csv_to_parquet.py`: This script handles the conversion of daily CSVs to per-month, per-layer Parquet files, with configurable parallel processing and data type normalization.
  * `validate_parquet.py`: Verifies row counts and basic consistency between the original CSVs and the newly created Parquet files.
  * `duckdb_loader.py`: Creates `argo.duckdb` and loads the Parquet files into per-layer tables (e.g., `core_measurements_YYYY_MM`). It also sets up indexes and virtual views.
  * `validate_duckdb.py`: Conducts schema and row count checks on the DuckDB tables, comparing them to the Parquet manifests.
  * `vector_ingest.py`: Flattens metadata rows into documents, generates embeddings, and ingests them into the Chroma `argo_metadata` collection.
  * `validate_chroma.py`: A sanity check script for the Chroma collection, verifying counts and sampling documents.
  * `query_validator.py`: Parses natural language queries into a structured format (intent, variables, bbox, dates) to guard against downstream SQL generation errors.
  * `data_validator.py`: Performs exploratory data checks like row counts and time range validation.
  * `rag_pipeline.py`: The central orchestration file for a RAG run, exposing the `run_rag_query(...)` function. It contains modular functions for each pipeline step, from semantic retrieval to calling the LLM.
  * `canvas_builder.py`: A deterministic builder that creates frontend-friendly JSON visuals (tables, charts, maps) directly from numeric and metadata dataframes **without relying on the LLM**, preventing hallucination.
  * `rag_test_suite.py`: An end-to-end test suite with "hero queries" that produce audit JSONs to validate the pipeline's behavior.
  * `rag_guard.py`: Contains business rules for the LLM, such as the `INSUFFICIENT_CONTEXT` behavior and strict prompt enforcement.
  * `requirements.txt`: Pinned dependencies for the Python virtual environment.

-----

### 3\. Complete Data Flow (Step-by-Step)

Here’s a detailed, step-by-step breakdown of the entire data pipeline.

#### A. Ingestion & Local Processing

  * **Source:** Raw `.nc` files are pulled from `ftp.ifremer.org`.
  * **Execution:** `batch_runner.py` (invoked by `year_runner.py`) runs the `extract_*` scripts for each `.nc` file.
      * `extract_metadata.py` writes `argo_metadata_full.csv` & `argo_metadata_clean.csv`.
      * `extract_location.py` writes `argo_time_location.csv`.
      * `extract_core.py` writes `argo_core_measurements.csv.gz`.
      * `extract_qc.py` writes various QC-related CSVs.
      * `extract_history.py` writes `argo_history.csv`.
  * **Output:** Each CSV file is saved to `output/<year>/<month>/<YYYYMMDD_prof>/`.

#### B. Flattening CSV → Parquet

  * **Execution:** `csv_to_parquet.py` processes data on a per-month basis.
  * **Process:** It reads the CSVs in chunks, normalizes data types (e.g., converting datetimes to UTC timestamps, floats to `Int64`), and then writes the data as per-layer Parquet files.
  * **Output:** Files are written to `parquet/<year>/<month>/<layer>.parquet`.
  * **Why Parquet?** I chose Parquet because it's a columnar, compressed format that offers excellent I/O performance for query engines like DuckDB. The per-month partitioning allows for fast partition pruning, balancing file size and reducing the "too many tiny files" problem.

#### C. Structured SQL Database (DuckDB)

  * **Execution:** `duckdb_loader.py` loads the Parquet files into `argo.duckdb`.
  * **Schema:** It creates per-layer tables with names like `core_measurements_YYYY_MM` and ensures column types are appropriate (e.g., `TIMESTAMP WITH TIME ZONE`).
  * **Why DuckDB?** It's a fast, local analytical SQL engine with zero operational overhead, making it perfect for prototyping and PoCs. Its ability to read Parquet files efficiently and perform complex aggregations is a huge advantage.

#### D. Semantic Indexing (Vector DB)

  * **Execution:** `vector_ingest.py` takes the `metadata_clean` data.
  * **Process:** It flattens the metadata rows into human-readable strings, sanitizes them, and then creates embeddings using my chosen embedding model.
  * **Storage:** These embeddings and their associated metadata are ingested into a persistent Chroma collection at `chroma_store/`.
  * **Why Chroma?** Chroma is a lightweight, local, and persistent vector store that's ideal for development. Its ease of use and good ergonomics make it a great choice for this PoC, and it can be easily swapped out for a more scalable solution like Weaviate or Milvus later.

#### E. RAG Pipeline (Query → Answer)

The `rag_pipeline.py` script orchestrates the core end-to-end flow:

1.  **Environment Check:** `assert_env()` ensures API keys are present.
2.  **Initialization:** `init_chroma()` and `init_embed_model()` set up the data stores and models.
3.  **Semantic Retrieval:** The query is embedded, and a search is performed on Chroma for the `top_k` nearest documents.
4.  **ID Extraction:** `build_platform_profile_set()` extracts `PLATFORM_NUMBER` and `profile_index` from the retrieved metadata.
5.  **Numeric Data Fetch:** `fetch_measurements_for_candidates()` and `fetch_metadata_for_candidates()` safely query the DuckDB tables for the identified candidates.
6.  **Context Assembly:** `assemble_context()` builds a compact context for the LLM, including top documents, sample numeric rows, and key numeric aggregates (min/max/mean).
7.  **Deterministic Canvas:** `decide_and_build()` (from `canvas_builder.py`) creates visualization suggestions and JSON payloads (tables, charts, maps) directly from the numeric data, completely bypassing the LLM to prevent hallucination.
8.  **LLM Call:** `call_azure_openai_chat()` sends the assembled context to Azure OpenAI. My system prompt strictly enforces rules, like returning `INSUFFICIENT_CONTEXT` if facts aren't present and providing provenance tags.
9.  **Final Output:** The pipeline returns a comprehensive JSON object containing the status, the generated answer, the visual canvas data, and an audit trail.

**Important Guards:**

  * **Query Validator:** Ensures structured parsing of the user query.
  * **Context Capping:** `assemble_context()` caps numeric rows and provides aggregates to ground the LLM's response and avoid token limits.
  * **Strict System Prompt:** The LLM is forced to respect the `INSUFFICIENT_CONTEXT` rule, which has been rigorously tested.

-----

### 4\. Data Contracts & Schema Mapping

This section outlines where specific data lives and why, detailing the schemas for each component.

#### Parquet / DuckDB Tables (Structured Data)

  * `core_measurements_*`: Stores numeric, per-profile, per-level data (`platform_number`, `juld`, `PRES`, `TEMP`, `PSAL`, etc.).
  * `qc_per_level_*`: Contains level-granular QC and adjusted values.
  * `metadata_clean_*` & `metadata_full_*`: Rich metadata per profile (`PLATFORM_NUMBER`, `PROJECT_NAME`, `LATITUDE`, `LONGITUDE`, etc.).
  * `time_location_*`: An alias table for quick location-based queries.

#### Vector DB Documents

  * `document`: A flattened string of platform or profile-level metadata used for semantic search.
  * `metadata`: A dictionary with filterable fields like `PLATFORM_NUMBER`, `PI_NAME`, and `PROJECT_NAME`.

#### RAG Output JSON Contract

This is the exact JSON format the frontend should expect:

  * `status`: `ok`, `error`, `insufficient_context`, etc.
  * `query`: The original user query.
  * `retrieved_count`, `candidates_count`, `numeric_rows`, `metadata_rows`: Counts for debugging.
  * `answer`: The human-readable text from the LLM.
  * `canvas`: The deterministic visualization data.
  * `numeric_sample_csv`: A URL to a CSV file for large datasets.
  * `audit`: Optional debug information.

The `canvas` object has a key `visuals`, which is a list of objects, each with a schema version, type, title, and data for rendering charts, maps, or tables.

-----

### 5\. Why I Chose This Tech Stack 🧐

  * **NetCDF → Python (`xarray`/`netCDF4`):** The industry standard for oceanographic data, these libraries safely handle complex data structures and decoding.
  * **Per-month Parquet (`pyarrow`):** Provides a columnar, compressed, and highly efficient format for analytical scans, compatible with a wide range of data engines.
  * **DuckDB:** A zero-admin, local OLAP engine that offers exceptional performance for complex SQL queries on Parquet files, making it perfect for rapid prototyping.
  * **Chroma:** A lightweight, persistent vector database that's easy to run locally for development and can be easily swapped out for a more robust production solution.
  * **Azure OpenAI:** A powerful and enterprise-ready LLM service for natural language understanding and generation, providing critical features like key management and security.
  * **Canvas Builder:** The key to avoiding hallucination. This deterministic component ensures visual payloads are reproducible and grounded in the data, not the LLM's imagination.
  * **DuckDB + Chroma Split:** This two-store architecture is crucial. I use Chroma for semantic discovery of relevant IDs and DuckDB for the precise, structured numeric truth, preventing the need to store massive numeric arrays in vectors.

-----

### 6\. Validation Checklist ✅

After each ingestion step, I run specific validation scripts to ensure data integrity.

  * **After Ingestion:**
      * `python pipeline/csv_to_parquet.py -y 2025 ...`
      * Check for the existence of Parquet files and their manifest JSONs.
  * **Parquet Validation:**
      * `python pipeline/validate_parquet.py -y 2025 ...`
      * Confirm that CSV row counts match Parquet row counts.
  * **DuckDB Load & Validate:**
      * `python pipeline/duckdb_loader.py -y 2025 ...`
      * `python pipeline/validate_duckdb.py -y 2025 ...`
      * Use `SHOW TABLES` and `SELECT COUNT(*)` in DuckDB to verify tables exist and have the correct number of rows and schemas.
  * **Vector DB Ingestion & Validate:**
      * `python pipeline/vector_ingest.py -y 2025 ...`
      * `python pipeline/validate_chroma.py`
      * Confirm the Chroma store exists, the collection has the expected count, and a sample of documents contains the correct metadata (`PLATFORM_NUMBER`, `PI_NAME`).
  * **RAG Smoke Tests:**
      * `python pipeline/rag_test_suite.py`
      * Run manual queries (`python rag_pipeline.py -q "..."`) and validate that the output JSON (`pipeline_outputs/*.json`) is correctly structured and contains accurate data.

-----

### 7\. Known Limitations & Mitigations ⚠️

  * **Hallucination:**
      * **Risk:** The LLM could invent numbers or facts.
      * **Mitigation:** I use a strict system prompt that requires `INSUFFICIENT_CONTEXT` if the information isn't present. I also explicitly supply numeric aggregates and provenance tags to ground the LLM's response in factual data.
  * **Top\_k / Coverage Tradeoff:**
      * **Risk:** The `top_k` semantic retrieval might miss relevant documents, or a very large `top_k` could be too costly.
      * **Mitigation:** I use a sensible default `top_k` (490) and employ a staged retrieval approach (Chroma → ID extraction → DuckDB fetch).
  * **Token Budget / LLM Context Size:**
      * **Risk:** Passing too many numeric rows could exceed the LLM's token limit.
      * **Mitigation:** I cap the number of rows passed to the LLM to 20 for context and provide aggregates for the rest. Larger datasets are made available via `numeric_sample_csv` for the frontend.
  * **Data Freshness:**
      * **Risk:** The FTP source may update historical files.
      * **Mitigation:** The pipeline is designed to be re-run periodically. For a production environment, I would version the Parquet files or use immutable paths in object storage.
  * **Scalability:**
      * **Risk:** Chroma is not designed for massive scale.
      * **Mitigation:** The architecture is modular, so Chroma can be easily replaced with a more scalable solution like Weaviate or Milvus.
  * **Edge Cases:**
      * **Risk:** Missing data (e.g., `lon: null`).
      * **Mitigation:** I handle these cases by filtering incomplete points or marking them in the response so the frontend can display a warning to the user.

-----

### 8\. How the RAG Pipeline Ensures Accuracy

This pipeline is designed with a clear separation of concerns to avoid mistakes and ensure explainability.

  * **DuckDB as the Numeric Truth:** The SQL queries on DuckDB provide deterministic, precise numeric data and aggregates, with no guesswork from the LLM.
  * **Chroma for Semantic Candidates Only:** Chroma's sole purpose is to identify relevant `PLATFORM_NUMBER` or `profile_index` IDs based on text similarity. It is never used to retrieve or assert numeric facts.
  * **LLM for Composition:** The LLM's role is to compose natural language explanations and summaries. All numeric claims it makes are forced to be sourced from the `assemble_context()` aggregates or the sample CSV data, enforced by the system prompt and provenance tags.
  * **Guarded Responses:** The pipeline is designed to return `INSUFFICIENT_CONTEXT` if the required information is not available, which prevents the LLM from hallucinating answers.
  * **Audited and Test-Driven:** The `query_validator.py`, `data_validator.py`, and `rag_test_suite.py` scripts constantly exercise these boundaries, proving the pipeline's robustness and test-driven design.

-----

### 9\. Connecting to the Frontend

I recommend building a small API server using **FastAPI** to serve as the glue between the frontend and the RAG pipeline. FastAPI is a great choice because of its asynchronous support and ease of deployment.

#### Suggested Server Endpoints

  * `POST /api/query`: Accepts a user query and returns a job ID. The server will run `run_rag_query(...)` as a background task.
  * `GET /api/query/{job_id}`: Allows the frontend to poll for the job's status and retrieve the final result once it's complete.
  * `GET /static/sample_csvs/{file}`: Serves the larger sample CSVs, allowing the frontend to safely download and visualize datasets that are too big for the LLM's context.

The RAG result JSON is the core contract. The frontend should use the `answer` for the human-readable summary and the `canvas.visuals` list to deterministically render tables, charts, and maps.

-----

### 10\. Next Steps Checklist ✅

Here is a practical checklist of what I'll do next to prepare the project for a demo or handover.

  * **Fix `canvas/structured` mismatch:** I'll patch `canvas_builder.py` to ensure that if a visualization is unavailable, no incomplete structured JSON is produced.
  * **Add Provenance to Canvas Payloads:** I'll include a `provenance` field in each canvas visual, detailing the source (e.g., `DuckDB core_measurements_2025_01`) and the number of rows used.
  * **Create the Server Wrapper:** I'll build `server/api.py` using FastAPI to handle the requests and run the RAG pipeline in the background. I'll provide the exact `uvicorn` command for running it.
  * **Update the README:** I'll add a new section to the `README.md` with detailed instructions on how to set up the environment, run the ingestion process, and start the server.
  * **Add CI Checks:** I'll add basic unit tests for the `canvas_builder` and `assemble_context` functions, as well as an end-to-end test.
  * **Prepare Demo Slides:** I will prepare a presentation to explain the architecture, show a hero query in action, and demonstrate the rendered visual outputs to showcase the project's impact.

-----

### 11\. Presentation-Ready Bullet Points

This is a concise summary for a viva or demo, highlighting the key achievements and innovations of the project.

  * **Problem:** The Argo oceanographic dataset is massive, complex, and difficult for non-experts to query.
  * **Solution:** I've built **FloatChat**, a unified pipeline that transforms raw NetCDF data into a RAG system capable of interactive, canvas-ready visualizations.
  * **Architecture Rationale:**
      * **Parquet + DuckDB:** Provides fast, precise, and auditable numeric truth for analytics.
      * **Chroma Vectors:** Enables flexible, semantic understanding of natural language queries.
      * **LLM:** Used for natural language composition and explanation, but is always grounded in deterministic data.
  * **Innovation:**
      * **Canvas-based Chat:** My deterministic canvas builder produces auditable, hallucination-free visuals, which are then complemented by the LLM's text. This two-part system is a huge innovation for building trustworthy applications.
      * **Two-Layer Separation:** I've cleanly separated semantic discovery (Chroma) from numeric truth retrieval (DuckDB).
  * **Feasibility & Risks:** This architecture is feasible for a PoC and departmental-level deployment. I've addressed key risks like hallucination and security with strict prompts, audits, and a robust design.
  * **Impact:** This pipeline lowers the barrier for scientists and decision-makers to quickly and confidently extract critical ocean metrics.

-----

### 12\. Future Files

Here are some files I recommend adding in the future to further productionize the project:

  * `server/api.py`: The FastAPI server wrapper.
  * `docker/Dockerfile` + `docker-compose.yml`: For easy one-command local deployment.
  * `ci/test_rag_end_to_end.sh`: A shell script for a quick E2E test.
  * `notes/arch_decisions.md`: A markdown file to document architectural decisions and trade-offs.

-----

### 13\. Commands Cheat Sheet (Copy/Paste)

This is a quick-reference guide for common commands.

```bash
# 1. Environment setup
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r pipeline/requirements.txt

# 2. Run ingestion for one month (example)
cd pipeline
python batch_runner.py -i ../data/2025/09 -o ../output/2025/09 --log ../output/2025/09/process_log.csv

# 3. Convert to parquet for the whole year
python csv_to_parquet.py -y 2025 -o ../output -p ../parquet --parallel

# 4. Validate parquet
python validate_parquet.py -y 2025 -o ../output -p ../parquet

# 5. Load to DuckDB
python duckdb_loader.py -y 2025 -p ../parquet -d ../argo.duckdb

# 6. Validate DuckDB
python validate_duckdb.py -y 2025 -d ../argo.duckdb

# 7. Ingest to Chroma
python vector_ingest.py -y 2025 -p ../parquet -s ../chroma_store

# 8. Validate Chroma
python validate_chroma.py

# 9. Run a RAG test
python rag_test_suite.py

# 10. Run a single ad-hoc RAG query (for production/demo)
python rag_pipeline.py -q "What is the minimum pressure recorded by floats in 2025?" -y 2025 --out ./pipeline_outputs/min_pressure_2025.json

# 11. Run the server (after adding server/api.py)
# uvicorn server.api:app --reload --host 0.0.0.0 --port 8080
```
## By Team Breaking Code 💥
@ Surendhiran – Argo Pipeline & RAG