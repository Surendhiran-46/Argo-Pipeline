import argparse
from pathlib import Path
import pandas as pd
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ----------------- Config -----------------
CHROMA_DIR = "../chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_and_store(year: str, month: str, parquet_root: Path, client: PersistentClient, model):
    month_dir = parquet_root / year / month
    if not month_dir.exists():
        print(f"SKIP: {month_dir} not found")
        return

    targets = ["metadata_clean.parquet", "time_location.parquet"]
    docs, metadatas, ids = [], [], []

    for t in targets:
        f = month_dir / t
        if not f.exists():
            continue
        df = pd.read_parquet(f)

        for i, row in df.iterrows():
            # Build free-text document for semantic search
            text = " ".join([f"{col}:{row[col]}" for col in df.columns if pd.notna(row[col])])
            
            # Structured metadata dict (for filtering)
            def sanitize_metadata(value):
                if pd.isna(value):
                    return None
                if isinstance(value, (pd.Timestamp, )):
                    return value.isoformat()   # ✅ store datetime as string
                if isinstance(value, (int, float, bool, str)):
                    return value
                return str(value)  # Fallback: cast unknowns to string
            
            metadata = {col: sanitize_metadata(row[col]) for col in df.columns}

            docs.append(text)
            metadatas.append(metadata)
            ids.append(f"{t}-{year}-{month}-{i}")

    if not docs:
        print(f"SKIP: No documents for {year}-{month}")
        return

    collection = client.get_or_create_collection(name="argo_metadata")
    embeddings = model.encode(docs, show_progress_bar=True).tolist()

    collection.upsert(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas   # ✅ FIX: store structured fields
    )
    print(f"✔ Ingested {len(docs)} docs for {year}-{month}")


def main(year: str, parquet_root: Path):
    client = PersistentClient(path=CHROMA_DIR)
    model = SentenceTransformer(EMBED_MODEL)

    months = [f"{m:02d}" for m in range(1, 13)]
    for m in months:
        embed_and_store(year, m, parquet_root, client, model)

    print(f"\n=== Vector ingestion complete for {year} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest ARGO metadata into ChromaDB with embeddings.")
    parser.add_argument("-y", "--year", required=True, help="Year to process (e.g. 2025)")
    parser.add_argument("-p", "--parquet", default="../parquet", help="Root parquet folder (default: ../parquet)")
    args = parser.parse_args()

    main(args.year, Path(args.parquet))
