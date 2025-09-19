from chromadb import PersistentClient

def validate_chroma(path="chroma_store", collection_name="argo_metadata"):
    client = PersistentClient(path=path)
    col = client.get_collection(collection_name)

    print(f"Collection: {collection_name}")
    print(f"Total embeddings: {col.count()}")

    # Peek at 3 docs
    docs = col.get(limit=5)
    for i, doc in enumerate(docs["documents"]):
        print(f"\nSample {i+1}:")
        print(f"Document: {doc[:200]}...")  # preview first 200 chars
        print(f"Metadata: {docs['metadatas'][i]}")

if __name__ == "__main__":
    validate_chroma()