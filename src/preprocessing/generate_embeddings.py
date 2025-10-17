"""Generate embeddings for vector store."""

import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


def generate_embeddings() -> None:
    """Generate embeddings and store in ChromaDB vector store."""
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Get project root
    root = Path(__file__).parent.parent.parent

    # Load documents
    documents_path = root / "data" / "processed" / "documents.json"
    if not documents_path.exists():
        raise FileNotFoundError(
            f"Documents not found at {documents_path}. "
            "Run process_data.py first."
        )

    print("Loading documents...")
    with open(documents_path, encoding="utf-8") as f:
        documents = json.load(f)
    print(f"Loaded {len(documents)} documents")

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    embedding_model = "text-embedding-3-small"

    # Initialize ChromaDB
    vector_store_path = root / "vector_store" / "chroma"
    vector_store_path.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(vector_store_path))

    # Delete existing collection if exists
    try:
        chroma_client.delete_collection(name="legal_questions")
        print("Deleted existing collection")
    except Exception:
        pass

    # Create new collection
    collection = chroma_client.create_collection(
        name="legal_questions",
        metadata={"description": "KMMLU Legal Questions", "model": embedding_model},
    )
    print(f"Created new collection: legal_questions")

    # Generate embeddings in batches
    batch_size = 100
    total_batches = (len(documents) + batch_size - 1) // batch_size

    print(f"\nGenerating embeddings using {embedding_model}...")
    for i in tqdm(range(0, len(documents), batch_size), total=total_batches):
        batch = documents[i : i + batch_size]

        # Extract texts
        texts = [doc["text"] for doc in batch]

        # Generate embeddings
        response = client.embeddings.create(model=embedding_model, input=texts)

        # Prepare data for ChromaDB
        embeddings = [e.embedding for e in response.data]
        ids = [doc["id"] for doc in batch]
        metadatas = [
            {
                "question": doc["question"][:500],  # Truncate for metadata
                "answer": doc["answer_letter"],
                "category": doc["category"],
                "difficulty": doc["difficulty"],
            }
            for doc in batch
        ]

        # Store in vector database
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"\n✓ Stored {len(documents)} embeddings in ChromaDB")
    print(f"✓ Vector store location: {vector_store_path}")
    print("Embedding generation completed")


if __name__ == "__main__":
    generate_embeddings()

