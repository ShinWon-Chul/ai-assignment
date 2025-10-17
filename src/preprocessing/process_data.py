"""Data processing script."""

import json
from pathlib import Path

import pandas as pd


def create_documents() -> list[dict[str, str | int | float | dict[str, str]]]:
    """
    Convert train.csv into structured documents for RAG.

    Returns:
        List of document dictionaries
    """
    # Get project root
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"

    # Load training data
    train_df = pd.read_csv(data_dir / "train_extended.csv")
    print(f"Loaded {len(train_df)} training samples")

    documents = []
    for idx, row in train_df.iterrows():
        # Each question becomes a document
        answer_idx = int(row["answer"]) - 1  # Convert 1-4 to 0-3
        answer_letter = ["A", "B", "C", "D"][answer_idx]

        doc = {
            "id": f"train_{idx}",
            "question": str(row["question"]),
            "options": {
                "A": str(row["A"]),
                "B": str(row["B"]),
                "C": str(row["C"]),
                "D": str(row["D"]),
            },
            "answer": int(row["answer"]),  # 1, 2, 3, 4
            "answer_letter": answer_letter,
            "category": str(row["Category"]),
            "difficulty": float(1 - row["Human Accuracy"]),  # Higher = harder
            # Search text: question + all options + answer
            "text": (
                f"{row['question']}\n"
                f"A. {row['A']}\n"
                f"B. {row['B']}\n"
                f"C. {row['C']}\n"
                f"D. {row['D']}\n"
                f"정답: {answer_letter}"
            ),
        }
        documents.append(doc)

    return documents


def process_data() -> None:
    """Process raw data and prepare for RAG system."""
    root = Path(__file__).parent.parent.parent

    # Create documents
    documents = create_documents()

    # Create processed directory
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(exist_ok=True)

    # Save documents
    output_path = processed_dir / "documents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"✓ Created {len(documents)} documents")
    print(f"✓ Saved to {output_path}")
    print("Data processing completed")


if __name__ == "__main__":
    process_data()
