"""Evaluation script for RAG agent with detailed inference logging."""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

# Load environment for direct agent usage
load_dotenv()


async def evaluate_with_api(
    dataset: str = "dev",
    api_url: str = "http://localhost:8000",
    save_details: bool = False,
) -> None:
    """
    Evaluate the RAG agent on a dataset using API.

    Args:
        dataset: Dataset to evaluate on ('dev' or 'test')
        api_url: URL of the inference server
        save_details: Whether to save detailed inference results
    """
    # Load dataset
    root = Path(__file__).parent.parent.parent
    data_path = root / "data" / f"{dataset}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {dataset} set")

    # Evaluate
    correct = 0
    total = len(df)
    results = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for idx, row in df.iterrows():
            # Format query
            question = row["question"]
            query = f"{question}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"

            # Get prediction
            try:
                response = await client.post(
                    f"{api_url}/predict", json={"query": query}
                )
                response.raise_for_status()
                prediction = response.json()["answer"]

                # Check if correct
                correct_answer_idx = int(row["answer"])
                correct_answer = ["A", "B", "C", "D"][correct_answer_idx - 1]

                is_correct = prediction == correct_answer
                if is_correct:
                    correct += 1

                if save_details:
                    results.append(
                        {
                            "idx": int(idx),
                            "question": question,
                            "prediction": prediction,
                            "correct_answer": correct_answer,
                            "is_correct": is_correct,
                        }
                    )

                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{total} samples")

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

    # Calculate accuracy
    accuracy = correct / total
    print(f"\n{'=' * 50}")
    print(f"Dataset: {dataset}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"{'=' * 50}")

    # Save results if requested
    if save_details and results:
        output_dir = root / "evaluation_results"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"eval_{dataset}_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"dataset": dataset, "accuracy": accuracy, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n✓ Saved evaluation results to: {output_path}")


async def evaluate_with_details(dataset: str = "dev") -> None:
    """
    Evaluate with detailed inference logging (retrieval + prompt).

    Args:
        dataset: Dataset to evaluate on ('dev' or 'test')
    """
    from src.agent.prompts import SYSTEM_PROMPT, create_rag_prompt
    from src.agent.rag_agent import RAGAgent

    # Load dataset
    root = Path(__file__).parent.parent.parent
    data_path = root / "data" / f"{dataset}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {dataset} set")
    print("Initializing RAG Agent...")

    # Initialize agent
    agent = RAGAgent()

    # Evaluate
    correct = 0
    total = len(df)
    detailed_results = []

    for idx, row in df.iterrows():
        # Format query
        question = row["question"]
        query = (
            f"{question}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
        )

        try:
            # Get retrieved documents
            retrieved = None
            prompt = None

            if agent.use_rag and agent.retriever:
                retrieved = await agent.retriever.retrieve(query=query, top_k=5)

                if retrieved:
                    # Create prompt
                    prompt = create_rag_prompt(query, retrieved)
                else:
                    from src.agent.prompts import create_zero_shot_prompt

                    prompt = create_zero_shot_prompt(query)
                    retrieved = []
            else:
                from src.agent.prompts import create_zero_shot_prompt

                prompt = create_zero_shot_prompt(query)
                retrieved = []

            # Get prediction
            prediction = await agent.predict(query)

            # Check if correct
            correct_answer_idx = int(row["answer"])
            correct_answer = ["A", "B", "C", "D"][correct_answer_idx - 1]

            is_correct = prediction == correct_answer
            if is_correct:
                correct += 1

            # Store detailed result
            result = {
                "idx": int(idx),
                "question": question,
                "options": {
                    "A": str(row["A"]),
                    "B": str(row["B"]),
                    "C": str(row["C"]),
                    "D": str(row["D"]),
                },
                "prediction": prediction,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "category": str(row["Category"]),
                "human_accuracy": float(row["Human Accuracy"]),
                "retrieved_documents": [
                    {
                        "question": doc["question"],
                        "answer": doc["answer"],
                        "similarity": doc.get("similarity", None),
                        "hybrid_score": doc.get("hybrid_score", None),
                        "semantic_score": doc.get("semantic_score", None),
                        "bm25_score": doc.get("bm25_score", None),
                    }
                    for doc in (retrieved or [])
                ],
                "prompt": {
                    "system": SYSTEM_PROMPT,
                    "user": prompt,
                },
            }

            detailed_results.append(result)

            if (idx + 1) % 10 == 0:
                current_accuracy = correct / (idx + 1)
                print(
                    f"Processed {idx + 1}/{total} samples (Accuracy: {current_accuracy:.4f})"
                )

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Calculate accuracy
    accuracy = correct / total
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"{'=' * 60}")

    # Save detailed results
    output_dir = root / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"detailed_eval_{dataset}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "total_samples": total,
                "correct": correct,
                "accuracy": accuracy,
                "timestamp": timestamp,
                "model": agent.model,
                "use_rag": agent.use_rag,
                "results": detailed_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n✓ Saved detailed evaluation results to: {output_path}")

    # Save human-readable summary
    summary_path = output_dir / f"summary_{dataset}_{timestamp}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Evaluation Summary\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {agent.model}\n")
        f.write(f"Mode: {'RAG' if agent.use_rag else 'Zero-shot'}\n\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")

        f.write("Sample Results (first 5):\n")
        f.write(f"{'=' * 60}\n\n")

        for i, result in enumerate(detailed_results[:5], 1):
            f.write(f"Sample {i}:\n")
            f.write(f"Question: {result['question']}\n")
            f.write(
                f"Prediction: {result['prediction']} | Correct: {result['correct_answer']}\n"
            )
            f.write(f"Status: {'✓ CORRECT' if result['is_correct'] else '✗ WRONG'}\n")
            f.write("\nRetrieved Documents:\n")
            for j, doc in enumerate(result["retrieved_documents"], 1):
                sim = doc.get("similarity")
                sim_str = f" (sim: {sim:.3f})" if sim else ""
                f.write(f"  {j}. {doc['question'][:100]}...{sim_str}\n")
                f.write(f"     Answer: {doc['answer']}\n")
            f.write(f"\n{'-' * 60}\n\n")

    print(f"✓ Saved human-readable summary to: {summary_path}")


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate RAG agent")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dev",
        choices=["dev", "train", "test"],
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API URL (for API mode)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed evaluation with retrieval and prompt logging",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save evaluation details (for API mode)",
    )

    args = parser.parse_args()

    if args.detailed:
        print("\n" + "=" * 60)
        print("Running DETAILED evaluation (with retrieval + prompt logging)")
        print("=" * 60 + "\n")
        asyncio.run(evaluate_with_details(args.dataset))
    else:
        print("\n" + "=" * 60)
        print("Running standard API evaluation")
        print("=" * 60 + "\n")
        asyncio.run(evaluate_with_api(args.dataset, args.api_url, args.save_details))


if __name__ == "__main__":
    main()
