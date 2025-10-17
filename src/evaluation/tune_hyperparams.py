"""Hyperparameter tuning for RAG Agent (top_k and BM25 weight)."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.agent.prompts import SYSTEM_PROMPT, create_rag_prompt
from src.agent.rag_agent import RAGAgent
from src.agent.retriever import QuestionRetriever

# Load environment variables
load_dotenv()


async def evaluate_single_sample(
    agent: RAGAgent,
    idx: int,
    row: pd.Series,
    top_k: int,
) -> dict[str, Any]:
    """
    Evaluate a single sample with the RAG agent.

    Args:
        agent: RAG agent instance
        idx: Sample index
        row: Data row
        top_k: Number of documents to retrieve

    Returns:
        Result dictionary for this sample
    """
    try:
        # Build question text
        question = str(row["question"])
        options_text = f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
        question_text = f"{question}\n{options_text}"

        # Get prediction
        if agent.use_rag and agent.retriever:
            retrieved = await agent.retriever.retrieve(query=question_text, top_k=top_k)
            prompt = create_rag_prompt(question_text, retrieved)

            response = await agent.client.chat.completions.create(
                model=agent.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )

            answer_text = response.choices[0].message.content
            prediction = agent._extract_answer(answer_text) if answer_text else None

            if not prediction:
                prediction = retrieved[0]["answer"] if retrieved else "A"
        else:
            prediction = await agent.predict(question_text)

        # Check correctness
        correct_answer = ["A", "B", "C", "D"][int(row["answer"]) - 1]
        is_correct = prediction == correct_answer

        # Store result
        result = {
            "idx": int(idx),
            "question": question,
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
                for doc in (retrieved if agent.use_rag else [])
            ],
        }

        return result

    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        # Return error result
        return {
            "idx": int(idx),
            "question": str(row.get("question", "")),
            "prediction": "A",
            "correct_answer": ["A", "B", "C", "D"][int(row["answer"]) - 1],
            "is_correct": False,
            "category": str(row.get("Category", "")),
            "human_accuracy": float(row.get("Human Accuracy", 0.0)),
            "retrieved_documents": [],
            "error": str(e),
        }


async def evaluate_with_params(
    df: pd.DataFrame,
    top_k: int,
    bm25_weight: float,
    output_dir: Path,
    experiment_id: str,
    retriever: QuestionRetriever,
) -> dict[str, Any]:
    """
    Evaluate RAG agent with specific hyperparameters (with parallel LLM calls).

    Args:
        df: Evaluation dataset
        top_k: Number of documents to retrieve
        bm25_weight: Weight for BM25 search
        output_dir: Directory to save results
        experiment_id: Unique experiment identifier
        retriever: Pre-initialized retriever (shared across experiments)

    Returns:
        Evaluation results dictionary
    """
    semantic_weight = 1.0 - bm25_weight

    print(f"\n{'=' * 80}")
    print(f"Experiment: {experiment_id}")
    print(
        f"Parameters: top_k={top_k}, bm25_weight={bm25_weight:.2f}, semantic_weight={semantic_weight:.2f}"
    )
    print(f"{'=' * 80}")

    # Initialize agent with pre-built retriever (no BM25 rebuild!)
    agent = RAGAgent(
        top_k=top_k,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        use_hybrid=True,
        retriever=retriever,
    )

    total = len(df)

    # Prepare tasks for parallel execution
    print(f"Preparing {total} evaluation tasks...")
    tasks = []
    for idx, row in df.iterrows():
        task = evaluate_single_sample(agent, idx, row, top_k)
        tasks.append(task)

    # Execute all tasks in parallel
    print(f"Evaluating {total} samples in parallel...")
    detailed_results = await asyncio.gather(*tasks)

    # Sort results by idx to maintain original order
    detailed_results = sorted(detailed_results, key=lambda x: x["idx"])

    # Calculate metrics
    correct = sum(1 for r in detailed_results if r["is_correct"])
    accuracy = correct / total

    print(f"\n{'=' * 80}")
    print(f"Results for {experiment_id}")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.4f}")
    print(f"{'=' * 80}\n")

    # Save detailed results (JSON)
    result_data = {
        "experiment_id": experiment_id,
        "parameters": {
            "top_k": top_k,
            "bm25_weight": bm25_weight,
            "semantic_weight": semantic_weight,
        },
        "metrics": {
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
        },
        "timestamp": datetime.now().isoformat(),
        "model": agent.model,
        "results": detailed_results,
    }

    json_path = output_dir / f"{experiment_id}_detailed.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved detailed results to {json_path}")

    # Save summary (TXT)
    txt_path = output_dir / f"{experiment_id}_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"  - top_k: {top_k}\n")
        f.write(f"  - bm25_weight: {bm25_weight:.2f}\n")
        f.write(f"  - semantic_weight: {semantic_weight:.2f}\n\n")
        f.write("Results:\n")
        f.write(f"  - Total samples: {total}\n")
        f.write(f"  - Correct predictions: {correct}\n")
        f.write(f"  - Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        f.write("Category-wise Accuracy:\n")

        # Calculate per-category accuracy
        df_results = pd.DataFrame(detailed_results)
        category_stats = (
            df_results.groupby("category")
            .agg({"is_correct": ["sum", "count", "mean"]})
            .round(4)
        )

        for category in category_stats.index:
            correct_cat = int(category_stats.loc[category, ("is_correct", "sum")])
            total_cat = int(category_stats.loc[category, ("is_correct", "count")])
            acc_cat = float(category_stats.loc[category, ("is_correct", "mean")])
            f.write(f"  - {category}: {correct_cat}/{total_cat} ({acc_cat:.4f})\n")

        f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"✓ Saved summary to {txt_path}\n")

    return {
        "experiment_id": experiment_id,
        "top_k": top_k,
        "bm25_weight": bm25_weight,
        "semantic_weight": semantic_weight,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


async def run_hyperparameter_tuning(
    dataset: str = "dev",
    top_k_range: tuple[int, int, int] = (3, 11, 1),  # (start, stop, step)
    bm25_weight_range: tuple[float, float, float] = (
        0.1,
        0.55,
        0.05,
    ),  # (start, stop, step)
    start_from: int = 1,  # Start from this experiment number
) -> None:
    """
    Run hyperparameter tuning experiments.

    Args:
        dataset: Dataset name (dev or train)
        top_k_range: Range for top_k (start, stop, step)
        bm25_weight_range: Range for bm25_weight (start, stop, step)
        start_from: Start from this experiment number (default: 1)
    """
    # Setup paths
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"
    output_dir = root / "evaluation_results"
    output_dir.mkdir(exist_ok=True)

    # Create real-time progress log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_log_path = output_dir / f"tuning_progress_{timestamp}.txt"

    # Load dataset
    data_path = data_dir / f"{dataset}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {dataset}.csv")

    # Generate experiment grid
    import numpy as np

    top_k_values = list(range(top_k_range[0], top_k_range[1], top_k_range[2]))
    bm25_weight_values = list(
        np.arange(bm25_weight_range[0], bm25_weight_range[1], bm25_weight_range[2])
    )

    total_experiments = len(top_k_values) * len(bm25_weight_values)

    print(f"\n{'=' * 80}")
    print("Hyperparameter Tuning Configuration")
    print(f"{'=' * 80}")
    print(f"Dataset: {dataset}")
    print(f"Total samples: {len(df)}")
    print(f"top_k values: {top_k_values}")
    print(f"bm25_weight values: {[f'{w:.2f}' for w in bm25_weight_values]}")
    print(f"Total experiments: {total_experiments}")
    print(f"Start from experiment: {start_from}")
    print(f"Output directory: {output_dir}")
    print(f"Progress log: {progress_log_path}")
    print(f"{'=' * 80}\n")

    # Initialize retriever ONCE (BM25 built only once!)
    print("Initializing retriever (building BM25 index)...")
    retriever = QuestionRetriever(use_hybrid=True)
    print("✓ Retriever initialized and ready for all experiments!\n")

    # Initialize progress log
    with open(progress_log_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Hyperparameter Tuning Progress Log\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Total Samples: {len(df)}\n")
        f.write(f"Total Experiments: {total_experiments}\n")
        f.write(f"top_k values: {top_k_values}\n")
        f.write(f"bm25_weight values: {[f'{w:.2f}' for w in bm25_weight_values]}\n")
        f.write("=" * 80 + "\n\n")

    # Run experiments
    all_results = []
    experiment_num = 0

    for top_k in top_k_values:
        for bm25_weight in bm25_weight_values:
            experiment_num += 1

            # Skip already completed experiments
            if experiment_num < start_from:
                print(
                    f"[{experiment_num}/{total_experiments}] Skipping (already completed)"
                )
                continue

            experiment_id = f"exp_{experiment_num:03d}_topk{top_k}_bm25_{int(bm25_weight * 100):02d}"

            print(
                f"\n[{experiment_num}/{total_experiments}] Running experiment: {experiment_id}"
            )

            try:
                result = await evaluate_with_params(
                    df=df,
                    top_k=top_k,
                    bm25_weight=bm25_weight,
                    output_dir=output_dir,
                    experiment_id=experiment_id,
                    retriever=retriever,
                )
                all_results.append(result)

                # Write real-time progress to log file
                with open(progress_log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{experiment_num}/{total_experiments}] {experiment_id}\n")
                    f.write(
                        f"  top_k: {top_k}, bm25_weight: {bm25_weight:.2f}, semantic_weight: {1.0 - bm25_weight:.2f}\n"
                    )
                    f.write(
                        f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy'] * 100:.2f}%)\n"
                    )
                    f.write(f"  Correct: {result['correct']}/{result['total']}\n")
                    f.write(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("-" * 80 + "\n\n")

                print(f"✓ Progress logged to {progress_log_path}")

            except Exception as e:
                print(f"❌ Error in experiment {experiment_id}: {e}")
                import traceback

                traceback.print_exc()

                # Log error to progress file
                with open(progress_log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{experiment_num}/{total_experiments}] {experiment_id}\n")
                    f.write(f"  ❌ ERROR: {str(e)}\n")
                    f.write(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("-" * 80 + "\n\n")

                continue

    # Save overall summary
    summary_path = output_dir / "tuning_summary.csv"
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values("accuracy", ascending=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 80}")
    print("Hyperparameter Tuning Complete!")
    print(f"{'=' * 80}")
    print("\nTop 5 configurations by accuracy:")
    print(summary_df.head(5).to_string(index=False))
    print(f"\n✓ Full results saved to: {summary_path}")
    print(f"✓ Individual experiment results saved to: {output_dir}/")
    print(f"\n{'=' * 80}\n")

    # Save best configuration
    best_config = summary_df.iloc[0]
    best_config_path = output_dir / "best_config.txt"
    with open(best_config_path, "w", encoding="utf-8") as f:
        f.write("Best Hyperparameter Configuration\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Experiment ID: {best_config['experiment_id']}\n")
        f.write(f"top_k: {int(best_config['top_k'])}\n")
        f.write(f"bm25_weight: {best_config['bm25_weight']:.2f}\n")
        f.write(f"semantic_weight: {best_config['semantic_weight']:.2f}\n")
        f.write(
            f"Accuracy: {best_config['accuracy']:.4f} ({best_config['accuracy'] * 100:.2f}%)\n"
        )
        f.write(f"Correct: {int(best_config['correct'])}/{int(best_config['total'])}\n")

    print(f"✓ Best configuration saved to: {best_config_path}\n")

    # Append final summary to progress log
    with open(progress_log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("Tuning Complete!\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"Total Experiments Completed: {len(all_results)}/{total_experiments}\n\n"
        )
        f.write("Top 5 Configurations by Accuracy:\n")
        f.write("-" * 80 + "\n")
        for i, row in summary_df.head(5).iterrows():
            f.write(f"{i + 1}. {row['experiment_id']}\n")
            f.write(
                f"   top_k={int(row['top_k'])}, bm25_weight={row['bm25_weight']:.2f}\n"
            )
            f.write(
                f"   Accuracy: {row['accuracy']:.4f} ({row['accuracy'] * 100:.2f}%)\n\n"
            )
        f.write("=" * 80 + "\n")
        f.write(f"Full results: {summary_path}\n")
        f.write(f"Best config: {best_config_path}\n")

    print(f"✓ Final summary appended to: {progress_log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune RAG hyperparameters")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dev",
        choices=["dev", "train", "test"],
        help="Dataset to use for tuning",
    )
    parser.add_argument(
        "--top-k-min",
        type=int,
        default=3,
        help="Minimum top_k value",
    )
    parser.add_argument(
        "--top-k-max",
        type=int,
        default=11,
        help="Maximum top_k value (exclusive)",
    )
    parser.add_argument(
        "--bm25-min",
        type=float,
        default=0.1,
        help="Minimum BM25 weight",
    )
    parser.add_argument(
        "--bm25-max",
        type=float,
        default=0.55,
        help="Maximum BM25 weight (exclusive)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start from this experiment number (default: 1)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_hyperparameter_tuning(
            dataset=args.dataset,
            top_k_range=(args.top_k_min, args.top_k_max, 1),
            bm25_weight_range=(args.bm25_min, args.bm25_max, 0.05),
            start_from=args.start_from,
        )
    )
