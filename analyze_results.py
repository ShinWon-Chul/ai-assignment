"""Analyze detailed evaluation results."""

import json
import sys
from collections import defaultdict
from pathlib import Path


def analyze_results(results_path: str) -> None:
    """
    Analyze detailed evaluation results.

    Args:
        results_path: Path to the detailed evaluation JSON file
    """
    print("=" * 70)
    print("Detailed Evaluation Analysis")
    print("=" * 70)
    print()

    # Load results
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total

    # Basic stats
    print("📊 Overall Statistics:")
    print(f"  Dataset: {data['dataset']}")
    print(f"  Model: {data['model']}")
    print(f"  Mode: {'RAG' if data['use_rag'] else 'Zero-shot'}")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.4f}")
    print()

    # Error analysis
    wrong_results = [r for r in results if not r["is_correct"]]
    print("❌ Error Analysis:")
    print(f"  Wrong predictions: {len(wrong_results)}")
    print()

    # Check retrieval quality for wrong answers
    if data["use_rag"]:
        retrieval_had_answer = 0
        retrieval_no_answer = 0

        for result in wrong_results:
            retrieved_answers = [doc["answer"] for doc in result["retrieved_documents"]]
            if result["correct_answer"] in retrieved_answers:
                retrieval_had_answer += 1
            else:
                retrieval_no_answer += 1

        print("  Retrieval Analysis:")
        print(f"    Correct answer in retrieved docs: {retrieval_had_answer}")
        print(f"    Correct answer NOT in retrieved docs: {retrieval_no_answer}")
        print(
            f"    → Retrieval miss rate: {retrieval_no_answer / len(wrong_results) * 100:.1f}%"
        )
        print()

    # Negation pattern analysis
    negation_patterns = ["옳지 않은", "틀린", "잘못된", "해당하지 않는", "아닌"]
    negation_results = [
        r
        for r in results
        if any(pattern in r["question"] for pattern in negation_patterns)
    ]

    if negation_results:
        negation_correct = sum(1 for r in negation_results if r["is_correct"])
        negation_accuracy = negation_correct / len(negation_results)

        print("🔄 Negation Pattern Analysis:")
        print(f"  Total negation questions: {len(negation_results)}")
        print(f"  Correct: {negation_correct}")
        print(f"  Accuracy: {negation_accuracy:.4f}")
        print()

    # Category analysis
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for result in results:
        cat = result["category"]
        category_stats[cat]["total"] += 1
        if result["is_correct"]:
            category_stats[cat]["correct"] += 1

    print("📚 Category Performance:")
    for cat, stats in sorted(category_stats.items()):
        acc = stats["correct"] / stats["total"]
        print(f"  {cat}: {acc:.4f} ({stats['correct']}/{stats['total']})")
    print()

    # Difficulty analysis
    difficulty_buckets = {"Easy (<0.3)": [], "Medium (0.3-0.7)": [], "Hard (>0.7)": []}

    for result in results:
        difficulty = result.get("human_accuracy", 0)
        # Note: difficulty = 1 - human_accuracy in the data
        if difficulty < 0.3:
            difficulty_buckets["Easy (<0.3)"].append(result)
        elif difficulty < 0.7:
            difficulty_buckets["Medium (0.3-0.7)"].append(result)
        else:
            difficulty_buckets["Hard (>0.7)"].append(result)

    print("🎯 Difficulty Analysis:")
    for bucket, bucket_results in difficulty_buckets.items():
        if bucket_results:
            bucket_correct = sum(1 for r in bucket_results if r["is_correct"])
            bucket_acc = bucket_correct / len(bucket_results)
            print(
                f"  {bucket}: {bucket_acc:.4f} ({bucket_correct}/{len(bucket_results)})"
            )
    print()

    # Retrieval quality
    if data["use_rag"]:
        similarities = []
        for result in results:
            docs = result["retrieved_documents"]
            if docs and docs[0].get("similarity"):
                similarities.append(docs[0]["similarity"])

        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            min_sim = min(similarities)
            max_sim = max(similarities)

            print("🔍 Retrieval Quality (Top-1 Similarity):")
            print(f"  Average: {avg_sim:.3f}")
            print(f"  Min: {min_sim:.3f}")
            print(f"  Max: {max_sim:.3f}")
            print()

    # Top wrong predictions
    print("🔴 Top 5 Wrong Predictions:")
    print()
    for i, result in enumerate(wrong_results[:5], 1):
        print(f"  {i}. Question: {result['question'][:80]}...")
        print(
            f"     Prediction: {result['prediction']} | Correct: {result['correct_answer']}"
        )
        if result["retrieved_documents"]:
            print(
                f"     Top retrieved answer: {result['retrieved_documents'][0]['answer']}"
            )
        print()

    print("=" * 70)


def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        # Find the latest detailed evaluation file
        results_dir = Path("evaluation_results")
        if not results_dir.exists():
            print("❌ No evaluation results found.")
            print(
                "Run: uv run python -m src.evaluation.evaluate --dataset dev --detailed"
            )
            sys.exit(1)

        json_files = sorted(results_dir.glob("detailed_eval_*.json"), reverse=True)
        if not json_files:
            print("❌ No detailed evaluation results found.")
            print(
                "Run: uv run python -m src.evaluation.evaluate --dataset dev --detailed"
            )
            sys.exit(1)

        results_path = json_files[0]
        print(f"Using latest results: {results_path}")
        print()
    else:
        results_path = Path(sys.argv[1])

    if not results_path.exists():
        print(f"❌ File not found: {results_path}")
        sys.exit(1)

    analyze_results(str(results_path))


if __name__ == "__main__":
    main()
