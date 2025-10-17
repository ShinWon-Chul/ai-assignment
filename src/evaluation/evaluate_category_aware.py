"""Evaluate RAG Agent with Category-Aware Retrieval."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.agent.prompts import SYSTEM_PROMPT, create_rag_prompt
from src.agent.rag_agent import RAGAgent

# Load environment variables
load_dotenv()


async def evaluate_single_sample(
    agent: RAGAgent,
    idx: int,
    row: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate a single sample with the RAG agent using category-aware retrieval.
    
    Args:
        agent: RAG agent instance
        idx: Sample index
        row: Data row
        
    Returns:
        Result dictionary for this sample
    """
    try:
        # Build question text
        question = str(row["question"])
        options_text = f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
        question_text = f"{question}\n{options_text}"
        category = str(row["Category"])
        
        # Get prediction with category-aware retrieval
        if agent.use_rag and agent.retriever:
            retrieved = await agent.retriever.retrieve(
                query=question_text, 
                top_k=agent.top_k,
                category=category  # Pass category for category-aware boost
            )
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
            prediction = await agent.predict(question_text, category)
        
        # Check correctness
        correct_answer = ["A", "B", "C", "D"][int(row["answer"]) - 1]
        is_correct = prediction == correct_answer
        
        # Count category-boosted documents
        boosted_count = sum(1 for doc in retrieved if doc.get("category_boosted", False)) if agent.use_rag else 0
        
        # Store result
        result = {
            "idx": int(idx),
            "question": question,
            "prediction": prediction,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "category": category,
            "human_accuracy": float(row["Human Accuracy"]),
            "boosted_docs_count": boosted_count,
            "retrieved_documents": [
                {
                    "question": doc["question"],
                    "answer": doc["answer"],
                    "category": doc.get("category", ""),
                    "hybrid_score": doc.get("hybrid_score", None),
                    "semantic_score": doc.get("semantic_score", None),
                    "bm25_score": doc.get("bm25_score", None),
                    "category_boosted": doc.get("category_boosted", False),
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
            "boosted_docs_count": 0,
            "retrieved_documents": [],
            "error": str(e),
        }


async def evaluate_category_aware(
    dataset: str = "dev",
    top_k: int = 5,
    semantic_weight: float = 0.5,
    bm25_weight: float = 0.5,
    category_boost: float = 0.15,
) -> None:
    """
    Evaluate RAG agent with category-aware retrieval.
    
    Args:
        dataset: Dataset name (dev or train)
        top_k: Number of documents to retrieve
        semantic_weight: Weight for semantic search
        bm25_weight: Weight for BM25 search
        category_boost: Boost score for same-category documents
    """
    # Setup paths
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"
    output_dir = root / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    data_path = data_dir / f"{dataset}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = pd.read_csv(data_path)
    total = len(df)
    
    print(f"\n{'='*80}")
    print(f"Category-Aware Retrieval Evaluation")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Total samples: {total}")
    print(f"Parameters:")
    print(f"  - top_k: {top_k}")
    print(f"  - semantic_weight: {semantic_weight:.2f}")
    print(f"  - bm25_weight: {bm25_weight:.2f}")
    print(f"  - category_boost: {category_boost:.2f}")
    print(f"{'='*80}\n")
    
    # Initialize agent with category-aware retrieval
    print("Initializing RAG Agent with Category-Aware Retrieval...")
    agent = RAGAgent(
        top_k=top_k,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        use_hybrid=True,
        category_boost=category_boost,
    )
    
    # Prepare tasks for parallel execution
    print(f"Preparing {total} evaluation tasks...")
    tasks = []
    for idx, row in df.iterrows():
        task = evaluate_single_sample(agent, idx, row)
        tasks.append(task)
    
    # Execute all tasks in parallel
    print(f"Evaluating {total} samples in parallel...")
    detailed_results = await asyncio.gather(*tasks)
    
    # Sort results by idx to maintain original order
    detailed_results = sorted(detailed_results, key=lambda x: x["idx"])
    
    # Calculate metrics
    correct = sum(1 for r in detailed_results if r["is_correct"])
    accuracy = correct / total
    
    # Calculate category-wise metrics
    df_results = pd.DataFrame(detailed_results)
    category_stats = df_results.groupby("category").agg({
        "is_correct": ["sum", "count", "mean"],
        "boosted_docs_count": "mean"
    }).round(4)
    
    print(f"\n{'='*80}")
    print(f"Evaluation Results")
    print(f"{'='*80}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nCategory-wise Results:")
    for category in category_stats.index:
        correct_cat = int(category_stats.loc[category, ("is_correct", "sum")])
        total_cat = int(category_stats.loc[category, ("is_correct", "count")])
        acc_cat = float(category_stats.loc[category, ("is_correct", "mean")])
        avg_boosted = float(category_stats.loc[category, ("boosted_docs_count", "mean")])
        print(f"  {category}: {correct_cat}/{total_cat} ({acc_cat:.4f}) - Avg boosted docs: {avg_boosted:.2f}/{top_k}")
    print(f"{'='*80}\n")
    
    # Save detailed results (JSON)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"category_aware_boost{int(category_boost*100):02d}"
    
    # Convert category_stats to JSON-serializable format
    category_stats_dict = {}
    for category in category_stats.index:
        category_stats_dict[category] = {
            "correct": int(category_stats.loc[category, ("is_correct", "sum")]),
            "total": int(category_stats.loc[category, ("is_correct", "count")]),
            "accuracy": float(category_stats.loc[category, ("is_correct", "mean")]),
            "avg_boosted_docs": float(category_stats.loc[category, ("boosted_docs_count", "mean")]),
        }
    
    result_data = {
        "experiment_id": experiment_id,
        "parameters": {
            "top_k": top_k,
            "semantic_weight": semantic_weight,
            "bm25_weight": bm25_weight,
            "category_boost": category_boost,
        },
        "metrics": {
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "category_stats": category_stats_dict,
        },
        "timestamp": datetime.now().isoformat(),
        "model": agent.model,
        "results": detailed_results,
    }
    
    json_path = output_dir / f"category_aware_{dataset}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved detailed results to {json_path}")
    
    # Save summary (TXT)
    txt_path = output_dir / f"category_aware_{dataset}_{timestamp}_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Category-Aware Retrieval Evaluation Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  - top_k: {top_k}\n")
        f.write(f"  - semantic_weight: {semantic_weight:.2f}\n")
        f.write(f"  - bm25_weight: {bm25_weight:.2f}\n")
        f.write(f"  - category_boost: {category_boost:.2f}\n\n")
        f.write(f"Overall Results:\n")
        f.write(f"  - Total samples: {total}\n")
        f.write(f"  - Correct predictions: {correct}\n")
        f.write(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write(f"Category-wise Results:\n")
        for category in category_stats.index:
            correct_cat = int(category_stats.loc[category, ("is_correct", "sum")])
            total_cat = int(category_stats.loc[category, ("is_correct", "count")])
            acc_cat = float(category_stats.loc[category, ("is_correct", "mean")])
            avg_boosted = float(category_stats.loc[category, ("boosted_docs_count", "mean")])
            f.write(f"  - {category}:\n")
            f.write(f"      Accuracy: {correct_cat}/{total_cat} ({acc_cat:.4f})\n")
            f.write(f"      Avg boosted docs: {avg_boosted:.2f}/{top_k}\n")
    
    print(f"✓ Saved summary to {txt_path}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Category-Aware Retrieval")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dev",
        choices=["dev", "train", "test"],
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.5,
        help="Weight for semantic search",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.5,
        help="Weight for BM25 search",
    )
    parser.add_argument(
        "--category-boost",
        type=float,
        default=0.15,
        help="Boost score for same-category documents (0.0-0.3)",
    )
    
    args = parser.parse_args()
    
    asyncio.run(
        evaluate_category_aware(
            dataset=args.dataset,
            top_k=args.top_k,
            semantic_weight=args.semantic_weight,
            bm25_weight=args.bm25_weight,
            category_boost=args.category_boost,
        )
    )

