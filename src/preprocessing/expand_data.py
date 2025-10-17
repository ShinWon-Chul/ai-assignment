"""Data expansion script for Query Expansion using GPT-4o with parallel processing."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def has_negation_pattern(question: str) -> bool:
    """
    Check if question contains negation patterns.
    
    Args:
        question: Question text to check
        
    Returns:
        True if question contains negation pattern
    """
    negation_patterns = [
        r"옳지\s*않은",
        r"틀린",
        r"잘못된",
        r"부적절한",
        r"해당하지\s*않는",
        r"아닌",
        r"제외",
        r"불가능한",
    ]
    
    return any(re.search(pattern, question) for pattern in negation_patterns)


def create_expansion_prompt(row: pd.Series) -> str:
    """
    Create prompt for GPT-4o to expand question with opposite form.
    
    Args:
        row: DataFrame row containing question data
        
    Returns:
        Formatted prompt string
    """
    question = row["question"]
    answer = int(row["answer"])
    options = {
        "A": row["A"],
        "B": row["B"],
        "C": row["C"],
        "D": row["D"],
    }
    
    # Calculate possible answers (excluding original answer)
    all_answers = {1, 2, 3, 4}
    possible_answers = sorted(all_answers - {answer})
    
    prompt = f"""당신은 법률 문제 변환 전문가입니다. 아래 문제를 부정형↔긍정형으로 변환해주세요.

원본 문제:
질문: {question}
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}
정답: {answer}번

변환 규칙:
1. "옳지 않은 것" → "옳은 것"으로 변환 (또는 그 반대)
2. "틀린 것" → "맞는 것"으로 변환  
3. "해당하지 않는" → "해당하는"으로 변환
4. 선택지는 그대로 유지
5. **중요**: 정답 번호 변환 로직
   - 원본이 "옳지 않은 것"이고 정답이 {answer}번이면
   - 변환된 "옳은 것" 문제의 정답은 {answer}번을 **제외한** 나머지({', '.join(map(str, possible_answers))}번)
   - **모든 가능한 정답을 리스트로 반환**하세요
   
6. 각 선택지를 검토하여 실제로 옳은(또는 틀린) 것을 모두 찾으세요

변환된 문제와 **모든 가능한 정답**을 다음 JSON 형식으로 반환하세요:
{{
    "converted_question": "변환된 질문",
    "possible_answers": [정답_번호들의_리스트],
    "answer_explanations": {{
        "1": "1번이 정답인지 아닌지와 그 이유",
        "2": "2번이 정답인지 아닌지와 그 이유",
        "3": "3번이 정답인지 아닌지와 그 이유",
        "4": "4번이 정답인지 아닌지와 그 이유"
    }}
}}

중요: 
- 변환이 불가능한 경우 null을 반환
- possible_answers는 반드시 원본 정답({answer}번)을 **제외**해야 합니다
- 실제로 옳은(또는 틀린) 모든 선택지를 possible_answers에 포함하세요
- 확실하지 않은 경우에도 가장 가능성 높은 것들을 포함하세요"""

    return prompt


def call_gpt4o(client: OpenAI, prompt: str, max_retries: int = 3) -> dict[str, Any] | None:
    """
    Call GPT-4o API to expand question.
    
    Args:
        client: OpenAI client instance
        prompt: Prompt to send to GPT-4o
        max_retries: Maximum number of retries
        
    Returns:
        Parsed JSON response or None if failed
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in converting Korean legal exam questions between positive and negative forms. Always return ALL possible correct answers.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
    
    return None


def process_single_row(
    row_data: tuple[int, pd.Series],
    client: OpenAI,
) -> list[dict[str, Any]]:
    """
    Process a single row and return multiple expanded rows (one for each possible answer).
    
    Args:
        row_data: Tuple of (index, row)
        client: OpenAI client
        
    Returns:
        List of expanded row dictionaries
    """
    idx, row = row_data
    expanded_rows = []
    
    prompt = create_expansion_prompt(row)
    result = call_gpt4o(client, prompt)
    
    if result and result.get("converted_question") and result.get("possible_answers"):
        converted_question = result["converted_question"]
        possible_answers = result["possible_answers"]
        
        # Validate that possible answers don't include original answer
        original_answer = int(row["answer"])
        valid_answers = [ans for ans in possible_answers if ans != original_answer]
        
        if not valid_answers:
            return []
        
        # Create one row for each possible answer
        for new_answer in valid_answers:
            expanded_row = {
                "question": converted_question,
                "answer": new_answer,
                "A": row["A"],
                "B": row["B"],
                "C": row["C"],
                "D": row["D"],
                "Category": row["Category"],
                "Human Accuracy": row["Human Accuracy"],
                "source": "expanded",
                "original_idx": idx,
                "expansion_type": f"negation_flip_{len(valid_answers)}_answers",
            }
            expanded_rows.append(expanded_row)
    
    return expanded_rows


def expand_dataset(
    input_path: Path,
    output_path: Path,
    output_only_path: Path,
    api_key: str | None = None,
    sample_size: int | None = None,
    max_workers: int = 10,
) -> None:
    """
    Expand dataset by converting questions between positive/negative forms.
    
    Args:
        input_path: Path to train.csv
        output_path: Path to save train_extended.csv (original + expanded)
        output_only_path: Path to save train_extended_only.csv (expanded only)
        api_key: OpenAI API key (if None, loads from env)
        sample_size: Number of samples to process (if None, process all)
        max_workers: Number of parallel workers
    """
    # Load API key
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)
    
    # Load training data
    train_df = pd.read_csv(input_path)
    print(f"Loaded {len(train_df)} training samples")
    
    # Filter questions with negation patterns
    negation_mask = train_df["question"].apply(has_negation_pattern)
    expandable_df = train_df[negation_mask].copy()
    print(f"Found {len(expandable_df)} questions with negation patterns")
    
    # Sample if needed
    if sample_size and sample_size < len(expandable_df):
        expandable_df = expandable_df.sample(n=sample_size, random_state=42)
        print(f"Sampling {sample_size} questions for expansion")
    
    # Prepare data for parallel processing
    row_data_list = list(expandable_df.iterrows())
    
    # Expand questions in parallel
    expanded_rows = []
    failed_count = 0
    total_questions = 0
    
    print(f"\nExpanding with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_row, row_data, client): row_data[0]
            for row_data in row_data_list
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(future_to_idx), desc="Processing") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_rows = future.result()
                    if result_rows:
                        expanded_rows.extend(result_rows)
                        total_questions += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"\nError processing row {idx}: {e}")
                    failed_count += 1
                finally:
                    pbar.update(1)
    
    print(f"\n✓ Successfully expanded: {total_questions} questions → {len(expanded_rows)} rows")
    print(f"  - Average answers per question: {len(expanded_rows) / max(total_questions, 1):.2f}")
    print(f"✗ Failed to expand: {failed_count} questions")
    
    # Create extended dataset (original + expanded)
    train_df_copy = train_df.copy()
    train_df_copy["source"] = "original"
    train_df_copy["original_idx"] = train_df_copy.index
    train_df_copy["expansion_type"] = "none"
    
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Combine original + expanded
    extended_df = pd.concat([train_df_copy, expanded_df], ignore_index=True)
    
    # Shuffle
    extended_df = extended_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save combined dataset
    extended_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved extended dataset to: {output_path}")
    print(f"  - Original samples: {len(train_df)}")
    print(f"  - Expanded samples: {len(expanded_rows)}")
    print(f"  - Total samples: {len(extended_df)}")
    
    # Save expanded-only dataset for comparison
    expanded_only_df = expanded_df.copy()
    expanded_only_df.to_csv(output_only_path, index=False)
    print(f"\n✓ Saved expanded-only dataset to: {output_only_path}")
    print(f"  - Expanded samples: {len(expanded_only_df)}")
    
    # Statistics
    print("\nExpansion Statistics:")
    print(f"  - Questions processed: {total_questions}")
    print(f"  - Total expanded rows: {len(expanded_rows)}")
    print(f"  - Success rate: {total_questions / len(expandable_df) * 100:.1f}%")
    print(f"  - Dataset increase: {len(expanded_rows) / len(train_df) * 100:.1f}%")
    print(f"  - Expansion ratio: {len(expanded_rows) / max(total_questions, 1):.2f}x")
    
    # Expansion type breakdown
    if len(expanded_df) > 0:
        print("\nExpansion Type Breakdown:")
        expansion_counts = expanded_df["expansion_type"].value_counts()
        for exp_type, count in expansion_counts.items():
            print(f"  - {exp_type}: {count} rows")


def main() -> None:
    """Main function to run data expansion."""
    # Get paths
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"
    
    input_path = data_dir / "train.csv"
    output_path = data_dir / "train_extended.csv"
    output_only_path = data_dir / "train_extended_only.csv"
    
    # Check if input exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Run expansion
    print("=" * 60)
    print("Data Expansion: Query Expansion (Multi-Answer)")
    print("=" * 60)
    print()
    
    # For testing, you can use sample_size parameter
    # expand_dataset(input_path, output_path, output_only_path, sample_size=10)
    expand_dataset(input_path, output_path, output_only_path)
    
    print("\n" + "=" * 60)
    print("Expansion completed!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  1. {output_path} (original + expanded)")
    print(f"  2. {output_only_path} (expanded only, for comparison)")


if __name__ == "__main__":
    main()
