"""Verification script to test expansion with a single example."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from expand_data import call_gpt4o, create_expansion_prompt, has_negation_pattern
from openai import OpenAI


def main() -> None:
    """Verify expansion works with a single example."""
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("✗ OPENAI_API_KEY not found in .env file")
        print("Please add: OPENAI_API_KEY=your-key-here")
        return

    print("✓ API Key loaded")

    # Load train data
    root = Path(__file__).parent.parent.parent
    train_path = root / "data" / "train.csv"
    train_df = pd.read_csv(train_path)

    # Find a question with negation
    negation_df = train_df[train_df["question"].apply(has_negation_pattern)]

    if len(negation_df) == 0:
        print("✗ No questions with negation patterns found")
        return

    # Test with first negation question
    test_row = negation_df.iloc[0]

    print("\n" + "=" * 60)
    print("Testing with sample question:")
    print("=" * 60)
    print(f"\n원본 질문: {test_row['question']}")
    print(f"A) {test_row['A']}")
    print(f"B) {test_row['B']}")
    print(f"C) {test_row['C']}")
    print(f"D) {test_row['D']}")
    print(f"정답: {test_row['answer']}번")
    print(f"카테고리: {test_row['Category']}")

    # Create prompt
    prompt = create_expansion_prompt(test_row)

    print("\n" + "-" * 60)
    print("Calling GPT-4o...")
    print("-" * 60)

    # Call API
    client = OpenAI(api_key=api_key)
    result = call_gpt4o(client, prompt)

    if result:
        print("\n✓ Conversion successful!")
        print("\n" + "=" * 60)
        print("Converted question:")
        print("=" * 60)
        print(f"\n변환된 질문: {result.get('converted_question')}")

        possible_answers = result.get("possible_answers", [])
        print(f"\n가능한 정답들: {possible_answers}")
        print(f"→ {len(possible_answers)}개의 확장 데이터가 생성됩니다")

        print("\n각 선택지 분석:")
        answer_explanations = result.get("answer_explanations", {})
        for i in range(1, 5):
            explanation = answer_explanations.get(str(i), "N/A")
            is_answer = "✓" if i in possible_answers else "✗"
            print(f"  {is_answer} {i}번: {explanation}")

        print("\n" + "=" * 60)
        print("✓ Verification completed successfully!")
        print("You can now run the full expansion script.")
        print("=" * 60)
    else:
        print("\n✗ Conversion failed")
        print("Please check your API key and try again")


if __name__ == "__main__":
    main()
