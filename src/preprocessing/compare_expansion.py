"""Compare original train.csv with expanded data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def compare_datasets(
    train_path: Path,
    extended_only_path: Path,
    num_samples: int = 5,
) -> None:
    """
    Compare original train.csv with train_extended_only.csv.

    Args:
        train_path: Path to train.csv
        extended_only_path: Path to train_extended_only.csv
        num_samples: Number of sample pairs to display
    """
    # Load datasets
    if not train_path.exists():
        print(f"✗ File not found: {train_path}")
        sys.exit(1)

    if not extended_only_path.exists():
        print(f"✗ File not found: {extended_only_path}")
        print("Please run expansion first: python src/preprocessing/expand_data.py")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    expanded_df = pd.read_csv(extended_only_path)

    print("=" * 80)
    print("Dataset Comparison: Original vs Expanded")
    print("=" * 80)
    print()

    # Overall statistics
    print("📊 Overall Statistics:")
    print(f"  - Original dataset: {len(train_df):,} samples")
    print(f"  - Expanded dataset: {len(expanded_df):,} samples")
    print(f"  - Increase: {len(expanded_df) / len(train_df):.2f}x")
    print()

    # Expansion type breakdown
    if "expansion_type" in expanded_df.columns:
        print("📈 Expansion Type Breakdown:")
        expansion_counts = expanded_df["expansion_type"].value_counts()
        for exp_type, count in expansion_counts.items():
            print(f"  - {exp_type}: {count:,} samples")
        print()

    # Group by original_idx to see how many expanded versions per original
    if "original_idx" in expanded_df.columns:
        expansions_per_original = expanded_df.groupby("original_idx").size()
        print("🔢 Expansions per Original Question:")
        print(f"  - Average: {expansions_per_original.mean():.2f}")
        print(f"  - Min: {expansions_per_original.min()}")
        print(f"  - Max: {expansions_per_original.max()}")
        print()

        # Distribution
        print("  Distribution:")
        dist = expansions_per_original.value_counts().sort_index()
        for num_expansions, count in dist.items():
            print(
                f"    {num_expansions} expansions: {count} questions ({count / len(expansions_per_original) * 100:.1f}%)"
            )
        print()

    # Sample comparisons
    print("=" * 80)
    print(f"Sample Comparisons (showing {num_samples} examples)")
    print("=" * 80)
    print()

    # Get unique original indices
    if "original_idx" in expanded_df.columns:
        unique_originals = expanded_df["original_idx"].unique()[:num_samples]

        for i, orig_idx in enumerate(unique_originals, 1):
            # Get original question
            orig_row = train_df.iloc[orig_idx]

            # Get all expanded versions
            expanded_rows = expanded_df[expanded_df["original_idx"] == orig_idx]

            print(f"Example {i}:")
            print("-" * 80)
            print(f"📌 Original (정답: {orig_row['answer']}번):")
            print(f"   {orig_row['question']}")
            print(f"   A) {orig_row['A']}")
            print(f"   B) {orig_row['B']}")
            print(f"   C) {orig_row['C']}")
            print(f"   D) {orig_row['D']}")
            print()

            print(f"🔄 Expanded ({len(expanded_rows)} versions):")
            for j, (_, exp_row) in enumerate(expanded_rows.iterrows(), 1):
                print(f"\n   Version {j} (정답: {exp_row['answer']}번):")
                print(f"   {exp_row['question']}")
            print()
            print("=" * 80)
            print()


def main() -> None:
    """Main function."""
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"

    train_path = data_dir / "train.csv"
    extended_only_path = data_dir / "train_extended_only.csv"

    # Check if test files exist
    test_extended_only = data_dir / "train_extended_only_test.csv"
    if test_extended_only.exists() and not extended_only_path.exists():
        print("ℹ️  Using test data for comparison")
        extended_only_path = test_extended_only

    compare_datasets(train_path, extended_only_path, num_samples=5)

    print(
        "\n💡 Tip: To see more examples, edit compare_expansion.py and change num_samples"
    )


if __name__ == "__main__":
    main()
