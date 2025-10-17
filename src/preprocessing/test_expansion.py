"""Test script for data expansion - processes only a few samples."""

from __future__ import annotations

import sys
from pathlib import Path

from expand_data import expand_dataset


def main() -> None:
    """Test data expansion with a small sample."""
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"

    input_path = data_dir / "train.csv"
    output_path = data_dir / "train_extended_test.csv"
    output_only_path = data_dir / "train_extended_only_test.csv"

    print("=" * 60)
    print("Testing Data Expansion (5 samples)")
    print("=" * 60)
    print()
    print("⚠️  This will cost approximately $0.05")
    print()

    response = input("Continue? (y/n): ")
    if response.lower() != "y":
        print("Cancelled.")
        sys.exit(0)

    # Test with 5 samples
    expand_dataset(input_path, output_path, output_only_path, sample_size=5)

    print("\n" + "=" * 60)
    print("Test completed!")
    print("Check the outputs:")
    print(f"  1. {output_path}")
    print(f"  2. {output_only_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
