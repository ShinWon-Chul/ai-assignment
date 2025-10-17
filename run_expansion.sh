#!/bin/bash

# Data Expansion Script Runner V2
# Multi-answer expansion with parallel processing

echo "==========================================="
echo "Query Expansion V2: Multi-Answer + Parallel"
echo "==========================================="
echo ""
echo "✨ New Features:"
echo "  - Multi-answer expansion (1 question → 2-3 expansions)"
echo "  - 10x parallel processing"
echo "  - Separate comparison file"
echo ""
echo "This will:"
echo "  1. Analyze train.csv for negation patterns"
echo "  2. Convert using GPT-4o (parallel)"
echo "  3. Create train_extended.csv (original + expanded)"
echo "  4. Create train_extended_only.csv (expanded only)"
echo ""
echo "⚠️  Estimated cost: ~\$2-3 USD"
echo "⚠️  Estimated time: 10-20 minutes (faster!)"
echo "⚠️  Expected: ~3,000 expanded rows"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Starting data expansion..."
echo ""

cd "$(dirname "$0")"
python src/preprocessing/expand_data.py

if [ $? -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "✓ Expansion completed successfully!"
    echo "==========================================="
    echo ""
    echo "📁 Generated files:"
    echo "  1. data/train_extended.csv (original + expanded)"
    echo "  2. data/train_extended_only.csv (expanded only)"
    echo ""
    echo "🔍 Compare results:"
    echo "  python src/preprocessing/compare_expansion.py"
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Compare expansion results"
    echo "  2. Update process_data.py to use train_extended.csv"
    echo "  3. Re-run: python src/preprocessing/generate_embeddings.py"
    echo "  4. Test: python src/evaluation/evaluate.py"
    echo ""
else
    echo ""
    echo "✗ Expansion failed. Check the error messages above."
    exit 1
fi

