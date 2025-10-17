#!/bin/bash

# Category-Aware Retrieval 평가 스크립트

echo "========================================"
echo "Category-Aware Retrieval 평가"
echo "========================================"
echo ""
echo "최적 하이퍼파라미터:"
echo "  - top_k: 5"
echo "  - semantic_weight: 0.50"
echo "  - bm25_weight: 0.50"
echo ""
echo "Category Boost 테스트 (0.05 간격):"
echo "  - 0.00 (baseline, no category boost)"
echo "  - 0.05"
echo "  - 0.10"
echo "  - 0.15"
echo "  - 0.20"
echo "  - 0.25"
echo "  - 0.30"
echo ""
echo "총 7개 실험"
echo "예상 시간: 약 35-50분"
echo ""
echo "⚠️  OpenAI API 비용이 발생합니다!"
echo ""
read -p "계속하시겠습니까? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "평가가 취소되었습니다."
    exit 1
fi

echo ""
echo "평가를 시작합니다..."
echo ""

cd "$(dirname "$0")"

# 1. Baseline (no category boost)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/7] Category Boost = 0.00 (Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m src.evaluation.evaluate_category_aware \
    --dataset dev \
    --top-k 5 \
    --semantic-weight 0.5 \
    --bm25-weight 0.5 \
    --category-boost 0.0

# 2. Boost = 0.05
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/7] Category Boost = 0.05"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m src.evaluation.evaluate_category_aware \
    --dataset dev \
    --top-k 5 \
    --semantic-weight 0.5 \
    --bm25-weight 0.5 \
    --category-boost 0.05

# 3. Boost = 0.10
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/7] Category Boost = 0.10"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m src.evaluation.evaluate_category_aware \
    --dataset dev \
    --top-k 5 \
    --semantic-weight 0.5 \
    --bm25-weight 0.5 \
    --category-boost 0.10

# 4. Boost = 0.15
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/7] Category Boost = 0.15"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m src.evaluation.evaluate_category_aware \
    --dataset dev \
    --top-k 5 \
    --semantic-weight 0.5 \
    --bm25-weight 0.5 \
    --category-boost 0.15

# 5. Boost = 0.20
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[5/7] Category Boost = 0.20"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m src.evaluation.evaluate_category_aware \
    --dataset dev \
    --top-k 5 \
    --semantic-weight 0.5 \
    --bm25-weight 0.5 \
    --category-boost 0.20

# 6. Boost = 0.25
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[6/7] Category Boost = 0.25"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m src.evaluation.evaluate_category_aware \
    --dataset dev \
    --top-k 5 \
    --semantic-weight 0.5 \
    --bm25-weight 0.5 \
    --category-boost 0.25

# 7. Boost = 0.30
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[7/7] Category Boost = 0.30"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run python -m src.evaluation.evaluate_category_aware \
    --dataset dev \
    --top-k 5 \
    --semantic-weight 0.5 \
    --bm25-weight 0.5 \
    --category-boost 0.30

echo ""
echo "========================================"
echo "평가 완료!"
echo "========================================"
echo ""
echo "결과 확인:"
echo "  ls -lh evaluation_results/category_aware_*"
echo ""
echo "모든 결과 요약:"
echo "  grep -h 'Accuracy:' evaluation_results/category_aware_dev_*_summary.txt | tail -7"
echo ""

