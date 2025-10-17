#!/bin/bash

# 하이퍼파라미터 튜닝 실행 스크립트

echo "========================================"
echo "RAG 하이퍼파라미터 튜닝 시작"
echo "========================================"
echo ""
echo "설정:"
echo "  - top_k: 3~10 (8개 값)"
echo "  - bm25_weight: 0.10~0.50 (0.05씩, 9개 값)"
echo "  - 총 실험 수: 72개"
echo "  - 예상 시간: 90~120분"
echo ""
echo "⚠️  OpenAI API 비용이 발생합니다!"
echo "⚠️  언제든지 Ctrl+C로 중단 가능합니다."
echo "⚠️  병렬 실행으로 속도가 빨라졌습니다."
echo ""
read -p "계속하시겠습니까? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "튜닝이 취소되었습니다."
    exit 1
fi

echo ""
echo "튜닝을 시작합니다..."
echo ""

# 실행
cd "$(dirname "$0")"
uv run python -m src.evaluation.tune_hyperparams \
    --dataset dev \
    --top-k-min 3 \
    --top-k-max 11 \
    --bm25-min 0.1 \
    --bm25-max 0.55

echo ""
echo "========================================"
echo "튜닝 완료!"
echo "========================================"
echo ""
echo "결과 확인:"
echo "  - 요약: evaluation_results/tuning_summary.csv"
echo "  - 최적 설정: evaluation_results/best_config.txt"
echo "  - 실시간 로그: evaluation_results/tuning_progress_*.txt"
echo "  - 개별 실험: evaluation_results/exp_*.json"
echo ""
