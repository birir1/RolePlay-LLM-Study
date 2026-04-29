#!/bin/bash

set -euo pipefail  # safer strict mode

echo "========================================"
echo " ROLEPLAY LLM BENCHMARK EVALUATION"
echo "========================================"

mkdir -p outputs/predictions
mkdir -p experiments/results
mkdir -p outputs/logs


# =========================
# STEP 1: GENERATE PREDICTIONS
# =========================
echo ""
echo " Step 1: Generating predictions (ALL MODELS)"
echo "----------------------------------------"

python src/phase3/generate_predictions.py

echo ""
echo "✔ Predictions complete"
echo "----------------------------------------"


# =========================
# STEP 2: EVALUATE MODELS
# =========================
echo ""
echo " Step 2: Evaluating models"
echo "----------------------------------------"

if compgen -G "outputs/predictions/*.json" > /dev/null; then
    for file in outputs/predictions/*.json
    do
        echo "→ Evaluating $file"
        python src/phase3/run_evaluation.py --predictions "$file"
    done
else
    echo "⚠ No prediction files found in outputs/predictions/"
fi

echo ""
echo "✔ Evaluation complete"
echo "----------------------------------------"


# =========================
# STEP 3: BUILD METRICS TABLE
# =========================
echo ""
echo " Step 3: Building results table + metrics aggregation"
echo "----------------------------------------"

python src/experiments/build_results_table.py

echo ""
echo "✔ Metrics aggregation complete"
echo "----------------------------------------"


# =========================
# STEP 4: GENERATE PLOTS
# =========================
echo ""
echo " Step 4: Generating plots"
echo "----------------------------------------"

python src/experiments/plot_metrics.py

echo ""
echo "✔ Plots generated"
echo "----------------------------------------"


# =========================
# STEP 5: SAF-RAG EVALUATION
# =========================
echo ""
echo " Step 5: Evaluating SAF-RAG"
echo "----------------------------------------"

if [ -f "src/phase4/evaluate_saf_rag.py" ]; then
    python src/phase4/evaluate_saf_rag.py || echo "⚠ SAF-RAG evaluation failed but continuing pipeline"
else
    echo "⚠ SAF-RAG evaluator not found"
fi


# =========================
# FINAL SUMMARY
# =========================
echo ""
echo "========================================"
echo " FULL PIPELINE COMPLETE"
echo " Results:"
echo " - experiments/results/"
echo " - outputs/predictions/"
echo " - experiments/results/plots/"
echo "========================================"