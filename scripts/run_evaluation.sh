#!/bin/bash

set -e  # stop on error

echo "========================================"
echo "🚀 STARTING FULL EVALUATION PIPELINE"
echo "========================================"

# Activate environment
source .venv/bin/activate

# Paths
PRED_DIR="outputs/predictions"
RESULT_DIR="experiments/results"
EVAL_SCRIPT="src/phase3/run_evaluation.py"

mkdir -p $RESULT_DIR

echo ""
echo "📂 Predictions directory: $PRED_DIR"
echo "📂 Results directory: $RESULT_DIR"
echo ""

# =========================
# CHECK FILES
# =========================
echo "🔍 Checking prediction files..."

models=("mt5_baseline" "transformer_baseline" "transformer_prompted")

for model in "${models[@]}"; do
    file="$PRED_DIR/$model.json"
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        exit 1
    else
        echo "✅ Found: $file"
    fi
done

echo ""
echo "========================================"
echo "📊 RUNNING EVALUATION"
echo "========================================"

# =========================
# RUN EVALUATION
# =========================
python $EVAL_SCRIPT \
    --pred_dir $PRED_DIR \
    --output_dir $RESULT_DIR

echo ""
echo "========================================"
echo "📈 GENERATING TABLES"
echo "========================================"

# Convert results to table formats
python scripts/runners/postprocess_results.py \
    --input $RESULT_DIR/report.json \
    --output $RESULT_DIR

echo ""
echo "========================================"
echo "✅ EVALUATION COMPLETE"
echo "========================================"

echo ""
echo "📄 Outputs:"
echo " - JSON Report: $RESULT_DIR/report.json"
echo " - Markdown Table: $RESULT_DIR/benchmark_table.md"
echo " - LaTeX Table: $RESULT_DIR/benchmark_table.tex"
echo " - Plots: $RESULT_DIR/plots/"
echo ""