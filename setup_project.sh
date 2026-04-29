#!/bin/bash

# Use current directory as root
ROOT="$(pwd)"

# --- Create folders ---
folders=(
"data/raw"
"data/processed"
"data/ground_truth"
"models/transformer_baseline"
"models/transformer_prompted"
"models/transformer_rag"
"models/lstm_baseline"
"experiments/evaluation_metrics"
"experiments/experiment_results"
"notebooks"
"configs"
"utils"
"scripts"
"logs"
"outputs"
)

for folder in "${folders[@]}"; do
    mkdir -p "$ROOT/$folder"
done

# --- Create files ---
files=(
"README.md"
"requirements.txt"
"configs/model_config.yaml"
"configs/experiment_config.yaml"
"utils/data_utils.py"
"utils/model_utils.py"
"utils/eval_utils.py"
"experiments/run_experiments.py"
"experiments/evaluation_metrics/role_consistency.py"
"experiments/evaluation_metrics/persona_drift.py"
"experiments/evaluation_metrics/hallucination_check.py"
"notebooks/data_exploration.ipynb"
"notebooks/metric_validation.ipynb"
"notebooks/experiment_visualization.ipynb"
"scripts/preprocess_data.py"
"scripts/postprocess_results.py"
"scripts/train_transformer_baseline.py"
"scripts/train_transformer_prompted.py"
"scripts/train_transformer_rag.py"
"scripts/train_lstm_baseline.py"
"scripts/evaluate_models.py"
)

for file in "${files[@]}"; do
    touch "$ROOT/$file"
done

# --- Add .gitkeep in empty folders ---
for folder in "${folders[@]}"; do
    touch "$ROOT/$folder/.gitkeep"
done

echo "Project structure created successfully in $ROOT!"