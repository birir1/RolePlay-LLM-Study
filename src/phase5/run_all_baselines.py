import json
import subprocess

BASELINES = [
    "transformer_baseline",
    "transformer_prompted",
    "mt5_baseline",
]

RAG_MODELS = [
    "rag_outputs",
    "rag_outputs_v2",
]

SAF = [
    "saf_rag_eval"
]


def run(cmd):
    print(f"\n[RUNNING] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    print("[INFO] Running all baselines...\n")

    # 1. Transformer baselines
    run("python src/phase3/generate_predictions.py --model transformer_baseline")
    run("python src/phase3/generate_predictions.py --model transformer_prompted")
    run("python src/phase3/generate_predictions.py --model mt5_baseline")

    # 2. RAG outputs
    run("python src/phase2/run_rag_pipeline.py")

    # 3. SAF evaluation
    run("python src/phase4/evaluate_saf_rag.py")

    print("\n[SUCCESS] All baselines executed.")


if __name__ == "__main__":
    main()