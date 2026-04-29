import os
import json
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from src.phase3.evaluators.rag_evaluator import RAGEvaluator
from src.models.saf_rag.saf_rag import SAFRAG
from src.retrieval.pipeline.saf_pipeline import SAFRetrievalPipeline

from src.phase4.grounding_scorer import GroundingScorer
from src.phase4.risk_controller import RiskController

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "data/processed/roleplay_benchmark/roleplay_test.json"
BASELINE_PRED_PATH = "outputs/predictions/transformer_baseline.json"
SAF_MODEL_PATH = "models/saf_rag_model"
CORPUS_PATH = "data/rag_corpus/rag_passages.json"

OUTPUT_DIR = "experiments/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SAMPLES = 50

# 🔥 GLOBAL CACHE (prevents re-encoding every run)
_RETRIEVER = None


# =========================================================
# ABSTENTION MESSAGE
# =========================================================
def abstain_message():
    return "I don't have enough evidence in the provided documents to answer this question reliably."


# =========================================================
# LOAD DATA
# =========================================================
def load_dataset():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data[:MAX_SAMPLES]


def load_corpus():
    with open(CORPUS_PATH, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return list(data.values())

    if isinstance(data, list):
        return [
            d["text"] if isinstance(d, dict) and "text" in d else str(d)
            for d in data
        ]

    return []


# =========================================================
# COMPONENT LOADING (FIXED)
# =========================================================
def load_retriever():
    global _RETRIEVER

    if _RETRIEVER is not None:
        return _RETRIEVER

    print("[INFO] Initializing retriever (this may take time once)...")

    _RETRIEVER = SAFRetrievalPipeline(
        corpus=load_corpus(),
        device=DEVICE,
        top_k_dense=5,
        top_k_sparse=3,
        fusion_alpha=0.6
    )

    return _RETRIEVER


def load_model():
    model = SAFRAG(device=DEVICE)

    state_path = os.path.join(SAF_MODEL_PATH, "pytorch_model.bin")

    if os.path.exists(state_path):
        state_dict = torch.load(state_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# RETRIEVAL (FIXED HARD)
# =========================================================
def retrieve_context(retriever, query):
    try:
        result = retriever.retrieve(query, top_k=7)

        docs = result.get("documents", [])

        safe_docs = []

        for d in docs:
            if isinstance(d, dict):
                safe_docs.append({
                    "text": str(d.get("text", "")),
                    "dense_score": float(d.get("dense_score", 0.0)),
                    "sparse_score": float(d.get("sparse_score", 0.0)),
                    "hybrid_score": float(d.get("hybrid_score", 0.0)),
                })
            else:
                safe_docs.append({
                    "text": str(d),
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "hybrid_score": 0.0,
                })

        if not safe_docs:
            return [{"text": ""}]

        return safe_docs

    except Exception as e:
        print(f"[RETRIEVAL ERROR] {e}")
        return [{"text": ""}]


# =========================================================
# GENERATION (SAFE)
# =========================================================
def generate(model, query, docs):
    try:
        with torch.no_grad():
            output = model.generate(query=query, documents=docs)

        if not output or len(output.strip()) < 3:
            return abstain_message()

        return output.strip()

    except Exception as e:
        print(f"[GENERATION ERROR] {e}")
        return abstain_message()


# =========================================================
# SAF PIPELINE (FIXED)
# =========================================================
def generate_saf_predictions(dataset):
    model = load_model()
    retriever = load_retriever()

    grounding_scorer = GroundingScorer(device=DEVICE)
    risk_controller = RiskController(device=DEVICE)

    results = []

    for item in tqdm(dataset):
        query = item.get("input", "")

        docs = retrieve_context(retriever, query)
        doc_texts = [d["text"] for d in docs]

        prediction = generate(model, query, docs)

        grounding = grounding_scorer.score(query, prediction, doc_texts)
        risk = risk_controller.estimate(query, doc_texts)

        # 🔥 sycophancy trigger
        sycophancy_trigger = int(
            any(
                phrase in query.lower()
                for phrase in [
                    "you must agree",
                    "don't contradict",
                    "be biased",
                    "support this claim"
                ]
            )
        )

        # 🔥 SAF GUARDRAIL
        if grounding < 0.25 and risk > 0.85:
            prediction = abstain_message()

        results.append({
            "query": query,
            "context": " ".join(doc_texts),
            "prediction": prediction,
            "risk": float(risk),
            "grounding": float(grounding),
            "sycophancy_trigger": sycophancy_trigger
        })

    return results


# =========================================================
# BASELINE
# =========================================================
def load_baseline():
    with open(BASELINE_PRED_PATH, "r") as f:
        data = json.load(f)

    return [
        {
            "query": x.get("input", ""),
            "context": str(x.get("context", "")),
            "prediction": x.get("prediction", "")
        }
        for x in data[:MAX_SAMPLES]
    ]


# =========================================================
# SAVE TABLE
# =========================================================
def save_comparison_table(saf_metrics, base_metrics):
    table = {
        "metric": list(saf_metrics.keys()),
        "SAF": list(saf_metrics.values()),
        "BASELINE": list(base_metrics.values())
    }

    path = os.path.join(OUTPUT_DIR, "comparison_table.json")
    with open(path, "w") as f:
        json.dump(table, f, indent=2)

    print(f"[SAVED] {path}")


# =========================================================
# PLOT RESULTS
# =========================================================
def plot_results(saf_metrics, base_metrics):
    metrics = list(saf_metrics.keys())
    saf_vals = list(saf_metrics.values())
    base_vals = list(base_metrics.values())

    x = range(len(metrics))

    plt.figure(figsize=(10, 5))

    plt.bar(x, saf_vals, label="SAF")
    plt.bar(x, base_vals, label="BASELINE", alpha=0.6)

    plt.xticks(x, metrics, rotation=30)
    plt.legend()
    plt.title("SAF vs Baseline Performance")

    path = os.path.join(OUTPUT_DIR, "comparison_plot.png")
    plt.tight_layout()
    plt.savefig(path)

    print(f"[SAVED] {path}")


# =========================================================
# MAIN
# =========================================================
def main():
    dataset = load_dataset()

    print("[INFO] Generating SAF predictions...")
    saf_preds = generate_saf_predictions(dataset)

    print("[INFO] Loading baseline...")
    base_preds = load_baseline()

    evaluator = RAGEvaluator(device=DEVICE)

    print("[INFO] Evaluating SAF...")
    saf_metrics = evaluator.evaluate(saf_preds)

    print("[INFO] Evaluating baseline...")
    base_metrics = evaluator.evaluate(base_preds)

    print("\n===== RESULTS =====")
    print("SAF:", saf_metrics)
    print("BASE:", base_metrics)

    # SAVE RESULTS
    with open(os.path.join(OUTPUT_DIR, "saf_rag_eval.json"), "w") as f:
        json.dump({"saf": saf_metrics, "base": base_metrics}, f, indent=2)

    save_comparison_table(saf_metrics, base_metrics)
    plot_results(saf_metrics, base_metrics)


if __name__ == "__main__":
    main()