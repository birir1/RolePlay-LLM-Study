import os
import json
from tqdm import tqdm

from src.phase4.saf_rag_pipeline import SAFRAGPipeline

# =========================
# PATHS (REAL DATA)
# =========================
INPUT_DATA = "data/processed/roleplay_benchmark/roleplay_test.json"

RAG_DATASET = "data/rag_index/my_dataset"
RAG_INDEX = "data/rag_index/hf_dataset_index.faiss"

MODEL_PATH = "models/transformer_baseline"

OUTPUT_PATH = "data/processed/saf_training/saf_train.json"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# =========================
# LOAD DATA
# =========================
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    print(f"[INFO] Loaded {len(data)} samples")
    return data


# =========================
# BUILD SAF DATASET
# =========================
def build_dataset():
    data = load_data(INPUT_DATA)

    pipeline = SAFRAGPipeline(
        model_path=MODEL_PATH,
        rag_dataset_path=RAG_DATASET,
        rag_index_path=RAG_INDEX
    )

    results = []

    for item in tqdm(data):
        query = item["input"]

        output = pipeline.run(query)

        #  KEY IDEA:
        # We train on SAFE QUERY → GROUNDED ANSWER
        results.append({
            "input": output["safe_query"],
            "target": output["prediction"],
            "original_query": query,
            "risk": output["risk"],
            "grounding": output["grounding"]
        })

    return results


# =========================
# SAVE
# =========================
def save(data):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[SUCCESS] Saved SAF training data → {OUTPUT_PATH}")
    print(f"[INFO] Total samples: {len(data)}")


# =========================
# MAIN
# =========================
def main():
    dataset = build_dataset()
    save(dataset)


if __name__ == "__main__":
    main()