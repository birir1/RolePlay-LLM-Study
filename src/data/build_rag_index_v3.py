import os
import json
import torch
import numpy as np
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from datasets import Dataset
from tqdm import tqdm

# -------------------------------
# Paths
# -------------------------------
RAG_CORPUS_JSON = "data/rag_corpus/rag_passages.json"
INDEX_DIR = "data/rag_index"
DATASET_PATH = os.path.join(INDEX_DIR, "my_dataset")
INDEX_PATH = os.path.join(INDEX_DIR, "hf_dataset_index.faiss")

os.makedirs(INDEX_DIR, exist_ok=True)

# -------------------------------
# 1. Load dataset
# -------------------------------
with open(RAG_CORPUS_JSON, "r", encoding="utf-8") as f:
    passages = json.load(f)

dataset = Dataset.from_list(passages)

# Ensure correct schema (CRITICAL for RAG)
if "text" not in dataset.column_names:
    raise ValueError("Dataset must contain 'text' column")

if "title" not in dataset.column_names:
    print("⚠️ 'title' column missing. Adding empty titles...")
    dataset = dataset.map(lambda x: {"title": ""})

print(f"✅ Loaded {len(dataset)} passages.")
print(f"Columns: {dataset.column_names}")

# -------------------------------
# 2. Load DPR encoder
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)

encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
).to(device)

encoder.eval()

# -------------------------------
# 3. Embedding function
# -------------------------------
def embed(batch):
    inputs = tokenizer(
        batch["text"],
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = encoder(**inputs)
        embeddings = outputs.pooler_output

    return {
        "embeddings": embeddings.cpu().numpy().astype(np.float32)
    }

# -------------------------------
# 4. Generate embeddings
# -------------------------------
print("🚀 Generating embeddings (GPU)...")

dataset = dataset.map(
    embed,
    batched=True,
    batch_size=64,   # safe for RTX A5000
    desc="Embedding passages"
)

print("✅ Embeddings generated.")

# -------------------------------
# 5. Build FAISS index
# -------------------------------
print("🔎 Building FAISS index...")

dataset.add_faiss_index(
    column="embeddings"
)

print("✅ FAISS index built.")

# -------------------------------
# 6. Save FAISS index FIRST
# -------------------------------
dataset.get_index("embeddings").save(INDEX_PATH)

print(f"✅ FAISS index saved at: {INDEX_PATH}")

# -------------------------------
# 7. Remove index BEFORE saving dataset
# -------------------------------
# IMPORTANT: do NOT assign (returns None in HF datasets)
dataset.drop_index("embeddings")

# -------------------------------
# 8. Save dataset
# -------------------------------
dataset.save_to_disk(DATASET_PATH)

print(f"✅ Dataset saved at: {DATASET_PATH}")

# -------------------------------
# DONE
# -------------------------------
print("\n🎉 RAG index build complete!")