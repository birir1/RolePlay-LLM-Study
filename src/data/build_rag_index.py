import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
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
# 1. Load passages
# -------------------------------
with open(RAG_CORPUS_JSON, "r", encoding="utf-8") as f:
    passages = json.load(f)

dataset = Dataset.from_list(passages)

# -------------------------------
# 2. Ensure RAG schema
# -------------------------------
if "text" not in dataset.column_names:
    raise ValueError("Dataset MUST contain 'text' column")

if "title" not in dataset.column_names:
    print("⚠️ Adding missing 'title' column...")
    dataset = dataset.map(lambda x: {"title": ""})

print(f"✅ Loaded {len(dataset)} passages")
print(f"Columns: {dataset.column_names}")

# -------------------------------
# 3. Load DPR encoder
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)

ctx_encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
).to(device)

ctx_encoder.eval()

# -------------------------------
# 4. Embedding function
# -------------------------------
def embed(batch):
    inputs = ctx_tokenizer(
        batch["text"],
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = ctx_encoder(**inputs)
        embeddings = outputs.pooler_output

    return {
        "embeddings": embeddings.cpu().numpy().astype(np.float32)
    }

# -------------------------------
# 5. Generate embeddings
# -------------------------------
print("🚀 Generating embeddings...")
dataset = dataset.map(
    embed,
    batched=True,
    batch_size=64,
    desc="Embedding passages"
)

print("✅ Embeddings complete")

# -------------------------------
# 6. Build FAISS index
# -------------------------------
print("🔎 Building FAISS index...")
dataset.add_faiss_index(column="embeddings")

print("✅ FAISS index built")

# -------------------------------
# 7. Save FAISS index FIRST
# -------------------------------
dataset.get_index("embeddings").save(INDEX_PATH)
print(f"✅ Index saved at {INDEX_PATH}")

# -------------------------------
# 8. Remove index BEFORE saving dataset
# -------------------------------
dataset.drop_index("embeddings")   # ⚠️ DO NOT ASSIGN

# -------------------------------
# 9. Save dataset
# -------------------------------
dataset.save_to_disk(DATASET_PATH)

print(f"✅ Dataset saved at {DATASET_PATH}")

print("\n🎉 RAG index build SUCCESSFUL")