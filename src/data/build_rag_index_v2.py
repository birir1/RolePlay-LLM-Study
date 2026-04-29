import os
import json
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from datasets import Dataset

# -------------------------------
# Paths
# -------------------------------
RAG_CORPUS_JSON = "data/rag_corpus/rag_passages.json"
INDEX_DIR = "data/rag_index"
DATASET_PATH = os.path.join(INDEX_DIR, "my_dataset")
INDEX_PATH = os.path.join(INDEX_DIR, "hf_dataset_index.faiss")
os.makedirs(INDEX_DIR, exist_ok=True)

# -------------------------------
# 1. Load passages & create Dataset
# -------------------------------
with open(RAG_CORPUS_JSON, "r", encoding="utf-8") as f:
    passages = json.load(f)

dataset = Dataset.from_list(passages)  # expects [{"title": "...", "text": "..."}]
print(f"Loaded {len(dataset)} passages.")

# -------------------------------
# 2. Generate embeddings (DPR)
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)

def embed(batch):
    inputs = ctx_tokenizer(batch["text"], truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = ctx_encoder(**inputs).pooler_output
    return {"embeddings": embeddings.cpu().numpy()}

print("Generating embeddings. This may take a while...")
dataset = dataset.map(embed, batched=True, batch_size=128, remove_columns=dataset.column_names)

# -------------------------------
# 3. Build & Save FAISS index
# -------------------------------
dataset.add_faiss_index(column="embeddings")

# Drop FAISS index before saving dataset to disk
dataset_no_index = dataset.drop_index("embeddings")
dataset_no_index.save_to_disk(DATASET_PATH)
print(f"Dataset saved at {DATASET_PATH} (without FAISS index).")

# Save FAISS index separately
dataset.get_index("embeddings").save(INDEX_PATH)
print(f"FAISS index built and saved at {INDEX_PATH}.")