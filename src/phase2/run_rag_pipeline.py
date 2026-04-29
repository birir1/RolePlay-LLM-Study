import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTEXT_TOKENS = 400
MAX_INPUT_TOKENS = 512


# =========================
# DATA LOADER (ROBUST)
# =========================
def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        # JSONL fallback
        data = [json.loads(line) for line in content.split("\n") if line.strip()]

    print(f"[INFO] Loaded {len(data)} samples from {path}")
    return data


# =========================
# CORPUS BUILDER
# =========================
def build_corpus(data: List[Dict]) -> List[str]:
    corpus = []

    for item in data:
        # Primary fields
        for key in ["context", "documents", "input", "output", "instruction"]:
            if key in item:
                if isinstance(item[key], list):
                    corpus.extend(item[key])
                elif isinstance(item[key], str):
                    corpus.append(item[key])

        # Fallback: long text fields
        for value in item.values():
            if isinstance(value, str) and len(value) > 80:
                corpus.append(value)

    # Clean
    corpus = list(set([c.strip() for c in corpus if isinstance(c, str) and c.strip()]))

    print(f"[INFO] Built corpus with {len(corpus)} documents")

    if not corpus:
        raise ValueError("[ERROR] Corpus is empty. Dataset format not supported.")

    return corpus


# =========================
# QUERY EXTRACTOR
# =========================
def extract_query(item: Dict) -> str:
    for key in ["query", "question", "input", "instruction"]:
        if key in item and isinstance(item[key], str):
            return item[key]

    for v in item.values():
        if isinstance(v, str) and len(v) > 20:
            return v

    return ""


# =========================
# RETRIEVER (IMPROVED)
# =========================
class SemanticRetriever:
    def __init__(self, documents: List[str]):
        self.documents = documents

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=DEVICE
        )

        print("[INFO] Encoding corpus...")
        self.doc_embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=128,
            normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        sims = cosine_similarity(query_emb, self.doc_embeddings)[0]

        top_indices = np.argsort(sims)[::-1][:top_k]

        return [self.documents[i] for i in top_indices]


# =========================
# GENERATOR (STRICT GROUNDED)
# =========================
class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        print(f"[INFO] Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

    def truncate_context(self, context: str) -> str:
        tokens = self.tokenizer.encode(context, truncation=True, max_length=MAX_CONTEXT_TOKENS)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate(self, query: str, context: str) -> str:
        context = self.truncate_context(context)

        prompt = f"""You are a STRICT factual QA system.

Rules:
- Use ONLY the provided context
- If answer is NOT explicitly stated → say "I don't know"
- Do NOT infer
- Do NOT hallucinate
- Answer must be grounded in context

Context:
{context}

Question:
{query}

Answer:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS
        ).to(DEVICE)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            num_beams=4,   # 🔥 improves factuality
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# =========================
# PIPELINE
# =========================
def run_pipeline(dataset_path: str, output_path: str, top_k: int = 3):
    data = load_dataset(dataset_path)

    corpus = build_corpus(data)

    retriever = SemanticRetriever(corpus)
    generator = Generator()

    results = []

    print("[INFO] Running RAG pipeline...")

    for item in tqdm(data):
        query = extract_query(item)

        if not query:
            continue

        retrieved_docs = retriever.retrieve(query, top_k=top_k)
        context = " ".join(retrieved_docs)

        # Fallback: empty context guard
        if not context.strip():
            prediction = "I don't know"
        else:
            prediction = generator.generate(query, context)

        results.append({
            "query": query,
            "context": context,
            "prediction": prediction
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[SUCCESS] Saved → {output_path}")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top_k", type=int, default=3)

    args = parser.parse_args()

    run_pipeline(
        dataset_path=args.dataset,
        output_path=args.output,
        top_k=args.top_k
    )