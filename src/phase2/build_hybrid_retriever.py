import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class HybridRetriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenized = [doc.split() for doc in corpus]

        print("[INFO] Building BM25...")
        self.bm25 = BM25Okapi(self.tokenized)

        print("[INFO] Loading embedding model...")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = self.embedder.encode(corpus, convert_to_numpy=True)

    def retrieve(self, query, top_k=5):
        # BM25
        bm25_scores = self.bm25.get_scores(query.split())

        # Dense
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        dense_scores = cosine_similarity(q_emb, self.embeddings)[0]

        # Normalize
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) + 1e-8)
        dense_scores = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) + 1e-8)

        # Combine
        scores = 0.5 * bm25_scores + 0.5 * dense_scores

        idx = np.argsort(scores)[::-1][:top_k]
        return [self.corpus[i] for i in idx]


def load_corpus(path):
    with open(path, "r") as f:
        data = json.load(f)

    return [d["context"] for d in data]


if __name__ == "__main__":
    corpus = load_corpus("data/processed/corpus.json")
    retriever = HybridRetriever(corpus)

    query = "What causes hallucinations in language models?"
    docs = retriever.retrieve(query)

    for d in docs:
        print(d[:200], "\n")