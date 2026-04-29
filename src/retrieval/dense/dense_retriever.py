import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DenseRetriever:
    """
    SAF-RAG Dense Retriever (Research Version)

    Enhancements:
    - Returns confidence + uncertainty signals
    - Stable normalization
    - Query-document similarity distribution stats
    - Fusion-ready output format
    """

    def __init__(
        self,
        corpus: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64
    ):
        self.device = device
        self.batch_size = batch_size

        self.corpus = [
            c.strip() for c in corpus
            if isinstance(c, str) and c.strip()
        ]

        if len(self.corpus) == 0:
            raise ValueError("DenseRetriever received empty corpus.")

        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

        self.doc_embeddings = self._encode_corpus(self.corpus)

    # =========================================================
    # ENCODING
    # =========================================================
    def _encode_corpus(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def _encode_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            query = "empty query"

        return self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    # =========================================================
    # CORE RETRIEVAL (WITH UNCERTAINTY SIGNALS)
    # =========================================================
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:

        q_emb = self._encode_query(query)

        sims = cosine_similarity(q_emb, self.doc_embeddings)[0]

        # -----------------------------
        # Normalize similarity scores
        # -----------------------------
        sims_norm = self._minmax_normalize(sims)

        # -----------------------------
        # Uncertainty signals (IMPORTANT FOR SAF)
        # -----------------------------
        mean_sim = np.mean(sims_norm)
        std_sim = np.std(sims_norm)

        # entropy-style uncertainty
        eps = 1e-8
        probs = sims_norm / (np.sum(sims_norm) + eps)
        entropy = -np.sum(probs * np.log(probs + eps))

        # -----------------------------
        # Ranking
        # -----------------------------
        top_indices = np.argsort(sims_norm)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.corpus[idx],
                "dense_score": float(sims_norm[idx]),
                "query_similarity_mean": float(mean_sim),
                "query_similarity_std": float(std_sim),
                "retrieval_entropy": float(entropy)
            })

        return results

    # =========================================================
    # NORMALIZATION
    # =========================================================
    def _minmax_normalize(self, scores: np.ndarray) -> np.ndarray:
        min_s = scores.min()
        max_s = scores.max()

        if max_s - min_s < 1e-8:
            return np.zeros_like(scores)

        return (scores - min_s) / (max_s - min_s)

    # =========================================================
    # CORPUS UPDATE (SAFE EXTENSION)
    # =========================================================
    def add_documents(self, new_docs: List[str]):
        cleaned = [
            d.strip() for d in new_docs
            if isinstance(d, str) and d.strip()
        ]

        if not cleaned:
            return

        new_embs = self._encode_corpus(cleaned)

        self.corpus.extend(cleaned)
        self.doc_embeddings = np.vstack([self.doc_embeddings, new_embs])