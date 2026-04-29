import numpy as np
import re
from typing import List, Dict
from rank_bm25 import BM25Okapi


class SparseRetriever:
    """
    SAF-RAG Sparse Retriever (BM25 + Uncertainty Signals)

    Fixed:
    - correct input type
    - safe tokenization
    - robust normalization
    - stable entropy computation
    """

    def __init__(
        self,
        documents: List[str],   # ✅ FIXED TYPE
        k1: float = 1.5,
        b: float = 0.75
    ):

        # -------------------------
        # CLEAN RAW DOCS
        # -------------------------
        self.raw_docs = [
            str(d).strip()
            for d in documents
            if isinstance(d, (str, dict))
        ]

        if len(self.raw_docs) == 0:
            raise ValueError("SparseRetriever: empty corpus")

        # -------------------------
        # TOKENIZATION (SAFE)
        # -------------------------
        self.tokenized_docs = []

        for d in self.raw_docs:
            text = d["text"] if isinstance(d, dict) and "text" in d else d
            tokens = self._tokenize(text)

            # ✅ avoid empty token lists
            if len(tokens) == 0:
                tokens = ["empty"]

            self.tokenized_docs.append(tokens)

        # -------------------------
        # BM25 INIT
        # -------------------------
        self.bm25 = BM25Okapi(
            self.tokenized_docs,
            k1=k1,
            b=b
        )

    # =========================================================
    # TOKENIZATION
    # =========================================================
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    # =========================================================
    # RETRIEVAL
    # =========================================================
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:

        if not isinstance(query, str) or not query.strip():
            query = "empty query"

        query_tokens = self._tokenize(query)

        if len(query_tokens) == 0:
            query_tokens = ["empty"]

        scores = np.array(self.bm25.get_scores(query_tokens))

        # -------------------------
        # SAFE NORMALIZATION
        # -------------------------
        norm_scores = self._safe_minmax(scores)

        # -------------------------
        # STATS
        # -------------------------
        mean_score = float(np.mean(norm_scores))
        std_score = float(np.std(norm_scores) + 1e-8)

        # SAFE PROBABILITY DISTRIBUTION
        total = np.sum(norm_scores)

        if total < 1e-8:
            probs = np.ones_like(norm_scores) / len(norm_scores)
        else:
            probs = norm_scores / total

        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

        sparsity = float(1.0 - entropy / np.log(len(norm_scores) + 1e-8))

        # -------------------------
        # TOP-K
        # -------------------------
        top_indices = np.argsort(norm_scores)[::-1][:top_k]

        results = []

        for idx in top_indices:
            score = float(norm_scores[idx])

            uncertainty = (1.0 - score) + std_score
            confidence = score

            results.append({
                "text": self.raw_docs[idx],
                "sparse_score": score,

                # SAF signals
                "lexical_uncertainty": float(uncertainty),
                "lexical_confidence": float(confidence),

                # distribution-level signals
                "retrieval_entropy": entropy,
                "retrieval_sparsity": sparsity,
                "score_mean": mean_score,
                "score_std": std_score
            })

        return results

    # =========================================================
    # SAFE NORMALIZATION
    # =========================================================
    def _safe_minmax(self, scores: np.ndarray) -> np.ndarray:
        min_s = scores.min()
        max_s = scores.max()

        if max_s - min_s < 1e-8:
            # ✅ instead of zeros → uniform signal
            return np.ones_like(scores) * 0.5

        return (scores - min_s) / (max_s - min_s)