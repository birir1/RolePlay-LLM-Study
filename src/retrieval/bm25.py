"""
SAF-RAG BM25 Retriever (Uncertainty-Aware Version)

Upgrades:
- lexical uncertainty estimation
- entropy-based retrieval confidence
- normalized score compatibility with dense retriever
- hybrid-ready + SAF-v2 compatible output format
"""

import os
import pickle
import numpy as np
import logging
import re
from typing import Dict, List, Optional, Any
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    SAF-aware BM25 retriever with calibrated uncertainty signals.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        index_path: Optional[str] = None
    ):
        self.k1 = k1
        self.b = b
        self.index_path = index_path

        self.corpus: List[str] = []
        self.corpus_metadata: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25 = None

        if index_path and os.path.exists(index_path):
            self._load_index()

    # =========================================================
    # INDEXING
    # =========================================================
    def index_corpus(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        tokenizer=None
    ) -> None:

        self.corpus = [d for d in documents if isinstance(d, str) and d.strip()]
        self.corpus_metadata = metadata or [{"id": i} for i in range(len(self.corpus))]

        tokenizer = tokenizer or self._default_tokenizer

        self.tokenized_corpus = [tokenizer(doc) for doc in self.corpus]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )

        logger.info(f"[BM25] Indexed {len(self.corpus)} documents")

    # =========================================================
    # RETRIEVAL
    # =========================================================
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        tokenizer=None
    ) -> List[Dict[str, Any]]:

        if self.bm25 is None:
            raise RuntimeError("BM25 index not initialized. Call index_corpus().")

        tokenizer = tokenizer or self._default_tokenizer
        query_tokens = tokenizer(query)

        scores = self.bm25.get_scores(query_tokens)

        if len(scores) == 0:
            return []

        # =====================================================
        # NORMALIZATION (CRITICAL FOR SAF FUSION)
        # =====================================================
        norm_scores = self._minmax(scores)

        mean_score = float(np.mean(norm_scores))
        std_score = float(np.std(norm_scores) + 1e-8)

        # =====================================================
        # PROBABILITY DISTRIBUTION (FOR ENTROPY)
        # =====================================================
        probs = norm_scores + 1e-8
        probs = probs / np.sum(probs)

        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(probs) + 1e-8)

        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
        sparsity = 1.0 - entropy_norm

        # =====================================================
        # TOP-K SELECTION
        # =====================================================
        top_indices = np.argsort(norm_scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []

        for idx in top_indices:

            score = float(norm_scores[idx])

            # =================================================
            # SAF v2 CALIBRATED SIGNALS
            # =================================================
            lexical_confidence = score
            lexical_uncertainty = (1.0 - score) + std_score
            instability = std_score * (1.0 - score)

            results.append({
                "text": self.corpus[idx],
                "score": score,
                "source": "bm25",

                # SAF signals (v2 aligned)
                "confidence": float(lexical_confidence),
                "uncertainty": float(lexical_uncertainty),
                "instability": float(instability),

                # distribution signals
                "entropy": float(entropy),
                "entropy_norm": float(entropy_norm),
                "sparsity": float(sparsity),
                "score_mean": mean_score,
                "score_std": std_score,

                # metadata
                "metadata": self.corpus_metadata[idx],
            })

        return results

    # =========================================================
    # BATCH RETRIEVAL
    # =========================================================
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        tokenizer=None
    ) -> List[List[Dict[str, Any]]]:

        return [self.retrieve(q, top_k, tokenizer) for q in queries]

    # =========================================================
    # NORMALIZATION
    # =========================================================
    def _minmax(self, scores: np.ndarray) -> np.ndarray:

        min_s = float(scores.min())
        max_s = float(scores.max())

        if abs(max_s - min_s) < 1e-8:
            return np.zeros_like(scores)

        return (scores - min_s) / (max_s - min_s + 1e-8)

    # =========================================================
    # TOKENIZER
    # =========================================================
    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:

        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if t]

    # =========================================================
    # INDEX LOADING
    # =========================================================
    def _load_index(self) -> None:

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        self.corpus = data["corpus"]
        self.corpus_metadata = data["corpus_metadata"]
        self.tokenized_corpus = data["tokenized_corpus"]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )

        logger.info(f"[BM25] Loaded index: {len(self.corpus)} docs")

    # =========================================================
    # STATS
    # =========================================================
    def stats(self) -> Dict[str, Any]:

        if not self.corpus:
            return {}

        lengths = [len(t) for t in self.tokenized_corpus]

        return {
            "num_documents": len(self.corpus),
            "avg_doc_length": float(np.mean(lengths)),
            "std_doc_length": float(np.std(lengths)),
            "total_tokens": int(sum(lengths)),
            "k1": self.k1,
            "b": self.b
        }