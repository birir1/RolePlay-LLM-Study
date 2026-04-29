import numpy as np
from typing import List, Dict, Union

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class GroundingScorer:
    """
    Robust Grounding Scorer for SAF-RAG

    Fixes:
    - embedding normalization (critical)
    - doc encoding cache (major speedup)
    - adaptive support threshold
    - softer consistency penalty
    - stable multi-doc aggregation
    """

    def __init__(self, device="cpu"):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=device
        )

        # 🔥 cache to avoid recomputing doc embeddings
        self.doc_cache: Dict[str, np.ndarray] = {}

    # =========================
    # SAFE ENCODE (WITH CACHE)
    # =========================
    def _encode_docs(self, docs: List[str]) -> np.ndarray:
        embeddings = []

        for d in docs:
            key = d.strip()

            if key in self.doc_cache:
                embeddings.append(self.doc_cache[key])
            else:
                emb = self.model.encode(
                    [key],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )[0]

                self.doc_cache[key] = emb
                embeddings.append(emb)

        return np.array(embeddings)

    # =========================
    # CORE SCORE (STABLE)
    # =========================
    def score(
        self,
        query: str,
        prediction: str,
        docs: List[Union[str, Dict]]
    ) -> float:

        # -------------------------
        # Safety checks
        # -------------------------
        if not prediction or not str(prediction).strip():
            return 0.0

        if not docs:
            return 0.0

        # normalize docs → list[str]
        clean_docs = []
        for d in docs:
            if isinstance(d, str):
                clean_docs.append(d.strip())
            elif isinstance(d, dict) and "text" in d:
                clean_docs.append(str(d["text"]).strip())

        clean_docs = [d for d in clean_docs if d]

        if not clean_docs:
            return 0.0

        # -------------------------
        # Encode (normalized embeddings)
        # -------------------------
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        p_emb = self.model.encode(
            [prediction],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        d_emb = self._encode_docs(clean_docs)

        # -------------------------
        # Similarities
        # -------------------------
        q_sim = cosine_similarity(p_emb, q_emb)[0][0]
        doc_sims = cosine_similarity(p_emb, d_emb)[0]

        # -------------------------
        # Core stats
        # -------------------------
        max_sim = float(np.max(doc_sims))
        mean_sim = float(np.mean(doc_sims))
        std_sim = float(np.std(doc_sims))

        # -------------------------
        # 🔥 ADAPTIVE SUPPORT THRESHOLD
        # -------------------------
        # instead of fixed 0.5 → dynamic
        threshold = max(0.4, mean_sim)

        supporting_docs = np.sum(doc_sims >= threshold)
        coverage = supporting_docs / (len(doc_sims) + 1e-8)

        # -------------------------
        # 🔥 SOFT CONSISTENCY PENALTY
        # -------------------------
        # prevent over-penalizing multi-hop answers
        consistency_penalty = std_sim * 0.5

        # -------------------------
        # 🔥 QUERY ALIGNMENT (CLIPPED)
        # -------------------------
        query_alignment = float(np.clip(q_sim, 0.0, 1.0))

        # -------------------------
        # 🔥 FINAL GROUNDING (REBALANCED)
        # -------------------------
        grounding = (
            0.5 * max_sim +
            0.2 * mean_sim +
            0.15 * coverage +
            0.15 * query_alignment
            - 0.15 * consistency_penalty
        )

        return float(np.clip(grounding, 0.0, 1.0))

    # =========================
    # BATCH SCORE (SAFE)
    # =========================
    def batch_score(
        self,
        queries: List[str],
        predictions: List[str],
        docs_list: List[List[Union[str, Dict]]]
    ) -> List[float]:

        scores = []

        for q, pred, docs in zip(queries, predictions, docs_list):
            try:
                scores.append(self.score(q, pred, docs))
            except Exception:
                scores.append(0.0)

        return scores

    # =========================
    # CONFIDENCE (CALIBRATED)
    # =========================
    def confidence(
        self,
        query: str,
        prediction: str,
        docs: List[Union[str, Dict]]
    ) -> float:

        score = self.score(query, prediction, docs)

        # 🔥 smoother than your previous sigmoid
        # avoids collapsing mid-range scores
        return float(1 / (1 + np.exp(-4 * (score - 0.5))))