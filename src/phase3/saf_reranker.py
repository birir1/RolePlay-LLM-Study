from typing import List, Dict, Optional
import numpy as np

from src.retrieval.reranker import CrossEncoderReranker


class SAFReranker:
    """
    SAF (Self-Aware Filtering) Reranker

    Goal:
    Improve generation reliability by combining:
    - semantic relevance (CrossEncoder)
    - hallucination risk penalty
    - faithfulness score
    - optional sycophancy / drift penalties

    Output:
    Best-ranked candidate response
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 16,
        alpha_relevance: float = 1.0,
        beta_hallucination: float = 0.6,
        gamma_faithfulness: float = 0.8,
        delta_drift: float = 0.4
    ):
        self.reranker = CrossEncoderReranker(
            device=device,
            batch_size=batch_size
        )

        # weighting factors
        self.alpha = alpha_relevance
        self.beta = beta_hallucination
        self.gamma = gamma_faithfulness
        self.delta = delta_drift

    # =========================================================
    # SCORE A SINGLE CANDIDATE
    # =========================================================
    def compute_score(self, candidate: Dict) -> float:
        """
        SAF scoring function

        Higher is better
        """

        relevance = candidate.get("reranker_score", 0.5)

        hallucination = candidate.get("hallucination_rate", 0.0)
        faithfulness = candidate.get("faithfulness", 0.5)
        drift = candidate.get("persona_drift", 0.0)

        # -------------------------------
        # SAF scoring equation
        # -------------------------------
        score = (
            self.alpha * relevance +
            self.gamma * faithfulness -
            self.beta * hallucination -
            self.delta * drift
        )

        return float(score)

    # =========================================================
    # RERANK CANDIDATES
    # =========================================================
    def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict],
        text_key: str = "text"
    ) -> List[Dict]:

        if not candidates:
            return []

        # -------------------------------
        # STEP 1: semantic rerank
        # -------------------------------
        pairs = [(query, c.get(text_key, "")) for c in candidates]
        scores = self.reranker.score_pairs(pairs)

        for c, s in zip(candidates, scores):
            c["reranker_score"] = float(s)

        # -------------------------------
        # STEP 2: SAF scoring
        # -------------------------------
        for c in candidates:
            c["saf_score"] = self.compute_score(c)

        # -------------------------------
        # STEP 3: FINAL SORT
        # -------------------------------
        candidates.sort(key=lambda x: x["saf_score"], reverse=True)

        return candidates

    # =========================================================
    # SELECT BEST RESPONSE
    # =========================================================
    def select_best(
        self,
        query: str,
        candidates: List[Dict],
        text_key: str = "text"
    ) -> Optional[Dict]:

        ranked = self.rerank_candidates(query, candidates, text_key)

        if not ranked:
            return None

        return ranked[0]

    # =========================================================
    # DEBUG VIEW
    # =========================================================
    def explain_ranking(self, candidate: Dict) -> Dict:
        """
        Useful for debugging / paper results
        """

        return {
            "reranker_score": candidate.get("reranker_score"),
            "hallucination": candidate.get("hallucination_rate"),
            "faithfulness": candidate.get("faithfulness"),
            "drift": candidate.get("persona_drift"),
            "final_saf_score": candidate.get("saf_score")
        }