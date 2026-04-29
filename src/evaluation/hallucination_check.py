import re
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# SINGLETON MODEL (CRITICAL FIX)
# =========================
_model_instance = None


def get_embedding_model(device="cuda"):
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=device
        )
    return _model_instance


# =========================
# HALLUCINATION DETECTOR
# =========================
class HallucinationDetector:
    """
    Advanced hallucination detection module.
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.model = get_embedding_model(device)

    # =========================
    # EVIDENCE GROUNDING
    # =========================
    def detect_evidence_hallucinations(
        self,
        answer: str,
        evidence_list: List[str],
        threshold: float = 0.6
    ) -> Dict:

        claims = self._split_into_claims(answer)

        if not claims or not evidence_list:
            return self._empty_result()

        evidence_text = " ".join(evidence_list)

        claim_emb = self.model.encode(claims, convert_to_numpy=True)
        evidence_emb = self.model.encode([evidence_text], convert_to_numpy=True)

        sims = cosine_similarity(claim_emb, evidence_emb).flatten()

        supported, unsupported = [], []

        for claim, score in zip(claims, sims):
            if score >= threshold:
                supported.append(claim)
            else:
                unsupported.append(claim)

        hallucination_rate = len(unsupported) / len(claims)

        return {
            "hallucination_rate": float(hallucination_rate),
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "num_claims": len(claims),
            "avg_similarity": float(np.mean(sims))
        }

    # =========================
    # SELF CONTRADICTIONS (IMPROVED)
    # =========================
    def detect_self_contradictions(self, answer: str) -> Dict:

        claims = self._split_into_claims(answer)

        if len(claims) < 2:
            return {
                "num_contradictions": 0,
                "has_self_contradictions": False,
                "self_contradictions": []
            }

        embeddings = self.model.encode(claims, convert_to_numpy=True)

        contradictions = []

        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]

                # Better heuristic:
                # VERY low similarity AND negation patterns
                if sim < 0.3 and self._is_negation_pair(claims[i], claims[j]):
                    contradictions.append((claims[i], claims[j]))

        return {
            "num_contradictions": len(contradictions),
            "has_self_contradictions": len(contradictions) > 0,
            "self_contradictions": contradictions
        }

    # =========================
    # FACTUALITY (GROUND TRUTH)
    # =========================
    def detect_factuality_violations(
        self,
        answer: str,
        ground_truth: str,
        threshold: float = 0.65
    ) -> Dict:

        if not answer or not ground_truth:
            return {
                "factuality_score": 0.0,
                "hallucination_count": 0,
                "contradicted": False
            }

        answer_emb = self.model.encode([answer], convert_to_numpy=True)
        gt_emb = self.model.encode([ground_truth], convert_to_numpy=True)

        similarity = cosine_similarity(answer_emb, gt_emb)[0][0]

        return {
            "factuality_score": float(similarity),
            "hallucination_count": int(similarity < threshold),
            "contradicted": similarity < threshold
        }

    # =========================
    # HELPERS
    # =========================
    def _split_into_claims(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)

        return [
            s.strip()
            for s in sentences
            if len(s.strip().split()) >= 4
        ]

    def _is_negation_pair(self, a: str, b: str) -> bool:
        neg_words = ["not", "no", "never", "none", "cannot", "n't"]

        a_neg = any(w in a.lower() for w in neg_words)
        b_neg = any(w in b.lower() for w in neg_words)

        return a_neg != b_neg  # one negated, one not

    def _empty_result(self):
        return {
            "hallucination_rate": 0.0,
            "supported_claims": [],
            "unsupported_claims": [],
            "num_claims": 0,
            "avg_similarity": 0.0
        }


# =========================
# PIPELINE WRAPPER (FIXED)
# =========================
_detector_instance = None


def compute_hallucination_rate(
    predictions: List[str],
    contexts: Optional[List[str]] = None
) -> float:
    """
    Dataset-level hallucination rate.

    NOTE:
    - contexts = retrieved evidence (NOT ground truth)
    """

    global _detector_instance

    if _detector_instance is None:
        _detector_instance = HallucinationDetector()

    if contexts is None:
        # fallback → no grounding available
        return 0.0

    total = 0
    hallucinated = 0

    for pred, ctx in zip(predictions, contexts):
        if not ctx:
            continue

        result = _detector_instance.detect_evidence_hallucinations(
            pred,
            [ctx]
        )

        hallucinated += result["hallucination_rate"]
        total += 1

    return hallucinated / total if total > 0 else 0.0