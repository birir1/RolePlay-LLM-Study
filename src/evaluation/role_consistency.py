import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class RoleConsistencyEvaluator:
    """
    Evaluates how consistent a model stays in its role across responses.
    """

    def __init__(self, device: str = None):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model for role consistency...")
            self.model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device=device if device else "cpu"
            )
            self.use_embeddings = True
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            logger.warning("Using fallback heuristic mode.")
            self.use_embeddings = False

    def compute_consistency(self, responses: List[str]) -> Dict:
        """
        Compute Role Consistency Score (RCS)

        Returns:
            {
                "role_consistency_score": float,
                "pairwise_similarities": List[float],
                "variance": float,
                "is_consistent": bool
            }
        """

        if len(responses) < 2:
            return {
                "role_consistency_score": 1.0,
                "pairwise_similarities": [],
                "variance": 0.0,
                "is_consistent": True
            }

        # =========================
        # FALLBACK MODE
        # =========================
        if not self.use_embeddings:
            lengths = [len(r.split()) for r in responses]
            variance = float(np.var(lengths))

            score = 1.0 / (1.0 + variance)

            return {
                "role_consistency_score": score,
                "pairwise_similarities": [],
                "variance": variance,
                "is_consistent": score > 0.5
            }

        # =========================
        # EMBEDDING MODE
        # =========================
        embeddings = self.model.encode(responses, convert_to_numpy=True)

        from sklearn.metrics.pairwise import cosine_similarity

        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0, 0]
            similarities.append(sim)

        similarities = np.array(similarities)

        score = float(np.mean(similarities))
        variance = float(np.var(similarities))

        return {
            "role_consistency_score": score,
            "pairwise_similarities": similarities.tolist(),
            "variance": variance,
            "is_consistent": score > 0.6
        }


# =========================
# PIPELINE FUNCTION (CRITICAL)
# =========================
def compute_role_consistency(predictions: List[str]) -> float:
    """
    Pipeline-compatible function.
    """
    evaluator = RoleConsistencyEvaluator()
    result = evaluator.compute_consistency(predictions)
    return result["role_consistency_score"]