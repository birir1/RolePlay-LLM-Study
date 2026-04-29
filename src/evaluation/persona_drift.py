import numpy as np
from typing import Dict, List, Optional
import logging
import torch

logger = logging.getLogger(__name__)


class PersonaDriftDetector:
    """
    Measures role/persona consistency throughout conversations.
    """

    def __init__(self, device: Optional[str] = None):
        # =========================
        # DEVICE SETUP (FIXED)
        # =========================
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"[PersonaDrift] Using device: {self.device}")

        # =========================
        # LOAD EMBEDDING MODEL (GPU ENABLED)
        # =========================
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading semantic similarity model...")
            self.similarity_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device=self.device
            )

            self.use_embeddings = True

        except Exception as e:
            logger.warning(f"Embedding model failed to load: {e}")
            logger.warning("Falling back to heuristic mode.")
            self.use_embeddings = False

    # =========================
    # CORE PIPELINE FUNCTION
    # =========================
    def compute_batch_persona_drift(self, predictions: List[str]) -> float:
        """
        Compute drift over a batch of predictions (pipeline-compatible).
        """
        if len(predictions) < 2:
            return 0.0

        # -------------------------
        # FALLBACK MODE
        # -------------------------
        if not self.use_embeddings:
            lengths = [len(p.split()) for p in predictions]
            return float(np.std(lengths) / (np.mean(lengths) + 1e-8))

        # -------------------------
        # EMBEDDING MODE (GPU)
        # -------------------------
        embeddings = self.similarity_model.encode(
            predictions,
            convert_to_numpy=True,
            batch_size=64,          # ✅ faster GPU batching
            show_progress_bar=False
        )

        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(embeddings[:-1], embeddings[1:])
        similarities = np.diag(similarities)

        drift_scores = 1.0 - similarities

        return float(np.mean(drift_scores))

    # =========================
    # ROLE DRIFT INDEX
    # =========================
    def compute_role_drift_index(
        self,
        initial_persona: str,
        conversation_turns: List[str],
        window_size: int = 3
    ) -> Dict:

        if not self.use_embeddings:
            return {
                "role_drift_index": self.compute_batch_persona_drift(conversation_turns),
                "drift_per_turn": [],
                "cumulative_drift": [],
                "has_high_drift": False,
                "num_turns": len(conversation_turns)
            }

        from sklearn.metrics.pairwise import cosine_similarity

        initial_embedding = self.similarity_model.encode(
            initial_persona,
            convert_to_numpy=True
        )

        turn_embeddings = self.similarity_model.encode(
            conversation_turns,
            convert_to_numpy=True
        )

        similarities = cosine_similarity(
            turn_embeddings,
            initial_embedding.reshape(1, -1)
        ).flatten()

        drift_per_turn = 1.0 - similarities
        overall_rdi = float(np.mean(drift_per_turn))

        cumulative_drift = np.cumsum(np.diff([1.0] + list(similarities)))

        return {
            "role_drift_index": overall_rdi,
            "drift_per_turn": drift_per_turn.tolist(),
            "cumulative_drift": cumulative_drift.tolist(),
            "max_drift_turn": int(np.argmax(drift_per_turn)),
            "has_high_drift": overall_rdi > 0.3,
            "num_turns": len(conversation_turns)
        }

    # =========================
    # PERSONA CONSISTENCY
    # =========================
    def compute_persona_consistency(
        self,
        persona_traits: Dict[str, str],
        conversation_turns: List[str]
    ) -> Dict:

        if not self.use_embeddings:
            return {
                "persona_consistency_score": 0.5,
                "trait_scores": {},
                "consistent_traits": [],
                "inconsistent_traits": []
            }

        from sklearn.metrics.pairwise import cosine_similarity

        turn_embeddings = self.similarity_model.encode(
            conversation_turns,
            convert_to_numpy=True
        )

        trait_scores = {}

        for trait_name, trait_description in persona_traits.items():
            trait_embedding = self.similarity_model.encode(
                trait_description,
                convert_to_numpy=True
            )

            similarities = cosine_similarity(
                turn_embeddings,
                trait_embedding.reshape(1, -1)
            ).flatten()

            trait_scores[trait_name] = float(np.mean(similarities))

        pcs = float(np.mean(list(trait_scores.values())))

        return {
            "persona_consistency_score": pcs,
            "trait_scores": trait_scores,
            "consistent_traits": [t for t, s in trait_scores.items() if s > 0.6],
            "inconsistent_traits": [t for t, s in trait_scores.items() if s < 0.4],
            "is_consistent": pcs > 0.6
        }

    # =========================
    # CHARACTER ADHERENCE
    # =========================
    def compute_character_adherence(
        self,
        character_instructions: str,
        conversation_turns: List[str],
        expected_keywords: Optional[List[str]] = None
    ) -> Dict:

        if not self.use_embeddings:
            return {
                "character_adherence_score": 0.5,
                "instruction_following": 0.5,
                "keyword_coverage": 0.0,
                "follows_character": False
            }

        from sklearn.metrics.pairwise import cosine_similarity

        instruction_embedding = self.similarity_model.encode(
            character_instructions,
            convert_to_numpy=True
        )

        turn_embeddings = self.similarity_model.encode(
            conversation_turns,
            convert_to_numpy=True
        )

        similarities = cosine_similarity(
            turn_embeddings,
            instruction_embedding.reshape(1, -1)
        ).flatten()

        instruction_following = float(np.mean(similarities))

        keyword_coverage = 1.0
        if expected_keywords:
            text = " ".join(conversation_turns).lower()
            matches = sum(1 for k in expected_keywords if k.lower() in text)
            keyword_coverage = matches / len(expected_keywords)

        ca = 0.7 * instruction_following + 0.3 * keyword_coverage

        return {
            "character_adherence_score": float(ca),
            "instruction_following": float(instruction_following),
            "keyword_coverage": float(keyword_coverage),
            "follows_character": ca > 0.6
        }


# =========================
# PIPELINE WRAPPER (FIXED)
# =========================
_detector_instance = None


def compute_persona_drift(predictions: List[str]) -> float:
    """
    Pipeline-compatible function used in evaluation.
    Uses singleton to avoid reloading model repeatedly.
    """
    global _detector_instance

    if _detector_instance is None:
        _detector_instance = PersonaDriftDetector()

    return _detector_instance.compute_batch_persona_drift(predictions)