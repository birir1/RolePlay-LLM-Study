"""
Phase 3: Evaluation modules

This package contains:
- Semantic evaluation metrics (hallucination, faithfulness, etc.)
- RAG evaluation pipeline (RAGEvaluator)
"""

# ✅ New evaluation core
from src.phase3.evaluators.rag_evaluator import RAGEvaluator

# ✅ Metrics (optional exposure)
from src.phase3.metrics.hallucination import hallucination_score
from src.phase3.metrics.faithfulness import faithfulness_score
from src.phase3.metrics.sycophancy import sycophancy_score
from src.phase3.metrics.persona_drift import persona_drift_score


__all__ = [
    "RAGEvaluator",
    "hallucination_score",
    "faithfulness_score",
    "sycophancy_score",
    "persona_drift_score",
]