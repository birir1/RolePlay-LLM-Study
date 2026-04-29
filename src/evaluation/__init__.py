"""
Evaluation package: unified exports for all evaluation modules.

This module exposes BOTH:
1. Pipeline-friendly functions (used in simple evaluation scripts)
2. Class-based evaluators (used in advanced Phase 3 evaluation)
"""

# =========================
# FUNCTION-LEVEL EXPORTS (PIPELINE)
# =========================
from .persona_drift import compute_persona_drift
from .role_consistency import compute_role_consistency
from .hallucination_check import compute_hallucination_rate
from .sycophancy_metrics import compute_sycophancy_score


# =========================
# CLASS-LEVEL EXPORTS (ADVANCED EVALUATION)
# =========================
from .persona_drift import PersonaDriftDetector
from .role_consistency import RoleConsistencyEvaluator
from .hallucination_check import HallucinationDetector
from .sycophancy_metrics import SycophancyMetrics
from .faithfulness_scorer import FaithfulnessScorer, FaithfulnessComparator
from .evaluator import Evaluator, EvaluationConfig


# =========================
# CLEAN EXPORT API
# =========================
__all__ = [
    # --- functions ---
    "compute_persona_drift",
    "compute_role_consistency",
    "compute_hallucination_rate",
    "compute_sycophancy_score",

    # --- classes ---
    "PersonaDriftDetector",
    "RoleConsistencyEvaluator",
    "HallucinationDetector",
    "SycophancyMetrics",
    "FaithfulnessScorer",
    "FaithfulnessComparator",
    "Evaluator",
    "EvaluationConfig",
]