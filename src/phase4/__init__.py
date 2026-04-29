"""
Phase 4: SAF-RAG System

Lightweight init to avoid circular imports and unnecessary dependencies.
Only expose stable, standalone components.
"""

# =========================
# SAFE CORE COMPONENTS
# =========================
from .risk_controller import RiskController
from .sycophancy_guard import SycophancyGuard
from .grounding_scorer import GroundingScorer

# =========================
# EXPORTS
# =========================
__all__ = [
    "RiskController",
    "SycophancyGuard",
    "GroundingScorer",
]