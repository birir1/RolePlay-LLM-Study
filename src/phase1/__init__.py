"""Phase 1: SAF-RAG core systems.

This package exposes the sycophancy-aware fusion modules used in Phase 1.
"""

from src.models.saf_rag import (
    SAFRAG,
    SAFRAGConfig,
    SycophancyDetector,
    RiskEstimator,
    FusionGate,
)

__all__ = [
    "SAFRAG",
    "SAFRAGConfig",
    "SycophancyDetector",
    "RiskEstimator",
    "FusionGate",
]
