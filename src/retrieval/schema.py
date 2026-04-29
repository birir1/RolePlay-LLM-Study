from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RetrievalResult:
    """
    Unified retrieval output format for ALL retrievers.

    This prevents:
    - KeyError crashes
    - fusion mismatches
    - SAF instability
    """

    text: str
    score: float
    source: str  # "bm25" | "dense" | "sparse" | "fusion"

    metadata: Dict[str, Any] = None

    # SAF signals (optional but standardized)
    uncertainty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": float(self.score),
            "source": self.source,
            "metadata": self.metadata or {},
            "uncertainty": float(self.uncertainty),
        }