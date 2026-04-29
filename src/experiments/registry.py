"""
Experiment Registry - Define all available experiments
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    
    name: str
    exp_type: Literal["llm", "rag", "saf-rag"]
    description: str
    
    # Model components
    detector: Optional[str] = None          # Phase 1 sycophancy detector
    retriever_type: str = "hybrid"          # Phase 2 retriever type
    generator_model: str = "facebook/bart-base"
    fusion_gate: Optional[str] = None       # Phase 1 fusion gate (None, "fixed", "learnable")
    
    # Flags
    use_sycophancy: bool = True
    use_reranker: bool = True
    skip_evaluation: bool = False
    
    # Ablation
    is_ablation: bool = False
    ablation_of: Optional[str] = None


class ExperimentRegistry:
    """Registry of all available experiments."""
    
    EXPERIMENTS: Dict[str, ExperimentConfig] = {}
    
    @classmethod
    def register(cls, config: ExperimentConfig) -> None:
        """Register an experiment."""
        cls.EXPERIMENTS[config.name] = config
        logger.debug(f"Registered experiment: {config.name}")
    
    @classmethod
    def get_experiment(cls, name: str) -> ExperimentConfig:
        """Get experiment by name."""
        if name not in cls.EXPERIMENTS:
            available = list(cls.EXPERIMENTS.keys())
            raise ValueError(
                f"Unknown experiment: {name}\n"
                f"Available: {', '.join(available)}"
            )
        return cls.EXPERIMENTS[name]
    
    @classmethod
    def list_experiments(cls) -> List[str]:
        """List all registered experiments."""
        return sorted(cls.EXPERIMENTS.keys())
    
    @classmethod
    def list_experiments_by_type(cls, exp_type: str) -> List[str]:
        """List experiments by type."""
        return [
            name for name, config in cls.EXPERIMENTS.items()
            if config.exp_type == exp_type
        ]
    
    @classmethod
    def list_ablations(cls) -> List[str]:
        """List all ablation experiments."""
        return [
            name for name, config in cls.EXPERIMENTS.items()
            if config.is_ablation
        ]


# ============================================================================
# CORE EXPERIMENTS
# ============================================================================

# Baseline: Vanilla LLM (no retrieval)
ExperimentRegistry.register(ExperimentConfig(
    name="baseline-llm",
    exp_type="llm",
    description="Vanilla LLM without retrieval (GPT-2 baseline)",
    detector=None,
    fusion_gate=None,
    use_sycophancy=False,
    skip_evaluation=False
))

# Baseline: Standard RAG (retrieve + generate)
ExperimentRegistry.register(ExperimentConfig(
    name="vanilla-rag",
    exp_type="rag",
    description="Standard RAG: retrieve + generate (no safety features)",
    detector=None,
    retriever_type="hybrid",
    fusion_gate=None,
    use_sycophancy=False,
    use_reranker=True,
    skip_evaluation=False
))

# Main: Full SAF-RAG
ExperimentRegistry.register(ExperimentConfig(
    name="saf-rag",
    exp_type="saf-rag",
    description="Full SAF-RAG with all Phase 1-3 components",
    detector="sycophancy",
    retriever_type="hybrid",
    fusion_gate="learnable",
    use_sycophancy=True,
    use_reranker=True,
    skip_evaluation=False
))

# ============================================================================
# ABLATION EXPERIMENTS (measure contribution of each phase)
# ============================================================================

# Ablation: Phase 1 only (remove sycophancy detection)
ExperimentRegistry.register(ExperimentConfig(
    name="ablation-no-phase1",
    exp_type="rag",
    description="Ablation: Vanilla RAG without Phase 1 (sycophancy detection)",
    detector=None,
    retriever_type="hybrid",
    fusion_gate=None,
    use_sycophancy=False,
    use_reranker=True,
    is_ablation=True,
    ablation_of="Phase 1: Sycophancy Detection"
))

# Ablation: Phase 2 minimal (BM25 only, no dense retrieval)
ExperimentRegistry.register(ExperimentConfig(
    name="ablation-bm25-only",
    exp_type="saf-rag",
    description="Ablation: SAF-RAG with BM25-only retrieval (no dense)",
    detector="sycophancy",
    retriever_type="bm25-only",
    fusion_gate="learnable",
    use_sycophancy=True,
    use_reranker=False,
    is_ablation=True,
    ablation_of="Phase 2: Dense Retrieval"
))

# Ablation: Phase 2 minimal (no reranker)
ExperimentRegistry.register(ExperimentConfig(
    name="ablation-no-reranker",
    exp_type="saf-rag",
    description="Ablation: SAF-RAG without cross-encoder reranking",
    detector="sycophancy",
    retriever_type="hybrid",
    fusion_gate="learnable",
    use_sycophancy=True,
    use_reranker=False,
    is_ablation=True,
    ablation_of="Phase 2: Reranking"
))

# Ablation: No Phase 3 evaluation (just measure generation speed)
ExperimentRegistry.register(ExperimentConfig(
    name="ablation-no-phase3",
    exp_type="saf-rag",
    description="Ablation: SAF-RAG without Phase 3 evaluation (generation only)",
    detector="sycophancy",
    retriever_type="hybrid",
    fusion_gate="learnable",
    use_sycophancy=True,
    use_reranker=True,
    skip_evaluation=True,
    is_ablation=True,
    ablation_of="Phase 3: Evaluation"
))

# Ablation: Fixed fusion gate vs learnable
ExperimentRegistry.register(ExperimentConfig(
    name="ablation-fixed-gate",
    exp_type="saf-rag",
    description="Ablation: SAF-RAG with fixed fusion gate (not learnable)",
    detector="sycophancy",
    retriever_type="hybrid",
    fusion_gate="fixed",
    use_sycophancy=True,
    use_reranker=True,
    is_ablation=True,
    ablation_of="Phase 1: Learnable Fusion Gate"
))


def print_experiment_info():
    """Print all registered experiments."""
    registry = ExperimentRegistry()
    
    print("\n" + "="*80)
    print("REGISTERED EXPERIMENTS")
    print("="*80)
    
    print("\n[CORE EXPERIMENTS]")
    for name in registry.list_experiments_by_type("llm"):
        config = registry.get_experiment(name)
        print(f"  • {name:30s} — {config.description}")
    
    for name in registry.list_experiments_by_type("rag"):
        if not registry.get_experiment(name).is_ablation:
            config = registry.get_experiment(name)
            print(f"  • {name:30s} — {config.description}")
    
    for name in registry.list_experiments_by_type("saf-rag"):
        if not registry.get_experiment(name).is_ablation:
            config = registry.get_experiment(name)
            print(f"  • {name:30s} — {config.description}")
    
    print("\n[ABLATION EXPERIMENTS]")
    for name in registry.list_ablations():
        config = registry.get_experiment(name)
        print(f"  • {name:30s} — {config.description}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print_experiment_info()
