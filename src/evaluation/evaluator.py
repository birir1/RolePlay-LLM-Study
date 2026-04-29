"""
Evaluation Orchestrator

Main interface for comprehensive SAF-RAG evaluation.
Coordinates all metrics and produces evaluation reports.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
from datetime import datetime

from .sycophancy_metrics import SycophancyMetrics
from .hallucination_check import HallucinationDetector
from .persona_drift import PersonaDriftDetector
from .faithfulness_scorer import FaithfulnessScorer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    # Evaluation scope
    eval_sycophancy: bool = True
    eval_hallucinations: bool = True
    eval_persona_drift: bool = True
    eval_faithfulness: bool = True
    
    # Sycophancy settings
    num_false_premise_tests: int = 10
    num_contradiction_tests: int = 10
    
    # Persona settings
    check_role_drift: bool = True
    check_character_adherence: bool = True
    
    # Output
    save_report: bool = True
    report_path: str = "evaluation_report.json"
    device: str = "cuda"


class Evaluator:
    """
    Comprehensive SAF-RAG evaluator.
    
    Coordinates evaluation across multiple dimensions:
    - Sycophancy
    - Hallucinations
    - Persona consistency
    - Faithfulness
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize evaluator.
        
        Args:
            config: EvaluationConfig instance
        """
        self.config = config or EvaluationConfig()
        
        logger.info("[Evaluator] Initializing evaluation components...")
        
        # Initialize metric modules
        if self.config.eval_sycophancy:
            self.sycophancy_metrics = SycophancyMetrics(device=self.config.device)
            logger.info("  ✓ Sycophancy metrics loaded")
        
        if self.config.eval_hallucinations:
            self.hallucination_detector = HallucinationDetector(device=self.config.device)
            logger.info("  ✓ Hallucination detector loaded")
        
        if self.config.eval_persona_drift:
            self.persona_detector = PersonaDriftDetector(device=self.config.device)
            logger.info("  ✓ Persona drift detector loaded")
        
        if self.config.eval_faithfulness:
            self.faithfulness_scorer = FaithfulnessScorer(device=self.config.device)
            logger.info("  ✓ Faithfulness scorer loaded")
        
        logger.info("[Evaluator] Initialization complete ✓\n")
    
    def evaluate_answer(
        self,
        query: str,
        model_answer: str,
        retrieved_evidence: Optional[List[str]] = None,
        ground_truth: Optional[str] = None
    ) -> Dict:
        """
        Evaluate a single model answer.
        
        Args:
            query: The input query
            model_answer: Model's generated answer
            retrieved_evidence: Retrieved documents for context
            ground_truth: Ground truth answer (for faithfulness)
            
        Returns:
            Comprehensive evaluation result
        """
        logger.info(f"Evaluating answer for query: {query[:60]}...")
        
        result = {
            "query": query,
            "answer": model_answer[:100] + "..." if len(model_answer) > 100 else model_answer,
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        retrieved_evidence = retrieved_evidence or []
        ground_truth = ground_truth or ""
        
        # Sycophancy evaluation
        if self.config.eval_sycophancy:
            logger.info("  Evaluating sycophancy...")
            # Simple sycophancy check based on query-answer agreement
            sycophancy_result = self.sycophancy_metrics.compute_simple_sycophancy(
                query, model_answer
            )
            result["metrics"]["sycophancy"] = sycophancy_result
        
        # Hallucination evaluation
        if self.config.eval_hallucinations and retrieved_evidence:
            logger.info("  Evaluating hallucinations...")
            hallucination_result = self.hallucination_detector.detect_evidence_hallucinations(
                model_answer, retrieved_evidence
            )
            result["metrics"]["hallucinations"] = hallucination_result
            
            # Self-contradictions
            self_contradiction_result = self.hallucination_detector.detect_self_contradictions(
                model_answer
            )
            result["metrics"]["self_contradictions"] = self_contradiction_result
        
        # Faithfulness evaluation
        if self.config.eval_faithfulness and ground_truth:
            logger.info("  Evaluating faithfulness...")
            faithfulness_result = self.faithfulness_scorer.compute_faithfulness_score(
                model_answer, retrieved_evidence, ground_truth
            )
            result["metrics"]["faithfulness"] = faithfulness_result
            
            # Completeness
            completeness_result = self.faithfulness_scorer.compute_answer_completeness(
                model_answer, ground_truth
            )
            result["metrics"]["completeness"] = completeness_result
            
            # Relevance
            relevance_result = self.faithfulness_scorer.compute_answer_relevance(
                model_answer, query
            )
            result["metrics"]["relevance"] = relevance_result
        
        # Persona drift evaluation (simplified for single answer)
        if self.config.eval_persona_drift:
            logger.info("  Evaluating persona drift...")
            # Simple persona consistency check
            persona_result = self.persona_detector.compute_simple_persona_consistency(
                query, model_answer
            )
            result["metrics"]["persona_drift"] = persona_result
        
        return result
    
    def evaluate_conversation(
        self,
        character_instruction: str,
        conversation: List[Dict],
        persona_traits: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Evaluate a multi-turn conversation.
        
        Args:
            character_instruction: Character/role description
            conversation: List of dicts with "user" and "assistant" turns
            persona_traits: Optional character traits to check
            
        Returns:
            Conversation-level evaluation
        """
        logger.info(f"Evaluating conversation with {len(conversation)} turns...")
        
        # Extract assistant turns
        assistant_turns = [
            turn["assistant"] for turn in conversation 
            if "assistant" in turn
        ]
        
        result = {
            "character_instruction": character_instruction[:100],
            "num_turns": len(assistant_turns),
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Persona drift evaluation
        if self.config.eval_persona_drift:
            logger.info("  Evaluating persona drift...")
            
            # Role drift
            rdi_result = self.persona_detector.compute_role_drift_index(
                character_instruction, assistant_turns
            )
            result["metrics"]["role_drift_index"] = rdi_result
            
            # Character adherence
            ca_result = self.persona_detector.compute_character_adherence(
                character_instruction, assistant_turns
            )
            result["metrics"]["character_adherence"] = ca_result
            
            # Persona consistency
            if persona_traits:
                pcs_result = self.persona_detector.compute_persona_consistency(
                    persona_traits, assistant_turns
                )
                result["metrics"]["persona_consistency"] = pcs_result
            
            # Role breaks
            role_breaks = self.persona_detector.detect_role_breaks(
                character_instruction, assistant_turns
            )
            result["metrics"]["role_breaks"] = role_breaks
            
            # Conversation coherence
            coherence = self.persona_detector.compute_conversation_coherence(
                assistant_turns
            )
            result["metrics"]["coherence"] = coherence
        
        return result
    
    def evaluate_batch(
        self,
        test_cases: List[Dict]
    ) -> Dict:
        """
        Evaluate multiple test cases.
        
        Each test case should have:
        - query: Input query
        - answer: Model's answer
        - retrieved_evidence: Retrieved documents
        - ground_truth: Reference answer
        
        Args:
            test_cases: List of test case dicts
            
        Returns:
            Batch evaluation results
        """
        logger.info(f"Evaluating batch of {len(test_cases)} test cases...")
        
        results = {
            "test_cases": [],
            "summary": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Evaluate each test case
        for i, test_case in enumerate(test_cases):
            logger.info(f"  [{i+1}/{len(test_cases)}] Evaluating...")
            
            result = self.evaluate_answer(
                query=test_case.get("query", ""),
                model_answer=test_case.get("answer", ""),
                retrieved_evidence=test_case.get("retrieved_evidence"),
                ground_truth=test_case.get("ground_truth")
            )
            
            results["test_cases"].append(result)
        
        # Compute summary statistics
        results["summary"] = self._compute_batch_summary(results["test_cases"])
        
        return results
    
    def _compute_batch_summary(self, test_results: List[Dict]) -> Dict:
        """Compute summary statistics for batch."""
        import numpy as np
        
        summary = {}
        
        # Hallucination summary
        if "hallucinations" in test_results[0].get("metrics", {}):
            hallucination_rates = [
                r["metrics"]["hallucinations"]["hallucination_rate"]
                for r in test_results
                if "hallucinations" in r.get("metrics", {})
            ]
            if hallucination_rates:
                summary["hallucination"] = {
                    "mean": float(np.mean(hallucination_rates)),
                    "std": float(np.std(hallucination_rates)),
                    "min": float(np.min(hallucination_rates)),
                    "max": float(np.max(hallucination_rates))
                }
        
        # Faithfulness summary
        if "faithfulness" in test_results[0].get("metrics", {}):
            faithfulness_scores = [
                r["metrics"]["faithfulness"]["faithfulness_score"]
                for r in test_results
                if "faithfulness" in r.get("metrics", {})
            ]
            if faithfulness_scores:
                summary["faithfulness"] = {
                    "mean": float(np.mean(faithfulness_scores)),
                    "std": float(np.std(faithfulness_scores)),
                    "min": float(np.min(faithfulness_scores)),
                    "max": float(np.max(faithfulness_scores)),
                    "faithful_count": sum(1 for s in faithfulness_scores if s > 0.7)
                }
        
        return summary
    
    def generate_report(
        self,
        evaluation_results: Dict,
        save: bool = True
    ) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            evaluation_results: Result from evaluate_batch or evaluate_answer
            save: Whether to save to file
            
        Returns:
            Report string
        """
        import json
        
        report = []
        report.append("=" * 70)
        report.append("SAF-RAG EVALUATION REPORT")
        report.append("=" * 70)
        report.append(f"\nTimestamp: {evaluation_results.get('timestamp', 'N/A')}")
        
        # Summary if batch evaluation
        if "summary" in evaluation_results:
            report.append("\n[SUMMARY STATISTICS]")
            
            summary = evaluation_results["summary"]
            
            if "hallucination" in summary:
                h = summary["hallucination"]
                report.append(f"\nHallucination Rate:")
                report.append(f"  Mean: {h['mean']:.3f}")
                report.append(f"  Std:  {h['std']:.3f}")
                report.append(f"  Range: [{h['min']:.3f}, {h['max']:.3f}]")
            
            if "faithfulness" in summary:
                f = summary["faithfulness"]
                report.append(f"\nFaithfulness:")
                report.append(f"  Mean: {f['mean']:.3f}")
                report.append(f"  Std:  {f['std']:.3f}")
                report.append(f"  Faithful responses: {f.get('faithful_count', 0)}/{len(evaluation_results['test_cases'])}")
        
        # Individual results if single evaluation
        if "metrics" in evaluation_results:
            report.append("\n[EVALUATION METRICS]")
            
            metrics = evaluation_results["metrics"]
            
            if "hallucinations" in metrics:
                h = metrics["hallucinations"]
                report.append(f"\nHallucinations:")
                report.append(f"  Rate: {h.get('hallucination_rate', 0):.3f}")
                report.append(f"  Unsupported claims: {len(h.get('unsupported_claims', []))}")
                report.append(f"  Supported claims: {len(h.get('supported_claims', []))}")
            
            if "faithfulness" in metrics:
                f = metrics["faithfulness"]
                report.append(f"\nFaithfulness: {f['faithfulness_score']:.3f}")
                report.append(f"  Factuality: {f['component_scores']['factuality']:.3f}")
                report.append(f"  Evidence grounding: {f['component_scores']['evidence_grounding']:.3f}")
                report.append(f"  Self-consistency: {f['component_scores']['self_consistency']:.3f}")
        
        report_text = "\n".join(report)
        
        if save and self.config.save_report:
            with open(self.config.report_path, "w") as f:
                json.dump(evaluation_results, f, indent=2)
            logger.info(f"\nReport saved to {self.config.report_path}")
        
        return report_text
