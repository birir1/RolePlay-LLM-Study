"""
Faithfulness Scorer Module

Comprehensive faithfulness metric combining:
- Factuality (against ground truth)
- Hallucination (against evidence)
- Self-consistency (internal contradictions)
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FaithfulnessScorer:
    """
    Comprehensive faithfulness evaluation.
    
    Combines multiple signals:
    - Factual correctness
    - Evidence grounding
    - Self-consistency
    - Answer completeness
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize faithfulness scorer.
        
        Args:
            device: torch device
        """
        self.device = device
        
        # Import evaluation modules
        from .hallucination_check import HallucinationDetector
        from .sycophancy_metrics import SycophancyMetrics
        
        self.hallucination_detector = HallucinationDetector(device)
        self.sycophancy_metrics = SycophancyMetrics(device)
    
    def compute_faithfulness_score(
        self,
        model_output: str,
        retrieved_evidence: List[str],
        ground_truth: str,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Compute overall faithfulness score.
        
        Combines:
        - Factuality (against ground truth)
        - Evidence grounding
        - Self-consistency
        
        Args:
            model_output: Model's generated answer
            retrieved_evidence: List of retrieved documents
            ground_truth: Ground truth answer
            weights: Optional custom weights
            
        Returns:
            Dictionary with:
                - faithfulness_score: [0, 1] (0=low, 1=high)
                - component_scores: Individual component scores
                - explanations: Reasoning for each component
                - is_faithful: boolean flag
        """
        # Default weights
        if weights is None:
            weights = {
                "factuality": 0.4,
                "evidence_grounding": 0.35,
                "self_consistency": 0.25
            }
        
        # Normalize weights
        weight_sum = sum(weights.values())
        weights = {k: v/weight_sum for k, v in weights.items()}
        
        # Component 1: Factuality against ground truth
        factuality = self._compute_factuality(model_output, ground_truth)
        
        # Component 2: Evidence grounding
        evidence_grounding = self._compute_evidence_grounding(
            model_output,
            retrieved_evidence
        )
        
        # Component 3: Self-consistency
        self_consistency = self._compute_self_consistency(model_output)
        
        # Weighted combination
        faithfulness = (
            weights["factuality"] * factuality["score"] +
            weights["evidence_grounding"] * evidence_grounding["score"] +
            weights["self_consistency"] * self_consistency["score"]
        )
        faithfulness = float(min(1.0, max(0.0, faithfulness)))
        
        return {
            "faithfulness_score": faithfulness,
            "component_scores": {
                "factuality": factuality["score"],
                "evidence_grounding": evidence_grounding["score"],
                "self_consistency": self_consistency["score"]
            },
            "component_details": {
                "factuality": factuality,
                "evidence_grounding": evidence_grounding,
                "self_consistency": self_consistency
            },
            "weights": weights,
            "is_faithful": faithfulness > 0.7,
            "main_issue": self._identify_main_issue(factuality, evidence_grounding, self_consistency)
        }
    
    def compute_answer_completeness(
        self,
        model_output: str,
        ground_truth: str
    ) -> Dict:
        """
        Measure whether answer covers all important points.
        
        Args:
            model_output: Model's output
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary with coverage analysis
        """
        # Extract key facts from ground truth
        key_facts = self._extract_key_facts(ground_truth)
        
        # Check which facts are covered in output
        covered_facts = []
        missing_facts = []
        
        output_lower = model_output.lower()
        
        for fact in key_facts:
            fact_lower = fact.lower()
            
            # Simple check: is fact mentioned?
            if self._is_fact_mentioned(fact_lower, output_lower):
                covered_facts.append(fact)
            else:
                missing_facts.append(fact)
        
        coverage_score = len(covered_facts) / len(key_facts) if key_facts else 1.0
        
        return {
            "completeness_score": float(coverage_score),
            "covered_facts": covered_facts,
            "missing_facts": missing_facts,
            "total_key_facts": len(key_facts),
            "is_complete": coverage_score > 0.8
        }
    
    def compute_answer_relevance(
        self,
        model_output: str,
        query: str
    ) -> Dict:
        """
        Measure how relevant answer is to the query.
        
        Args:
            model_output: Model's output
            query: Original query
            
        Returns:
            Dictionary with relevance analysis
        """
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        
        # Encode query and answer
        query_embedding = model.encode(query, convert_to_numpy=True)
        answer_embedding = model.encode(
            model_output[:512],  # Limit for efficiency
            convert_to_numpy=True
        )
        
        # Compute similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1),
            answer_embedding.reshape(1, -1)
        )[0, 0]
        
        return {
            "relevance_score": float(similarity),
            "is_relevant": similarity > 0.5,
            "query": query[:50],
            "answer_preview": model_output[:100]
        }
    
    def _compute_factuality(self, output: str, ground_truth: str) -> Dict:
        """Compute factuality component."""
        hallucination_result = self.hallucination_detector.detect_factuality_violations(
            output, ground_truth
        )
        
        return {
            "score": hallucination_result["factuality_score"],
            "hallucination_count": hallucination_result["hallucination_count"],
            "contradicted": hallucination_result["contradicted"]
        }
    
    def _compute_evidence_grounding(self, output: str, evidence: List[str]) -> Dict:
        """Compute evidence grounding component."""
        hallucination_result = self.hallucination_detector.detect_evidence_hallucinations(
            output, evidence
        )
        
        # Score: 1.0 if well-grounded, 0.0 if highly hallucinated
        evidence_score = 1.0 - hallucination_result["hallucination_rate"]
        
        return {
            "score": evidence_score,
            "hallucination_rate": hallucination_result["hallucination_rate"],
            "unsupported_claims": len(hallucination_result["unsupported_claims"]),
            "supported_claims": len(hallucination_result["supported_claims"])
        }
    
    def _compute_self_consistency(self, output: str) -> Dict:
        """Compute self-consistency component."""
        contradiction_result = self.hallucination_detector.detect_self_contradictions(output)
        
        # Score: 1.0 if no self-contradictions, decreases with contradictions
        num_contradictions = contradiction_result["num_contradictions"]
        self_consistency_score = 1.0 / (1.0 + num_contradictions)  # 1.0 if none, ~0.5 if 1, etc
        
        return {
            "score": float(self_consistency_score),
            "num_contradictions": num_contradictions,
            "has_self_contradictions": contradiction_result["has_self_contradictions"],
            "contradictions": contradiction_result["self_contradictions"]
        }
    
    @staticmethod
    def _identify_main_issue(
        factuality: Dict,
        evidence: Dict,
        consistency: Dict
    ) -> str:
        """Identify the main faithfulness issue."""
        scores = {
            "factuality": factuality["score"],
            "evidence_grounding": evidence["score"],
            "self_consistency": consistency["score"]
        }
        
        min_component = min(scores, key=scores.get)
        min_score = scores[min_component]
        
        if min_score < 0.5:
            return f"Low {min_component}"
        elif min_score < 0.7:
            return f"Moderate {min_component} issues"
        else:
            return "Faithful"
    
    @staticmethod
    def _extract_key_facts(text: str) -> List[str]:
        """Extract key factual statements from text."""
        # Simple heuristic: split by sentence and filter
        import re
        sentences = re.split(r'[.!?]+', text)
        
        facts = []
        for sent in sentences:
            sent = sent.strip()
            # Look for factual statements (not questions, not too short)
            if (sent and not sent.endswith("?") and 
                len(sent.split()) >= 4 and
                not sent.lower().startswith(("i think", "in my", "perhaps"))):
                facts.append(sent)
        
        return facts
    
    @staticmethod
    def _is_fact_mentioned(fact: str, text: str) -> bool:
        """Check if fact is mentioned in text."""
        # Simple word overlap check
        fact_words = set(fact.split()) - {"the", "is", "are", "a", "an"}
        text_words = set(text.split())
        
        if not fact_words:
            return False
        
        overlap = len(fact_words & text_words) / len(fact_words)
        return overlap >= 0.6


class FaithfulnessComparator:
    """Compare faithfulness across multiple models."""
    
    def __init__(self, device: str = "cuda"):
        """Initialize comparator."""
        self.scorer = FaithfulnessScorer(device)
        self.device = device
    
    def compare_models(
        self,
        test_cases: List[Dict],
        model_outputs: Dict[str, List[str]]
    ) -> Dict:
        """
        Compare faithfulness across models.
        
        Args:
            test_cases: List of test case dicts with:
                - query
                - retrieved_evidence
                - ground_truth
            model_outputs: Dict mapping model names to output lists
            
        Returns:
            Comparison results
        """
        results = {}
        
        for model_name, outputs in model_outputs.items():
            scores = []
            
            for i, output in enumerate(outputs):
                if i >= len(test_cases):
                    break
                
                test_case = test_cases[i]
                
                try:
                    score = self.scorer.compute_faithfulness_score(
                        model_output=output,
                        retrieved_evidence=test_case.get("retrieved_evidence", []),
                        ground_truth=test_case.get("ground_truth", ""),
                        weights=None
                    )
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"Error evaluating model {model_name}: {e}")
                    continue
            
            if scores:
                faithfulness_scores = [s["faithfulness_score"] for s in scores]
                results[model_name] = {
                    "mean_faithfulness": float(np.mean(faithfulness_scores)),
                    "std_faithfulness": float(np.std(faithfulness_scores)),
                    "min_faithfulness": float(np.min(faithfulness_scores)),
                    "max_faithfulness": float(np.max(faithfulness_scores)),
                    "num_faithful": sum(1 for s in scores if s["is_faithful"]),
                    "num_samples": len(scores),
                    "component_means": {
                        "factuality": float(np.mean([s["component_scores"]["factuality"] for s in scores])),
                        "evidence_grounding": float(np.mean([s["component_scores"]["evidence_grounding"] for s in scores])),
                        "self_consistency": float(np.mean([s["component_scores"]["self_consistency"] for s in scores]))
                    }
                }
        
        return results
