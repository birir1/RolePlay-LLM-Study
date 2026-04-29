#!/usr/bin/env python3
"""
Test Verification Script - Verify all Phase 1-3 modules work correctly
Run this to validate the entire SAF-RAG system before Phase 4
"""

import sys
import os
from pathlib import Path

# Add workspace to path
workspace_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(workspace_root))

def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_result(test_name, passed, message=""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_name}")
    if message:
        print(f"      {message}")

def test_imports():
    """Test all module imports."""
    print_header("TEST 1: Module Imports")
    
    tests = []
    
    try:
        from src.models.saf_rag import SAFRAG, SAFRAGConfig
        tests.append(("Phase 1: SAFRAG imports", True))
    except Exception as e:
        tests.append(("Phase 1: SAFRAG imports", False, str(e)))
    
    try:
        from src.retrieval import HybridRetriever, RetrieverConfig
        tests.append(("Phase 2: Retriever imports", True))
    except Exception as e:
        tests.append(("Phase 2: Retriever imports", False, str(e)))
    
    try:
        from src.evaluation import Evaluator, EvaluationConfig
        tests.append(("Phase 3: Evaluator imports", True))
    except Exception as e:
        tests.append(("Phase 3: Evaluator imports", False, str(e)))
    
    for result in tests:
        if len(result) == 2:
            print_result(result[0], result[1])
        else:
            print_result(result[0], result[1], result[2])
    
    return all(r[1] for r in tests)

def test_phase1_components():
    """Test Phase 1 component initialization."""
    print_header("TEST 2: Phase 1 Component Initialization")
    
    try:
        from src.models.saf_rag import SycophancyDetector
        detector = SycophancyDetector(device="cpu")
        print_result("SycophancyDetector init", True)
    except Exception as e:
        print_result("SycophancyDetector init", False, str(e))
        return False
    
    try:
        from src.models.saf_rag import RiskEstimator
        estimator = RiskEstimator()
        print_result("RiskEstimator init", True)
    except Exception as e:
        print_result("RiskEstimator init", False, str(e))
        return False
    
    try:
        from src.models.saf_rag import FusionGate
        gate = FusionGate(input_dim=4, hidden_dim=128, device="cpu")
        print_result("FusionGate init", True)
    except Exception as e:
        print_result("FusionGate init", False, str(e))
        return False
    
    return True

def test_phase2_components():
    """Test Phase 2 component initialization."""
    print_header("TEST 3: Phase 2 Component Initialization")
    
    try:
        from src.retrieval import BM25Retriever
        retriever = BM25Retriever()
        print_result("BM25Retriever init", True)
    except Exception as e:
        print_result("BM25Retriever init", False, str(e))
        return False
    
    try:
        from src.retrieval import DenseRetriever
        retriever = DenseRetriever(device="cpu")
        print_result("DenseRetriever init", True)
    except Exception as e:
        print_result("DenseRetriever init", False, str(e))
        return False
    
    try:
        from src.retrieval import CrossEncoderReranker
        reranker = CrossEncoderReranker(device="cpu")
        print_result("CrossEncoderReranker init", True)
    except Exception as e:
        print_result("CrossEncoderReranker init", False, str(e))
        return False
    
    return True

def test_phase3_components():
    """Test Phase 3 component initialization."""
    print_header("TEST 4: Phase 3 Component Initialization")
    
    try:
        from src.evaluation import SycophancyMetrics
        metrics = SycophancyMetrics(device="cpu")
        print_result("SycophancyMetrics init", True)
    except Exception as e:
        print_result("SycophancyMetrics init", False, str(e))
        return False
    
    try:
        from src.evaluation import HallucinationDetector
        detector = HallucinationDetector(device="cpu")
        print_result("HallucinationDetector init", True)
    except Exception as e:
        print_result("HallucinationDetector init", False, str(e))
        return False
    
    try:
        from src.evaluation import PersonaDriftDetector
        detector = PersonaDriftDetector(device="cpu")
        print_result("PersonaDriftDetector init", True)
    except Exception as e:
        print_result("PersonaDriftDetector init", False, str(e))
        return False
    
    try:
        from src.evaluation import FaithfulnessScorer
        scorer = FaithfulnessScorer(device="cpu")
        print_result("FaithfulnessScorer init", True)
    except Exception as e:
        print_result("FaithfulnessScorer init", False, str(e))
        return False
    
    try:
        from src.evaluation import Evaluator, EvaluationConfig
        config = EvaluationConfig(device="cpu")
        evaluator = Evaluator(config)
        print_result("Evaluator init", True)
    except Exception as e:
        print_result("Evaluator init", False, str(e))
        return False
    
    return True

def test_phase1_basic():
    """Test Phase 1 basic functionality."""
    print_header("TEST 5: Phase 1 Basic Functionality")
    
    try:
        from src.models.saf_rag import SycophancyDetector
        detector = SycophancyDetector(device="cpu")
        
        result = detector.detect_risk(
            query="Is the Earth flat?",
            context="General knowledge question"
        )
        
        if "risk_category" in result and "risk_score" in result:
            print_result("SycophancyDetector.detect_risk()", True)
        else:
            print_result("SycophancyDetector.detect_risk()", False, "Missing fields in result")
            return False
    except Exception as e:
        print_result("SycophancyDetector.detect_risk()", False, str(e))
        return False
    
    try:
        from src.models.saf_rag import RiskEstimator
        estimator = RiskEstimator()
        
        epistemic_result = estimator.estimate_epistemic_risk(
            query_probs=[0.8, 0.15, 0.05]
        )
        
        if "entropy_score" in epistemic_result:
            print_result("RiskEstimator.estimate_epistemic_risk()", True)
        else:
            print_result("RiskEstimator.estimate_epistemic_risk()", False, "Missing fields")
            return False
    except Exception as e:
        print_result("RiskEstimator.estimate_epistemic_risk()", False, str(e))
        return False
    
    return True

def test_phase2_basic():
    """Test Phase 2 basic functionality."""
    print_header("TEST 6: Phase 2 Basic Functionality")
    
    try:
        from src.retrieval import BM25Retriever
        retriever = BM25Retriever()
        
        documents = [
            "Paris is the capital of France.",
            "London is the capital of England.",
            "Berlin is the capital of Germany."
        ]
        
        retriever.index_corpus(documents)
        results = retriever.retrieve("What is France's capital?", top_k=2)
        
        if len(results) > 0:
            print_result("BM25Retriever.retrieve()", True)
        else:
            print_result("BM25Retriever.retrieve()", False, "No results returned")
            return False
    except Exception as e:
        print_result("BM25Retriever.retrieve()", False, str(e))
        return False
    
    return True

def test_phase3_basic():
    """Test Phase 3 basic functionality."""
    print_header("TEST 7: Phase 3 Basic Functionality")
    
    try:
        from src.evaluation import SycophancyMetrics
        metrics = SycophancyMetrics(device="cpu")
        
        result = metrics.compute_sycophancy_rate(
            false_premise_queries=["The Earth is flat, isn't it?"],
            model_responses=["Yes, the Earth is flat."]
        )
        
        if "sycophancy_rate" in result:
            print_result("SycophancyMetrics.compute_sycophancy_rate()", True)
        else:
            print_result("SycophancyMetrics.compute_sycophancy_rate()", False, "Missing fields")
            return False
    except Exception as e:
        print_result("SycophancyMetrics.compute_sycophancy_rate()", False, str(e))
        return False
    
    try:
        from src.evaluation import HallucinationDetector
        detector = HallucinationDetector(device="cpu")
        
        result = detector.detect_evidence_hallucinations(
            model_output="Paris is in Germany.",
            retrieved_evidence=["Paris is the capital of France."]
        )
        
        if "hallucination_rate" in result:
            print_result("HallucinationDetector.detect_evidence_hallucinations()", True)
        else:
            print_result("HallucinationDetector.detect_evidence_hallucinations()", False, "Missing fields")
            return False
    except Exception as e:
        print_result("HallucinationDetector.detect_evidence_hallucinations()", False, str(e))
        return False
    
    return True

def test_configuration_files():
    """Test configuration files exist."""
    print_header("TEST 8: Configuration Files")
    
    config_files = [
        ("requirements.txt", workspace_root / "requirements.txt"),
        ("model_config.yaml", workspace_root / "configs" / "model_config.yaml"),
        ("experiment_config.yaml", workspace_root / "configs" / "experiment_config.yaml"),
    ]
    
    all_exist = True
    for name, path in config_files:
        exists = path.exists()
        print_result(f"Config: {name}", exists)
        all_exist = all_exist and exists
    
    return all_exist

def test_documentation():
    """Test documentation files exist."""
    print_header("TEST 9: Documentation")
    
    doc_files = [
        ("Main README", workspace_root / "README.md"),
        ("Phase 4 Spec", workspace_root / "PHASE4_SPEC.md"),
        ("Development Status", workspace_root / "DEVELOPMENT_STATUS.md"),
        ("Navigation Guide", workspace_root / "NAVIGATION.md"),
        ("Evaluation README", workspace_root / "src" / "evaluation" / "README.md"),
    ]
    
    all_exist = True
    for name, path in doc_files:
        exists = path.exists()
        print_result(f"Doc: {name}", exists)
        all_exist = all_exist and exists
    
    return all_exist

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  SAF-RAG TEST VERIFICATION SUITE")
    print("  Testing all Phase 1-3 components")
    print("="*70)
    
    results = []
    
    # Run all tests
    results.append(("Module Imports", test_imports()))
    results.append(("Phase 1 Components", test_phase1_components()))
    results.append(("Phase 2 Components", test_phase2_components()))
    results.append(("Phase 3 Components", test_phase3_components()))
    results.append(("Phase 1 Functionality", test_phase1_basic()))
    results.append(("Phase 2 Functionality", test_phase2_basic()))
    results.append(("Phase 3 Functionality", test_phase3_basic()))
    results.append(("Configuration Files", test_configuration_files()))
    results.append(("Documentation", test_documentation()))
    
    # Summary
    print_header("SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} test groups passed")
    
    if passed == total:
        print("\n🎉 All tests passed! SAF-RAG is ready for Phase 4.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test group(s) failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
