#!/usr/bin/env python3
"""
Phase 1 Validation Script
Tests the critical bug fixes implemented in Phase 1:
1. LLM parameter standardization
2. Missing metric implementations completion
3. Composite metric inheritance fixes
4. Import and validation fixes
"""

import sys
import os

def test_imports():
    """Test that all imports work correctly after changes."""
    print("Testing imports...")
    
    try:
        # Test main evaluator import
        from src import RAGEvaluator
        print("✅ RAGEvaluator import successful")
        
        # Test metrics imports
        from src import (
            Faithfulness, AnswerRelevance, Coherence, 
            ContextRelevance, LLMJudge, RAGCertainty, TrustScore
        )
        print("✅ All metrics import successful")
        
        # Test core types
        from src import RAGInput, MetricResult
        print("✅ Core types import successful")
        
        # Test LLM module
        from src import create_llm
        print("✅ LLM module import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metric_initialization():
    """Test that metrics can be initialized with standardized parameters."""
    print("\nTesting metric initialization...")
    
    try:
        # Mock LLM for testing
        class MockLLM:
            def __init__(self):
                self.model_name = "test-model"
        
        mock_llm = MockLLM()
        
        # Test generation metrics
        from src import Faithfulness
        faithfulness = Faithfulness(llm=mock_llm)
        print(f"✅ Faithfulness initialized: {faithfulness.name()}")
        
        # Test composite metrics with new inheritance
        from src.metrics.composite.llm_judge import LLMJudge, JudgmentCriteria
        llm_judge = LLMJudge(llm=mock_llm)
        print(f"✅ LLMJudge initialized: {llm_judge.name()}")
        print(f"✅ JudgmentCriteria enum available: {list(JudgmentCriteria)}")
        
        from src import RAGCertainty
        rag_certainty = RAGCertainty(llm=mock_llm)
        print(f"✅ RAGCertainty initialized: {rag_certainty.name()}")
        
        from src import TrustScore
        trust_score = TrustScore(llm=mock_llm)
        print(f"✅ TrustScore initialized: {trust_score.name()}")
        
        return True
    except Exception as e:
        print(f"❌ Metric initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inheritance():
    """Test that composite metrics inherit correctly."""
    print("\nTesting inheritance...")
    
    try:
        from src.metrics.composite.base_composite import BaseCompositeMetric
        from src.metrics.composite.llm_judge import LLMJudge
        from src.metrics.composite.rag_certainty import RAGCertainty
        from src.metrics.composite.trust_score import TrustScore
        
        # Check inheritance
        assert issubclass(LLMJudge, BaseCompositeMetric), "LLMJudge should inherit from BaseCompositeMetric"
        assert issubclass(RAGCertainty, BaseCompositeMetric), "RAGCertainty should inherit from BaseCompositeMetric"
        assert issubclass(TrustScore, BaseCompositeMetric), "TrustScore should inherit from BaseCompositeMetric"
        
        print("✅ All composite metrics inherit correctly from BaseCompositeMetric")
        return True
    except Exception as e:
        print(f"❌ Inheritance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_parameter_consistency():
    """Test that all metrics use consistent LLM parameter naming."""
    print("\nTesting LLM parameter consistency...")
    
    try:
        import inspect
        from src import Faithfulness, Helpfulness
        from src.metrics.composite.llm_judge import LLMJudge
        
        # Check constructor signatures
        metrics_to_test = [Faithfulness, Helpfulness, LLMJudge]
        
        for metric_class in metrics_to_test:
            sig = inspect.signature(metric_class.__init__)
            params = list(sig.parameters.keys())
            
            # Should have 'llm' parameter, not 'llm_provider'
            if 'llm_provider' in params:
                raise ValueError(f"{metric_class.__name__} still uses 'llm_provider' parameter")
            
            if 'llm' not in params:
                raise ValueError(f"{metric_class.__name__} missing 'llm' parameter")
                
            print(f"✅ {metric_class.__name__} uses standardized 'llm' parameter")
        
        return True
    except Exception as e:
        print(f"❌ LLM parameter consistency test failed: {e}")
        return False

def test_evaluator_integration():
    """Test that RAGEvaluator works with the standardized interface."""
    print("\nTesting RAGEvaluator integration...")
    
    try:
        from src.evaluator import RAGEvaluator
        
        # Test that evaluator can be initialized with mock LLM
        class MockLLM:
            def __init__(self):
                self.model_name = "test-model"
        
        mock_llm = MockLLM()
        evaluator = RAGEvaluator(
            metrics=["faithfulness", "llm_judge", "rag_certainty"],
            llm=mock_llm  # Use pre-configured mock LLM
        )
        
        print(f"✅ RAGEvaluator initialized with metrics: {evaluator.list_metrics()}")
        
        # Test available metrics
        available = RAGEvaluator.list_available_metrics()
        expected_metrics = ["faithfulness", "llm_judge", "rag_certainty", "trust_score"]
        
        for metric in expected_metrics:
            if metric not in available:
                raise ValueError(f"Missing metric in available metrics: {metric}")
        
        print("✅ All expected metrics are available")
        return True
    except Exception as e:
        print(f"❌ RAGEvaluator integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("PHASE 1 VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_metric_initialization,
        test_inheritance,
        test_llm_parameter_consistency,
        test_evaluator_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("Phase 1 changes are working correctly!")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total})")
        print("Please review the failed tests above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 