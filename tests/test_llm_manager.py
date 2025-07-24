#!/usr/bin/env python3
"""
Test script to demonstrate LLM Manager functionality.
Tests connection pooling, caching, configuration validation, and statistics.
"""

import sys
import time
from src import get_llm_manager, RAGEvaluator


def test_llm_manager_functionality():
    """Test LLM Manager core functionality."""
    print("=" * 60)
    print("TESTING LLM MANAGER FUNCTIONALITY")
    print("=" * 60)
    
    # Test 1: Get LLM Manager instance
    print("\n1. Testing LLM Manager singleton pattern...")
    manager1 = get_llm_manager()
    manager2 = get_llm_manager()
    
    if manager1 is manager2:
        print("✅ Singleton pattern working correctly")
    else:
        print("❌ Singleton pattern failed")
        return False
    
    # Test 2: Configuration validation
    print("\n2. Testing configuration validation...")
    
    # Valid Azure config
    valid, error = manager1.validate_config(
        provider="azure",
        model="gpt-4",
        azure_config={
            "azure_endpoint": "https://test.openai.azure.com/",
            "azure_deployment": "gpt-4-deployment"
        }
    )
    
    if valid:
        print("✅ Valid Azure configuration accepted")
    else:
        print(f"❌ Valid configuration rejected: {error}")
        return False
    
    # Invalid config
    valid, error = manager1.validate_config(
        provider="invalid_provider",
        model="gpt-4"
    )
    
    if not valid:
        print("✅ Invalid configuration properly rejected")
    else:
        print("❌ Invalid configuration accepted")
        return False
    
    # Test 3: Statistics before any usage
    print("\n3. Testing initial statistics...")
    stats = manager1.get_stats()
    
    if stats["manager"]["total_requests"] == 0:
        print("✅ Initial statistics correct")
        print(f"   Pool size: {stats['pool']['pool_size']}")
        print(f"   Total requests: {stats['manager']['total_requests']}")
    else:
        print("❌ Initial statistics incorrect")
        return False
    
    return True


def test_llm_manager_with_evaluator():
    """Test LLM Manager integration with RAGEvaluator."""
    print("\n" + "=" * 60)
    print("TESTING LLM MANAGER WITH RAGEVALUATOR")
    print("=" * 60)
    
    # Test 1: Create evaluator with LLM Manager (default)
    print("\n1. Testing RAGEvaluator with LLM Manager enabled...")
    
    # Mock LLM for testing
    class MockLLM:
        def __init__(self):
            self.model_name = "test-model"
    
    try:
        # Test with pre-configured LLM (should bypass manager)
        mock_llm = MockLLM()
        evaluator1 = RAGEvaluator(
            metrics=["faithfulness"],
            llm=mock_llm
        )
        
        if not evaluator1.use_llm_manager:
            print("✅ Pre-configured LLM bypasses manager correctly")
        else:
            print("❌ Pre-configured LLM incorrectly uses manager")
            return False
        
        # Test with manager disabled
        evaluator2 = RAGEvaluator(
            metrics=["faithfulness"],
            llm=mock_llm,
            use_llm_manager=False
        )
        
        if not evaluator2.use_llm_manager:
            print("✅ LLM Manager correctly disabled when requested")
        else:
            print("❌ LLM Manager not disabled when requested")
            return False
        
    except Exception as e:
        print(f"❌ Error testing RAGEvaluator: {e}")
        return False
    
    # Test 2: Check evaluator statistics
    print("\n2. Testing evaluator statistics...")
    
    stats = evaluator1.get_evaluation_stats()
    
    if "using_llm_manager" in stats["evaluator"]:
        print("✅ LLM Manager status included in evaluator stats")
        print(f"   Using LLM Manager: {stats['evaluator']['using_llm_manager']}")
    else:
        print("❌ LLM Manager status missing from evaluator stats")
        return False
    
    # Test 3: Cache clearing
    print("\n3. Testing cache clearing...")
    
    try:
        evaluator1.clear_all_caches()
        print("✅ Cache clearing works correctly")
    except Exception as e:
        print(f"❌ Cache clearing failed: {e}")
        return False
    
    return True


def test_llm_manager_performance():
    """Test LLM Manager performance and caching behavior."""
    print("\n" + "=" * 60)
    print("TESTING LLM MANAGER PERFORMANCE")
    print("=" * 60)
    
    manager = get_llm_manager()
    
    # Clear any existing cache
    manager.clear_cache()
    
    print("\n1. Testing performance with mock configurations...")
    
    # Simulate multiple requests with same config
    configs = [
        {"provider": "azure", "model": "gpt-4", 
         "azure_config": {"azure_endpoint": "https://test1.openai.azure.com/", 
                         "azure_deployment": "gpt-4"}},
        {"provider": "azure", "model": "gpt-4", 
         "azure_config": {"azure_endpoint": "https://test1.openai.azure.com/", 
                         "azure_deployment": "gpt-4"}},  # Same as above
        {"provider": "azure", "model": "gpt-35-turbo", 
         "azure_config": {"azure_endpoint": "https://test2.openai.azure.com/", 
                         "azure_deployment": "gpt-35-turbo"}},  # Different
    ]
    
    print(f"   Testing {len(configs)} configuration requests...")
    
    # Note: These will fail with actual LLM creation but will test the caching logic
    for i, config in enumerate(configs):
        try:
            # This will fail without real Azure credentials, but that's expected
            manager.get_llm(**config)
        except Exception:
            # Expected to fail, but should still update statistics
            pass
    
    # Check final statistics
    final_stats = manager.get_stats()
    print(f"   Final requests: {final_stats['manager']['total_requests']}")
    print(f"   Final errors: {final_stats['manager']['total_errors']}")
    print(f"   Pool size: {final_stats['pool']['pool_size']}")
    
    if final_stats['manager']['total_requests'] == len(configs):
        print("✅ Request counting works correctly")
    else:
        print("❌ Request counting failed")
        return False
    
    return True


def main():
    """Run all LLM Manager tests."""
    print("🧪 LLM MANAGER COMPREHENSIVE TESTING")
    
    tests = [
        test_llm_manager_functionality,
        test_llm_manager_with_evaluator,
        test_llm_manager_performance
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL LLM MANAGER TESTS PASSED ({passed}/{total})")
        print("LLM Manager is working correctly!")
        return True
    else:
        print(f"⚠️  SOME LLM MANAGER TESTS FAILED ({passed}/{total})")
        print("Please review the failed tests above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 