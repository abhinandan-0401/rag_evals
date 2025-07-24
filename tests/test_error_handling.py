#!/usr/bin/env python3
"""
Test script to demonstrate error handling framework functionality.
Tests error classification, recovery strategies, graceful degradation, and integration.
"""

import sys
import asyncio
import time
from src.core.error_handler import (
    ErrorSeverity, ErrorCategory, ErrorClassifier, 
    ErrorRecoveryManager, with_error_recovery, error_context, sync_error_context
)
from src.core.exceptions import LLMError, MetricError, LLMConnectionError


def test_error_classification():
    """Test error classification functionality."""
    print("=" * 60)
    print("TESTING ERROR CLASSIFICATION")
    print("=" * 60)
    
    classifier = ErrorClassifier()
    
    # Test different types of errors
    test_cases = [
        (LLMConnectionError("Connection timeout"), ErrorCategory.NETWORK),
        (LLMError("Authentication failed"), ErrorCategory.AUTHENTICATION),
        (LLMError("Rate limit exceeded"), ErrorCategory.RATE_LIMIT),
        (MetricError("Invalid input format"), ErrorCategory.VALIDATION),
        (Exception("Unknown error"), ErrorCategory.UNKNOWN)
    ]
    
    print("\n1. Testing error categorization...")
    
    for error, expected_category in test_cases:
        context = classifier.classify(error, "test_component")
        
        if context.category == expected_category:
            print(f"✅ {error.__class__.__name__} -> {context.category.value}")
        else:
            print(f"❌ {error.__class__.__name__} -> {context.category.value} (expected {expected_category.value})")
            return False
    
    print("\n2. Testing severity assignment...")
    
    # Test severity levels
    severity_tests = [
        (LLMConnectionError("timeout"), ErrorSeverity.MEDIUM),
        (MetricError("config error"), ErrorSeverity.MEDIUM)
    ]
    
    for error, expected_severity in severity_tests:
        context = classifier.classify(error, "test_component")
        
        if context.severity == expected_severity:
            print(f"✅ {error.__class__.__name__} -> {context.severity.value}")
        else:
            print(f"❌ {error.__class__.__name__} -> {context.severity.value} (expected {expected_severity.value})")
            return False
    
    return True


def test_recovery_strategies():
    """Test recovery strategies and retry logic."""
    print("\n" + "=" * 60)
    print("TESTING RECOVERY STRATEGIES")
    print("=" * 60)
    
    recovery_manager = ErrorRecoveryManager()
    
    # Test 1: Successful operation (no recovery needed)
    print("\n1. Testing successful operation...")
    
    def successful_operation():
        return "success"
    
    result = recovery_manager.handle_sync(
        successful_operation,
        component="test",
        operation_name="success_test"
    )
    
    if result == "success":
        print("✅ Successful operation handled correctly")
    else:
        print("❌ Successful operation failed")
        return False
    
    # Test 2: Operation that fails then succeeds on retry
    print("\n2. Testing retry mechanism...")
    
    attempt_count = 0
    
    def retry_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:
            raise LLMConnectionError("Temporary connection error")
        return "success_after_retry"
    
    attempt_count = 0  # Reset
    result = recovery_manager.handle_sync(
        retry_operation,
        component="test",
        operation_name="retry_test"
    )
    
    if result == "success_after_retry" and attempt_count == 2:
        print("✅ Retry mechanism working correctly")
    else:
        print(f"❌ Retry mechanism failed (result: {result}, attempts: {attempt_count})")
        return False
    
    # Test 3: Operation that always fails -> fallback
    print("\n3. Testing fallback mechanism...")
    
    def always_fail_operation():
        raise LLMError("Persistent error")
    
    result = recovery_manager.handle_sync(
        always_fail_operation,
        component="test",
        operation_name="fallback_test"
    )
    
    # Should return fallback value (None for most strategies)
    if result is None:
        print("✅ Fallback mechanism working correctly")
    else:
        print(f"❌ Fallback mechanism failed (result: {result})")
        return False
    
    return True


async def test_async_recovery():
    """Test async error recovery."""
    print("\n" + "=" * 60)
    print("TESTING ASYNC RECOVERY")
    print("=" * 60)
    
    recovery_manager = ErrorRecoveryManager()
    
    # Test async operation with recovery
    print("\n1. Testing async retry mechanism...")
    
    async_attempt_count = 0
    
    async def async_retry_operation():
        nonlocal async_attempt_count
        async_attempt_count += 1
        if async_attempt_count < 2:
            raise LLMConnectionError("Async connection error")
        return "async_success"
    
    async_attempt_count = 0  # Reset
    result = await recovery_manager.handle_async(
        async_retry_operation,
        component="test",
        operation_name="async_retry_test"
    )
    
    if result == "async_success" and async_attempt_count == 2:
        print("✅ Async retry mechanism working correctly")
        return True
    else:
        print(f"❌ Async retry mechanism failed (result: {result}, attempts: {async_attempt_count})")
        return False


def test_error_decorators():
    """Test error handling decorators."""
    print("\n" + "=" * 60)
    print("TESTING ERROR DECORATORS")
    print("=" * 60)
    
    # Test sync decorator
    print("\n1. Testing sync decorator...")
    
    decorator_attempt_count = 0
    
    @with_error_recovery("test_component", "decorated_operation")
    def decorated_function():
        nonlocal decorator_attempt_count
        decorator_attempt_count += 1
        if decorator_attempt_count < 2:
            raise LLMConnectionError("Decorator test error")
        return "decorator_success"
    
    decorator_attempt_count = 0  # Reset
    result = decorated_function()
    
    if result == "decorator_success" and decorator_attempt_count == 2:
        print("✅ Sync decorator working correctly")
    else:
        print(f"❌ Sync decorator failed (result: {result}, attempts: {decorator_attempt_count})")
        return False
    
    # Test async decorator
    print("\n2. Testing async decorator...")
    
    async_decorator_attempt_count = 0
    
    @with_error_recovery("test_component", "async_decorated_operation")
    async def async_decorated_function():
        nonlocal async_decorator_attempt_count
        async_decorator_attempt_count += 1
        if async_decorator_attempt_count < 2:
            raise LLMConnectionError("Async decorator test error")
        return "async_decorator_success"
    
    async def test_async_decorator():
        nonlocal async_decorator_attempt_count
        async_decorator_attempt_count = 0  # Reset
        result = await async_decorated_function()
        return result, async_decorator_attempt_count
    
    result, attempts = asyncio.run(test_async_decorator())
    
    if result == "async_decorator_success" and attempts == 2:
        print("✅ Async decorator working correctly")
        return True
    else:
        print(f"❌ Async decorator failed (result: {result}, attempts: {attempts})")
        return False


def test_context_managers():
    """Test error context managers."""
    print("\n" + "=" * 60)
    print("TESTING CONTEXT MANAGERS")
    print("=" * 60)
    
    # Test sync context manager
    print("\n1. Testing sync context manager...")
    
    try:
        with sync_error_context("test_component", "context_test"):
            # This should complete successfully
            pass
        print("✅ Sync context manager working correctly")
    except Exception as e:
        print(f"❌ Sync context manager failed: {e}")
        return False
    
    # Test async context manager
    print("\n2. Testing async context manager...")
    
    async def test_async_context():
        try:
            async with error_context("test_component", "async_context_test"):
                # This should complete successfully
                pass
            return True
        except Exception as e:
            print(f"❌ Async context manager failed: {e}")
            return False
    
    result = asyncio.run(test_async_context())
    
    if result:
        print("✅ Async context manager working correctly")
        return True
    else:
        return False


def test_integration():
    """Test integration with other components."""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATION")
    print("=" * 60)
    
    print("\n1. Testing error handler import in core module...")
    
    try:
        from src.core import ErrorSeverity, ErrorCategory, get_recovery_manager
        print("✅ Error handler components imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import error handler components: {e}")
        return False
    
    print("\n2. Testing error handler availability...")
    
    try:
        recovery_manager = get_recovery_manager()
        if recovery_manager is not None:
            print("✅ Error recovery manager accessible")
        else:
            print("❌ Error recovery manager is None")
            return False
    except Exception as e:
        print(f"❌ Failed to get recovery manager: {e}")
        return False
    
    return True


def main():
    """Run all error handling tests."""
    print("🧪 ERROR HANDLING FRAMEWORK COMPREHENSIVE TESTING")
    
    tests = [
        test_error_classification,
        test_recovery_strategies,
        lambda: asyncio.run(test_async_recovery()),
        test_error_decorators,
        test_context_managers,
        test_integration
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
        print(f"🎉 ALL ERROR HANDLING TESTS PASSED ({passed}/{total})")
        print("Error handling framework is working correctly!")
        return True
    else:
        print(f"⚠️  SOME ERROR HANDLING TESTS FAILED ({passed}/{total})")
        print("Please review the failed tests above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 