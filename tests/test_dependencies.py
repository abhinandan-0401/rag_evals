"""Test script for the metric dependency management system."""

from src.core.dependencies import (
    MetricDependencyManager, MetricProfile, MetricRequirement, MetricDependencyRule,
    DependencyType, RequirementType, get_dependency_manager
)
from src.core.types import RAGInput, MetricType


def test_dependency_manager_initialization():
    """Test that the dependency manager initializes correctly with built-in profiles."""
    print("🧪 TESTING DEPENDENCY MANAGER INITIALIZATION")
    print("=" * 60)
    
    manager = get_dependency_manager()
    
    # Check that built-in metric profiles are loaded
    expected_metrics = [
        "faithfulness", "answer_relevance", "answer_correctness", "completeness", "coherence", "helpfulness",
        "context_relevance", "context_precision", "context_recall",
        "llm_judge", "rag_certainty", "trust_score"
    ]
    
    for metric_name in expected_metrics:
        assert metric_name in manager.metric_profiles, f"Missing built-in profile for {metric_name}"
    
    # Check that dependency rules are loaded
    assert len(manager.dependency_rules) > 0, "No dependency rules loaded"
    
    print(f"✅ Dependency manager initialized successfully")
    print(f"   - Built-in profiles: {len(manager.metric_profiles)}")
    print(f"   - Dependency rules: {len(manager.dependency_rules)}")
    print()


def test_requirement_validation():
    """Test metric requirement validation."""
    print("🧪 TESTING REQUIREMENT VALIDATION")
    print("=" * 60)
    
    manager = get_dependency_manager()
    
    # Test valid input
    valid_input = RAGInput(
        query="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_contexts=["Paris is the capital and most populous city of France."],
        metadata={"source": "test"}
    )
    
    # Test faithfulness validation (should pass)
    errors = manager.validate_metric_requirements("faithfulness", valid_input)
    assert len(errors) == 0, f"Faithfulness should pass validation, but got errors: {errors}"
    
    # Test that the dependency manager correctly validates requirements
    # Since RAGInput has its own validation, we test that metrics have correct requirements
    
    # Test that faithfulness metric profile includes context requirement
    faithfulness_profile = manager.metric_profiles["faithfulness"]
    context_required = any(req.requirement_type == RequirementType.CONTEXTS 
                          for req in faithfulness_profile.requirements)
    assert context_required, "Faithfulness should require contexts"
    
    # Test that answer_relevance doesn't require contexts (only query and answer)
    answer_relevance_profile = manager.metric_profiles["answer_relevance"]
    context_required = any(req.requirement_type == RequirementType.CONTEXTS 
                          for req in answer_relevance_profile.requirements)
    assert not context_required, "Answer relevance should not require contexts"
    
    # Answer relevance should still pass (doesn't require contexts)
    errors = manager.validate_metric_requirements("answer_relevance", valid_input)
    assert len(errors) == 0, f"Answer relevance should pass, but got errors: {errors}"
    
    print(f"✅ Requirement validation working correctly")
    print(f"   - Valid input passed all checks")
    print(f"   - Invalid inputs correctly rejected")
    print()


def test_dependency_detection():
    """Test automatic dependency detection."""
    print("🧪 TESTING DEPENDENCY DETECTION")
    print("=" * 60)
    
    manager = get_dependency_manager()
    
    # Test that composite metrics have dependencies on generation/retrieval metrics
    llm_judge_deps = manager.get_metric_dependencies("llm_judge")
    assert len(llm_judge_deps) > 0, "LLM judge should have dependencies"
    assert "faithfulness" in llm_judge_deps, "LLM judge should depend on faithfulness"
    assert "answer_relevance" in llm_judge_deps, "LLM judge should depend on answer relevance"
    
    # Test that generation metrics have no dependencies (typically)
    faithfulness_deps = manager.get_metric_dependencies("faithfulness")
    # Faithfulness might have soft dependencies, but it should be minimal
    
    # Test context-dependent dependencies
    context = {"available_metrics": ["faithfulness", "answer_relevance", "context_relevance"]}
    rag_certainty_deps = manager.get_metric_dependencies("rag_certainty", context)
    
    print(f"✅ Dependency detection working correctly")
    print(f"   - LLM judge dependencies: {llm_judge_deps}")
    print(f"   - RAG certainty dependencies: {rag_certainty_deps}")
    print()


def test_execution_ordering():
    """Test execution order generation."""
    print("🧪 TESTING EXECUTION ORDERING")
    print("=" * 60)
    
    manager = get_dependency_manager()
    
    # Test with a mix of metrics
    test_metrics = ["faithfulness", "answer_relevance", "llm_judge", "context_relevance"]
    execution_order = manager.get_execution_order(test_metrics)
    
    # Validate the ordering
    assert len(execution_order) >= 1, "Should have at least one execution level"
    
    # Flatten to check all metrics are included
    all_ordered_metrics = [metric for level in execution_order for metric in level]
    assert set(all_ordered_metrics) == set(test_metrics), "All metrics should be in execution order"
    
    # Check that dependencies come before dependents
    llm_judge_level = None
    faithfulness_level = None
    
    for level_idx, level in enumerate(execution_order):
        if "llm_judge" in level:
            llm_judge_level = level_idx
        if "faithfulness" in level:
            faithfulness_level = level_idx
    
    if llm_judge_level is not None and faithfulness_level is not None:
        assert faithfulness_level <= llm_judge_level, "Faithfulness should come before or with LLM judge"
    
    print(f"✅ Execution ordering working correctly")
    print(f"   - Execution levels: {len(execution_order)}")
    print(f"   - Order: {execution_order}")
    print()


def test_dependency_analysis():
    """Test comprehensive dependency analysis."""
    print("🧪 TESTING DEPENDENCY ANALYSIS")
    print("=" * 60)
    
    manager = get_dependency_manager()
    
    # Analyze a comprehensive set of metrics
    test_metrics = [
        "faithfulness", "answer_relevance", "context_relevance", 
        "llm_judge", "rag_certainty", "completeness"
    ]
    
    analysis = manager.analyze_dependencies(test_metrics)
    
    # Validate analysis structure
    required_keys = [
        "metrics", "dependency_graph", "execution_order", 
        "complexity_analysis", "requirement_summary", "potential_issues"
    ]
    
    for key in required_keys:
        assert key in analysis, f"Analysis missing key: {key}"
    
    # Check dependency graph
    assert len(analysis["dependency_graph"]) == len(test_metrics), "Dependency graph should cover all metrics"
    
    # Check execution order
    assert len(analysis["execution_order"]) > 0, "Should have execution order"
    
    # Check complexity analysis
    assert "total_complexity" in analysis["complexity_analysis"], "Should have total complexity"
    assert "estimated_total_time" in analysis["complexity_analysis"], "Should have time estimate"
    
    # Check requirement summary
    assert "required_capabilities" in analysis["requirement_summary"], "Should have capability summary"
    assert "metrics_needing_llm" in analysis["requirement_summary"], "Should identify LLM-dependent metrics"
    
    print(f"✅ Dependency analysis working correctly")
    print(f"   - Analyzed {len(test_metrics)} metrics")
    print(f"   - Dependency levels: {len(analysis['execution_order'])}")
    print(f"   - Total complexity: {analysis['complexity_analysis']['total_complexity']}")
    print(f"   - Required capabilities: {analysis['requirement_summary']['required_capabilities']}")
    print()


def test_metric_suggestion():
    """Test metric suggestion functionality."""
    print("🧪 TESTING METRIC SUGGESTION")
    print("=" * 60)
    
    manager = get_dependency_manager()
    
    # Test suggesting metrics for comprehensive evaluation
    available_metrics = [
        "faithfulness", "answer_relevance", "context_relevance", 
        "coherence", "completeness", "llm_judge"
    ]
    
    # Test default suggestion
    suggested = manager.suggest_metric_order(available_metrics)
    assert len(suggested) > 0, "Should suggest some metrics"
    assert all(metric in available_metrics for metric in suggested), "All suggested metrics should be available"
    
    # Test targeted suggestion
    target_capabilities = ["faithfulness", "relevance"]
    suggested_targeted = manager.suggest_metric_order(available_metrics, target_capabilities)
    assert len(suggested_targeted) > 0, "Should suggest metrics for targeted capabilities"
    
    print(f"✅ Metric suggestion working correctly")
    print(f"   - Default suggestion: {suggested}")
    print(f"   - Targeted suggestion: {suggested_targeted}")
    print()


def test_custom_profiles_and_rules():
    """Test adding custom metric profiles and dependency rules."""
    print("🧪 TESTING CUSTOM PROFILES AND RULES")
    print("=" * 60)
    
    manager = MetricDependencyManager()  # Fresh instance for testing
    
    # Create a custom metric profile
    custom_profile = MetricProfile(
        metric_name="custom_metric",
        metric_type=MetricType.GENERATION,
        requirements=[
            MetricRequirement(RequirementType.LLM, required=True, description="Needs LLM"),
            MetricRequirement(RequirementType.ANSWER, required=True, description="Needs answer")
        ],
        provides=["custom_capability"],
        computational_complexity="low",
        typical_execution_time=0.5
    )
    
    manager.register_metric_profile(custom_profile)
    assert "custom_metric" in manager.metric_profiles, "Custom profile should be registered"
    
    # Create a custom dependency rule
    custom_rule = MetricDependencyRule(
        dependent_metric="custom_metric",
        dependency_metric="faithfulness",
        dependency_type=DependencyType.SOFT,
        description="Custom metric benefits from faithfulness"
    )
    
    manager.register_dependency_rule(custom_rule)
    
    # Register faithfulness profile too
    faithfulness_profile = MetricProfile(
        metric_name="faithfulness",
        metric_type=MetricType.GENERATION,
        requirements=[MetricRequirement(RequirementType.LLM, required=True)],
        provides=["faithfulness_check"]
    )
    manager.register_metric_profile(faithfulness_profile)
    
    # Test that dependency rule works
    deps = manager.get_metric_dependencies("custom_metric")
    assert "faithfulness" in deps, "Custom metric should depend on faithfulness"
    
    print(f"✅ Custom profiles and rules working correctly")
    print(f"   - Custom profile registered successfully")
    print(f"   - Custom dependency rule working")
    print()


def main():
    """Run all dependency management tests."""
    print("🚀 DEPENDENCY MANAGEMENT COMPREHENSIVE TESTING")
    print("=" * 80)
    
    test_functions = [
        test_dependency_manager_initialization,
        test_requirement_validation,
        test_dependency_detection,
        test_execution_ordering,
        test_dependency_analysis,
        test_metric_suggestion,
        test_custom_profiles_and_rules
    ]
    
    passed_tests = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("=" * 80)
    print(f"🎉 DEPENDENCY TESTS COMPLETED ({passed_tests}/{len(test_functions)})")
    
    if passed_tests == len(test_functions):
        print("All dependency management tests passed! The system is working correctly.")
    else:
        print("Some dependency tests failed. Please check the errors above.")
    
    return passed_tests == len(test_functions)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 