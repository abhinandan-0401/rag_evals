"""Test script for the evaluation pipeline system."""

import asyncio
import time
from typing import List, Dict, Any

# Test imports
from src.core.pipeline import (
    EvaluationPipeline, PipelineConfig, ExecutionStrategy, DegradationMode
)
from src.core.types import RAGInput, MetricResult, MetricType
from src.core.base_metric import BaseMetric


class MockMetric(BaseMetric):
    """Mock metric for testing pipeline functionality."""
    
    def __init__(self, name: str, execution_time: float = 0.1, failure_rate: float = 0.0, 
                 requires_llm: bool = False, requires_contexts: bool = False, 
                 requires_answer: bool = True, metric_type: MetricType = MetricType.GENERATION):
        # Set name first since BaseMetric.__init__ calls self.name()
        self._name = name
        self.execution_time = execution_time
        self.failure_rate = failure_rate
        self._requires_llm = requires_llm
        self._requires_contexts = requires_contexts
        self._requires_answer = requires_answer
        self._metric_type = metric_type
        self.call_count = 0
        
        # Initialize parent class after setting required attributes
        super().__init__(llm=None, cache_enabled=False)
        
    def name(self) -> str:
        return self._name
    
    @property
    def metric_type(self) -> MetricType:
        return self._metric_type
    
    @property
    def requires_llm(self) -> bool:
        return self._requires_llm
    
    @property
    def requires_contexts(self) -> bool:
        return self._requires_contexts
    
    @property
    def requires_answer(self) -> bool:
        return self._requires_answer
    
    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Mock evaluation with configurable timing and failure."""
        self.call_count += 1
        
        # Simulate execution time
        await asyncio.sleep(self.execution_time)
        
        # Simulate random failures
        import random
        if random.random() < self.failure_rate:
            raise Exception(f"Mock failure in {self.name()}")
        
        # Return mock result
        score = 0.8 + (0.2 * random.random())  # Random score between 0.8-1.0
        return MetricResult(
            score=score,
            explanation=f"Mock evaluation result for {self.name()}",
            details={"execution_time": self.execution_time, "call_count": self.call_count}
        )
    
    async def evaluate_with_cache(self, rag_input: RAGInput) -> MetricResult:
        """Override to avoid metric_logger dependency in testing."""
        return await self.evaluate(rag_input)


async def test_basic_pipeline():
    """Test basic pipeline functionality."""
    print("🧪 TESTING BASIC PIPELINE FUNCTIONALITY")
    print("=" * 60)
    
    # Create mock metrics
    metrics = [
        MockMetric("faithfulness", 0.1),
        MockMetric("answer_relevance", 0.15),
        MockMetric("context_relevance", 0.08, requires_contexts=True, metric_type=MetricType.RETRIEVAL),
    ]
    
    # Create pipeline
    config = PipelineConfig(
        execution_strategy=ExecutionStrategy.PARALLEL_BY_LEVEL,
        degradation_mode=DegradationMode.GRACEFUL
    )
    pipeline = EvaluationPipeline(config)
    
    # Add metrics without dependencies
    for metric in metrics:
        pipeline.add_metric(metric)
    
    # Create test input
    rag_input = RAGInput(
        query="What is the capital of France?",
        answer="The capital of France is Paris.",
        retrieved_contexts=["Paris is the capital and most populous city of France."]
    )
    
    # Execute pipeline
    start_time = time.time()
    result = await pipeline.execute(rag_input)
    execution_time = time.time() - start_time
    
    # Validate results
    assert result.pipeline_success, "Pipeline should succeed"
    assert len(result.metric_results) == 3, f"Expected 3 results, got {len(result.metric_results)}"
    assert len(result.failed_metrics) == 0, f"No metrics should fail, but got: {result.failed_metrics}"
    
    print(f"✅ Basic pipeline executed successfully in {execution_time:.2f}s")
    print(f"   - Metrics completed: {len(result.metric_results)}")
    print(f"   - Overall execution time: {result.total_execution_time:.2f}s")
    print(f"   - Performance metrics: {result.performance_metrics.get('parallel_efficiency', 'N/A')}")
    print()


async def test_dependency_ordering():
    """Test pipeline with metric dependencies."""
    print("🧪 TESTING DEPENDENCY ORDERING")
    print("=" * 60)
    
    # Create metrics with dependencies
    faithfulness = MockMetric("faithfulness", 0.1, metric_type=MetricType.GENERATION)
    answer_relevance = MockMetric("answer_relevance", 0.1, metric_type=MetricType.GENERATION)
    context_relevance = MockMetric("context_relevance", 0.1, requires_contexts=True, metric_type=MetricType.RETRIEVAL)
    llm_judge = MockMetric("llm_judge", 0.2, metric_type=MetricType.COMPOSITE)
    
    # Create pipeline
    config = PipelineConfig(execution_strategy=ExecutionStrategy.PARALLEL_BY_LEVEL)
    pipeline = EvaluationPipeline(config)
    
    # Add metrics with explicit dependencies
    pipeline.add_metric(faithfulness)
    pipeline.add_metric(answer_relevance)
    pipeline.add_metric(context_relevance)
    pipeline.add_metric(llm_judge, dependencies=["faithfulness", "answer_relevance"], optional_deps=True)
    
    # Build execution order
    execution_order = pipeline.build_execution_order()
    
    print(f"   Execution order (levels): {execution_order}")
    
    # Validate dependency ordering
    assert len(execution_order) == 2, f"Expected 2 dependency levels, got {len(execution_order)}"
    
    # First level should have base metrics
    level_0 = set(execution_order[0])
    expected_level_0 = {"faithfulness", "answer_relevance", "context_relevance"}
    assert level_0 == expected_level_0, f"Level 0 mismatch: {level_0} vs {expected_level_0}"
    
    # Second level should have composite metric
    level_1 = set(execution_order[1])
    expected_level_1 = {"llm_judge"}
    assert level_1 == expected_level_1, f"Level 1 mismatch: {level_1} vs {expected_level_1}"
    
    # Execute pipeline
    rag_input = RAGInput(
        query="Test query",
        answer="Test answer",
        retrieved_contexts=["Test context"]
    )
    
    result = await pipeline.execute(rag_input)
    
    # Validate execution
    assert result.pipeline_success, "Pipeline should succeed"
    assert len(result.metric_results) == 4, f"Expected 4 results, got {len(result.metric_results)}"
    
    print(f"✅ Dependency ordering working correctly")
    print(f"   - Level 0 metrics: {execution_order[0]}")
    print(f"   - Level 1 metrics: {execution_order[1]}")
    print(f"   - All metrics completed successfully")
    print()


async def test_error_handling_and_degradation():
    """Test error handling and graceful degradation."""
    print("🧪 TESTING ERROR HANDLING AND DEGRADATION")
    print("=" * 60)
    
    # Create metrics with some failures
    reliable_metric = MockMetric("reliable_metric", 0.1, failure_rate=0.0)
    failing_metric = MockMetric("failing_metric", 0.1, failure_rate=1.0)  # Always fails
    another_reliable = MockMetric("another_reliable", 0.1, failure_rate=0.0)
    
    # Test graceful degradation
    config = PipelineConfig(
        execution_strategy=ExecutionStrategy.PARALLEL_BY_LEVEL,
        degradation_mode=DegradationMode.GRACEFUL,
        retry_failed_metrics=True,
        max_retries=1
    )
    pipeline = EvaluationPipeline(config)
    
    # Add metrics
    for metric in [reliable_metric, failing_metric, another_reliable]:
        pipeline.add_metric(metric)
    
    # Execute pipeline
    rag_input = RAGInput(
        query="Test query",
        answer="Test answer",
        retrieved_contexts=["Test context"]
    )
    
    result = await pipeline.execute(rag_input)
    
    # Validate graceful degradation
    assert result.pipeline_success, "Pipeline should succeed in graceful mode even with failures"
    assert len(result.failed_metrics) == 1, f"Expected 1 failed metric, got {len(result.failed_metrics)}"
    assert "failing_metric" in result.failed_metrics, "failing_metric should be in failed list"
    
    # Should still have results for successful metrics + degraded result for failed metric
    assert len(result.metric_results) == 3, f"Expected 3 results (2 success + 1 degraded), got {len(result.metric_results)}"
    
    # Check degraded result
    degraded_result = result.metric_results["failing_metric"]
    assert degraded_result.score == 0.0, "Degraded result should have score 0.0"
    assert degraded_result.details.get("degraded_mode") is True, "Should be marked as degraded mode"
    
    print(f"✅ Graceful degradation working correctly")
    print(f"   - Successful metrics: {len(result.metric_results) - len(result.failed_metrics)}")
    print(f"   - Failed metrics: {len(result.failed_metrics)}")
    print(f"   - Pipeline still succeeded: {result.pipeline_success}")
    print()


async def test_execution_strategies():
    """Test different execution strategies."""
    print("🧪 TESTING EXECUTION STRATEGIES")
    print("=" * 60)
    
    # Create test metrics
    metrics = [
        MockMetric(f"metric_{i}", execution_time=0.1) 
        for i in range(5)
    ]
    
    strategies = [
        ExecutionStrategy.SEQUENTIAL,
        ExecutionStrategy.PARALLEL_BY_LEVEL,
        ExecutionStrategy.FULL_PARALLEL
    ]
    
    rag_input = RAGInput(
        query="Test query",
        answer="Test answer",
        retrieved_contexts=["Test context"]
    )
    
    execution_times = {}
    
    for strategy in strategies:
        config = PipelineConfig(execution_strategy=strategy)
        pipeline = EvaluationPipeline(config)
        
        # Add metrics (reset call counts)
        for metric in metrics:
            metric.call_count = 0
            pipeline.add_metric(metric)
        
        # Execute
        start_time = time.time()
        result = await pipeline.execute(rag_input)
        execution_time = time.time() - start_time
        execution_times[strategy.value] = execution_time
        
        # Validate
        assert result.pipeline_success, f"Strategy {strategy.value} should succeed"
        assert len(result.metric_results) == 5, f"Expected 5 results for {strategy.value}"
        
        print(f"   - {strategy.value}: {execution_time:.2f}s")
    
    # Validate performance differences
    # Sequential should be slowest, full parallel should be fastest
    assert execution_times["sequential"] > execution_times["full_parallel"], \
        "Sequential should be slower than full parallel"
    
    print(f"✅ All execution strategies working correctly")
    print()


async def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    print("🧪 TESTING PERFORMANCE MONITORING")
    print("=" * 60)
    
    # Create pipeline with performance monitoring
    config = PipelineConfig(
        execution_strategy=ExecutionStrategy.PARALLEL_BY_LEVEL,
        enable_performance_monitoring=True
    )
    pipeline = EvaluationPipeline(config)
    
    # Add test metrics
    metrics = [
        MockMetric("fast_metric", 0.05),
        MockMetric("slow_metric", 0.2),
        MockMetric("medium_metric", 0.1)
    ]
    
    for metric in metrics:
        pipeline.add_metric(metric)
    
    # Execute multiple times to build history
    rag_input = RAGInput(query="Test query", answer="Test answer", retrieved_contexts=["Test context"])
    
    for i in range(3):
        result = await pipeline.execute(rag_input)
        assert result.pipeline_success, f"Execution {i+1} should succeed"
    
    # Check performance history
    performance_summary = pipeline.get_performance_summary()
    
    assert performance_summary["total_executions"] == 3, "Should have 3 executions in history"
    assert "average_execution_time" in performance_summary, "Should track average execution time"
    assert "average_success_rate" in performance_summary, "Should track success rate"
    assert len(performance_summary["recent_executions"]) == 3, "Should have 3 recent executions"
    
    print(f"✅ Performance monitoring working correctly")
    print(f"   - Total executions: {performance_summary['total_executions']}")
    print(f"   - Average execution time: {performance_summary['average_execution_time']:.3f}s")
    print(f"   - Average success rate: {performance_summary['average_success_rate']:.1%}")
    print()


async def main():
    """Run all pipeline tests."""
    print("🚀 EVALUATION PIPELINE COMPREHENSIVE TESTING")
    print("=" * 80)
    
    test_functions = [
        test_basic_pipeline,
        test_dependency_ordering,
        test_error_handling_and_degradation,
        test_execution_strategies,
        test_performance_monitoring
    ]
    
    passed_tests = 0
    
    for test_func in test_functions:
        try:
            await test_func()
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("=" * 80)
    print(f"🎉 PIPELINE TESTS COMPLETED ({passed_tests}/{len(test_functions)})")
    
    if passed_tests == len(test_functions):
        print("All pipeline tests passed! The evaluation pipeline is working correctly.")
    else:
        print("Some pipeline tests failed. Please check the errors above.")
    
    return passed_tests == len(test_functions)


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 