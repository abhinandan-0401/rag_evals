"""Test script for the performance monitoring system."""

import asyncio
import time
from typing import List, Dict, Any

# Test imports
from src.core.performance import (
    PerformanceAnalyzer, MetricPerformanceData, PipelinePerformanceData,
    PerformanceTrend, SystemResourceMonitor, PerformanceMetricType, PerformanceLevel,
    get_performance_analyzer
)
from src.core.pipeline import EvaluationPipeline, PipelineConfig
from src.core.types import RAGInput, MetricResult, MetricType
from src.core.base_metric import BaseMetric


class MockMetric(BaseMetric):
    """Mock metric for testing performance monitoring."""
    
    def __init__(self, name: str, execution_time: float = 0.1, failure_rate: float = 0.0):
        # Set name first since BaseMetric.__init__ calls self.name()
        self._name = name
        self.execution_time = execution_time
        self.failure_rate = failure_rate
        self._requires_llm = False
        self._requires_contexts = False
        self._requires_answer = True
        self._metric_type = MetricType.GENERATION
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


def test_system_resource_monitor():
    """Test system resource monitoring."""
    print("🧪 TESTING SYSTEM RESOURCE MONITOR")
    print("=" * 60)
    
    monitor = SystemResourceMonitor(sampling_interval=0.01)  # Fast sampling for testing
    
    # Test monitoring
    monitor.start_monitoring()
    
    # Simulate some work
    time.sleep(0.1)
    
    peak_memory, avg_cpu = monitor.stop_monitoring()
    
    # Validate results
    assert peak_memory > 0, "Should have captured memory usage"
    assert avg_cpu >= 0, "CPU usage should be non-negative"
    assert len(monitor.samples) > 0, "Should have collected samples"
    
    print(f"✅ System resource monitoring working correctly")
    print(f"   - Peak memory: {peak_memory:.1f}MB")
    print(f"   - Average CPU: {avg_cpu:.1f}%")
    print(f"   - Samples collected: {len(monitor.samples)}")
    print()


def test_performance_data_structures():
    """Test performance data structures."""
    print("🧪 TESTING PERFORMANCE DATA STRUCTURES")
    print("=" * 60)
    
    # Test MetricPerformanceData
    metric_perf = MetricPerformanceData(
        metric_name="test_metric",
        execution_time=1.5,
        memory_usage_mb=100.0,
        success=True
    )
    
    # Test serialization
    metric_dict = metric_perf.to_dict()
    assert metric_dict["metric_name"] == "test_metric"
    assert metric_dict["execution_time"] == 1.5
    assert metric_dict["success"] is True
    
    # Test PipelinePerformanceData
    pipeline_perf = PipelinePerformanceData(
        pipeline_id="test_pipeline",
        total_execution_time=5.0,
        metrics_count=3,
        successful_metrics=3,
        failed_metrics=0,
        parallel_efficiency=0.8,
        metric_performances=[metric_perf]
    )
    
    # Test calculated properties
    assert pipeline_perf.success_rate == 1.0
    assert pipeline_perf.throughput == 0.6  # 3 metrics / 5 seconds
    
    # Test serialization
    pipeline_dict = pipeline_perf.to_dict()
    assert pipeline_dict["pipeline_id"] == "test_pipeline"
    assert pipeline_dict["success_rate"] == 1.0
    assert pipeline_dict["throughput"] == 0.6
    
    print(f"✅ Performance data structures working correctly")
    print(f"   - Metric performance data validated")
    print(f"   - Pipeline performance data validated")
    print(f"   - Serialization working correctly")
    print()


def test_performance_analyzer():
    """Test performance analyzer functionality."""
    print("🧪 TESTING PERFORMANCE ANALYZER")
    print("=" * 60)
    
    analyzer = PerformanceAnalyzer()
    
    # Create some test performance data
    metric_perfs = [
        MetricPerformanceData("faithfulness", 1.0, success=True),
        MetricPerformanceData("faithfulness", 1.2, success=True),
        MetricPerformanceData("faithfulness", 0.8, success=True),
        MetricPerformanceData("answer_relevance", 0.5, success=True),
        MetricPerformanceData("answer_relevance", 0.6, success=False, error_message="Test error")
    ]
    
    pipeline_perfs = [
        PipelinePerformanceData(
            pipeline_id="test_1",
            total_execution_time=3.0,
            metrics_count=2,
            successful_metrics=2,
            failed_metrics=0,
            parallel_efficiency=0.7,
            metric_performances=metric_perfs[:3]
        ),
        PipelinePerformanceData(
            pipeline_id="test_2",
            total_execution_time=3.5,
            metrics_count=2,
            successful_metrics=1,
            failed_metrics=1,
            parallel_efficiency=0.6,
            metric_performances=metric_perfs[3:]
        )
    ]
    
    # Add data to analyzer
    for pipeline_perf in pipeline_perfs:
        analyzer.add_pipeline_performance(pipeline_perf)
    
    # Test metric analysis
    faithfulness_analysis = analyzer.analyze_metric_performance("faithfulness")
    assert "metric_name" in faithfulness_analysis
    assert faithfulness_analysis["metric_name"] == "faithfulness"
    assert faithfulness_analysis["sample_size"] == 3
    assert faithfulness_analysis["success_rate"] == 1.0
    assert "avg_execution_time" in faithfulness_analysis
    
    # Test pipeline analysis
    pipeline_analysis = analyzer.analyze_pipeline_performance()
    assert "sample_size" in pipeline_analysis
    assert pipeline_analysis["sample_size"] == 2
    assert "avg_success_rate" in pipeline_analysis
    assert "optimization_suggestions" in pipeline_analysis
    
    print(f"✅ Performance analyzer working correctly")
    print(f"   - Metric analysis: {faithfulness_analysis['performance_level']}")
    print(f"   - Pipeline analysis: {pipeline_analysis['performance_level']}")
    print(f"   - Optimization suggestions: {len(pipeline_analysis['optimization_suggestions'])}")
    print()


def test_trend_detection():
    """Test performance trend detection."""
    print("🧪 TESTING TREND DETECTION")
    print("=" * 60)
    
    analyzer = PerformanceAnalyzer()
    
    # Create performance data showing degrading performance
    base_time = time.time()
    metric_perfs = []
    
    # First window: fast performance
    for i in range(10):
        metric_perfs.append(MetricPerformanceData(
            "test_metric", 
            execution_time=1.0 + (i * 0.01),  # Slightly increasing
            success=True,
            timestamp=base_time + i
        ))
    
    # Second window: slower performance (degradation)
    for i in range(10):
        metric_perfs.append(MetricPerformanceData(
            "test_metric", 
            execution_time=1.5 + (i * 0.01),  # Significantly higher
            success=True,
            timestamp=base_time + 10 + i
        ))
    
    # Add to analyzer
    for metric_perf in metric_perfs:
        analyzer.metric_history["test_metric"].append(metric_perf)
    
    # Analyze trends
    analysis = analyzer.analyze_metric_performance("test_metric", recent_count=10)
    trends = analysis.get("trends", [])
    
    # Should detect degrading performance
    execution_time_trends = [t for t in trends if t.metric_type == PerformanceMetricType.EXECUTION_TIME]
    assert len(execution_time_trends) > 0, "Should detect execution time trend"
    
    trend = execution_time_trends[0]
    assert trend.trend_direction == "degrading", "Should detect degrading performance"
    assert trend.change_percentage > 0, "Should show positive change (degradation)"
    
    print(f"✅ Trend detection working correctly")
    print(f"   - Detected trend: {trend}")
    print(f"   - Change: {trend.change_percentage:+.1f}%")
    print(f"   - Confidence: {trend.confidence:.2f}")
    print()


async def test_pipeline_performance_integration():
    """Test performance monitoring integration with pipeline."""
    print("🧪 TESTING PIPELINE PERFORMANCE INTEGRATION")
    print("=" * 60)
    
    # Create pipeline with performance monitoring enabled
    config = PipelineConfig(enable_performance_monitoring=True)
    pipeline = EvaluationPipeline(config)
    
    # Add test metrics
    metrics = [
        MockMetric("fast_metric", 0.05),
        MockMetric("slow_metric", 0.15),
        MockMetric("medium_metric", 0.1)
    ]
    
    for metric in metrics:
        pipeline.add_metric(metric)
    
    # Execute pipeline multiple times
    rag_input = RAGInput(
        query="Test query",
        answer="Test answer",
        retrieved_contexts=["Test context"]
    )
    
    for i in range(3):
        result = await pipeline.execute(rag_input)
        assert result.pipeline_success, f"Pipeline execution {i+1} should succeed"
    
    # Get performance analysis
    performance_analysis = pipeline.get_performance_analysis()
    assert "pipeline_summary" in performance_analysis, "Should have pipeline summary"
    assert "metric_summaries" in performance_analysis, "Should have metric summaries"
    
    pipeline_summary = performance_analysis["pipeline_summary"]
    assert "avg_execution_time" in pipeline_summary, "Should have average execution time"
    assert "performance_level" in pipeline_summary, "Should have performance level"
    
    # Test metric-specific analysis
    fast_metric_analysis = pipeline.get_metric_performance_analysis("fast_metric")
    assert fast_metric_analysis["metric_name"] == "fast_metric", "Should analyze specific metric"
    
    # Test optimization suggestions
    suggestions = pipeline.get_optimization_suggestions()
    assert isinstance(suggestions, list), "Should return list of suggestions"
    
    print(f"✅ Pipeline performance integration working correctly")
    print(f"   - Pipeline executions: 3")
    print(f"   - Performance level: {pipeline_summary.get('performance_level', 'unknown')}")
    print(f"   - Average execution time: {pipeline_summary.get('avg_execution_time', 0):.3f}s")
    print(f"   - Optimization suggestions: {len(suggestions)}")
    print()


def test_performance_report_generation():
    """Test comprehensive performance report generation."""
    print("🧪 TESTING PERFORMANCE REPORT GENERATION")
    print("=" * 60)
    
    analyzer = get_performance_analyzer()
    
    # Generate a performance report
    report = analyzer.generate_performance_report(detailed=True)
    
    # Validate report structure
    required_keys = ["timestamp", "pipeline_summary", "metric_summaries", "overall_health"]
    for key in required_keys:
        assert key in report, f"Report missing key: {key}"
    
    # Check if detailed data is included
    if "detailed_history" in report:
        assert "pipeline_history" in report["detailed_history"]
        assert "metric_history" in report["detailed_history"]
    
    print(f"✅ Performance report generation working correctly")
    print(f"   - Report keys: {list(report.keys())}")
    print(f"   - Overall health: {report['overall_health']}")
    print(f"   - Metrics analyzed: {len(report['metric_summaries'])}")
    print()


def main():
    """Run all performance monitoring tests."""
    print("🚀 PERFORMANCE MONITORING COMPREHENSIVE TESTING")
    print("=" * 80)
    
    test_functions = [
        test_system_resource_monitor,
        test_performance_data_structures,
        test_performance_analyzer,
        test_trend_detection,
        test_pipeline_performance_integration,
        test_performance_report_generation
    ]
    
    passed_tests = 0
    
    for test_func in test_functions:
        try:
            if asyncio.iscoroutinefunction(test_func):
                asyncio.run(test_func())
            else:
                test_func()
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("=" * 80)
    print(f"🎉 PERFORMANCE TESTS COMPLETED ({passed_tests}/{len(test_functions)})")
    
    if passed_tests == len(test_functions):
        print("All performance monitoring tests passed! The system is working correctly.")
        return True
    else:
        print("Some performance tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 