"""Advanced evaluation pipeline for RAG metrics with dependency management."""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from enum import Enum
import logging

from .base_metric import BaseMetric
from .types import RAGInput, MetricResult, MetricType
from .exceptions import MetricError, EvaluatorError
from .logging import get_logger
from .performance import (
    SystemResourceMonitor, MetricPerformanceData, PipelinePerformanceData,
    get_performance_analyzer
)


class ExecutionStrategy(Enum):
    """Strategy for pipeline execution."""
    SEQUENTIAL = "sequential"
    PARALLEL_BY_LEVEL = "parallel_by_level"
    FULL_PARALLEL = "full_parallel"


class DegradationMode(Enum):
    """Mode for handling metric failures."""
    FAIL_FAST = "fail_fast"  # Stop on first failure
    CONTINUE = "continue"  # Continue with remaining metrics
    GRACEFUL = "graceful"  # Use fallback strategies


@dataclass
class MetricDependency:
    """Defines a dependency between metrics."""
    metric_name: str
    depends_on: List[str]
    optional: bool = False  # If True, missing dependencies don't block execution
    
    
@dataclass
class PipelineConfig:
    """Configuration for the evaluation pipeline."""
    execution_strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL_BY_LEVEL
    degradation_mode: DegradationMode = DegradationMode.GRACEFUL
    max_parallel_metrics: int = 10
    metric_timeout: float = 60.0  # seconds
    enable_performance_monitoring: bool = True
    enable_result_caching: bool = True
    retry_failed_metrics: bool = True
    max_retries: int = 2


@dataclass
class MetricExecution:
    """Tracks execution state of a metric."""
    metric: BaseMetric
    status: str = "pending"  # pending, running, completed, failed, skipped
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[MetricResult] = None
    error: Optional[Exception] = None
    retry_count: int = 0
    dependencies_met: bool = False
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    metric_results: Dict[str, MetricResult]
    execution_stats: Dict[str, Any]
    performance_metrics: Dict[str, float]
    failed_metrics: List[str]
    skipped_metrics: List[str]
    total_execution_time: float
    pipeline_success: bool


class EvaluationPipeline:
    """Advanced evaluation pipeline with dependency management and performance monitoring."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the evaluation pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self.logger = get_logger("pipeline")
        
        # Pipeline state
        self.metrics: List[BaseMetric] = []
        self.dependencies: Dict[str, MetricDependency] = {}
        self.execution_order: List[List[str]] = []  # List of dependency levels
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
    def add_metric(self, metric: BaseMetric, dependencies: Optional[List[str]] = None, optional_deps: bool = False):
        """Add a metric to the pipeline.
        
        Args:
            metric: The metric to add
            dependencies: List of metric names this metric depends on
            optional_deps: Whether dependencies are optional
        """
        metric_name = metric.name()
        
        # Check for duplicates
        existing_names = [m.name() for m in self.metrics]
        if metric_name in existing_names:
            raise ValueError(f"Metric {metric_name} already added to pipeline")
        
        self.metrics.append(metric)
        
        if dependencies:
            self.dependencies[metric_name] = MetricDependency(
                metric_name=metric_name,
                depends_on=dependencies,
                optional=optional_deps
            )
        
        self.logger.debug(f"Added metric {metric_name} with dependencies: {dependencies or 'none'}")
    
    def add_metrics(self, metrics: List[BaseMetric], auto_detect_dependencies: bool = True):
        """Add multiple metrics to the pipeline.
        
        Args:
            metrics: List of metrics to add
            auto_detect_dependencies: Whether to auto-detect dependencies based on metric types
        """
        for metric in metrics:
            dependencies = []
            optional_deps = False
            
            if auto_detect_dependencies:
                dependencies, optional_deps = self._detect_dependencies(metric)
            
            self.add_metric(metric, dependencies, optional_deps)
    
    def _detect_dependencies(self, metric: BaseMetric) -> Tuple[List[str], bool]:
        """Auto-detect dependencies for a metric based on its type and characteristics.
        
        Args:
            metric: The metric to analyze
            
        Returns:
            Tuple of (dependencies, optional_flag)
        """
        from .dependencies import get_dependency_manager
        
        dependency_manager = get_dependency_manager()
        
        # Get existing metric names for context
        existing_metrics = [m.name() for m in self.metrics]
        
        # Get dependencies from the dependency manager
        dependencies = dependency_manager.get_metric_dependencies(
            metric.name(), 
            context={"available_metrics": existing_metrics}
        )
        
        # Filter to only include metrics that are actually available
        available_dependencies = [dep for dep in dependencies if dep in existing_metrics]
        
        # For auto-detected dependencies, they are typically optional
        optional = len(available_dependencies) > 0
        
        return available_dependencies, optional
    
    def build_execution_order(self) -> List[List[str]]:
        """Build execution order using topological sort.
        
        Returns:
            List of dependency levels, where each level can be executed in parallel
        """
        # Create dependency graph
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        all_metrics = set(m.name() for m in self.metrics)
        
        # Initialize in-degree for all metrics
        for metric_name in all_metrics:
            in_degree[metric_name] = 0
        
        # Build graph and calculate in-degrees
        for metric_name, dependency in self.dependencies.items():
            for dep in dependency.depends_on:
                if dep in all_metrics or not dependency.optional:
                    if dep in all_metrics:
                        graph[dep].add(metric_name)
                        in_degree[metric_name] += 1
                    elif not dependency.optional:
                        raise ValueError(f"Required dependency {dep} not found for metric {metric_name}")
        
        # Topological sort using Kahn's algorithm
        levels = []
        queue = deque([name for name in all_metrics if in_degree[name] == 0])
        
        while queue:
            # Current level contains all metrics with no remaining dependencies
            current_level = list(queue)
            levels.append(current_level)
            queue.clear()
            
            # Process current level
            for metric_name in current_level:
                for neighbor in graph[metric_name]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        # Check for cycles
        processed_metrics = set()
        for level in levels:
            processed_metrics.update(level)
        
        if len(processed_metrics) != len(all_metrics):
            unprocessed = all_metrics - processed_metrics
            raise ValueError(f"Circular dependencies detected involving metrics: {unprocessed}")
        
        self.execution_order = levels
        self.logger.info(f"Built execution order with {len(levels)} dependency levels")
        return levels
    
    async def execute(self, rag_input: RAGInput) -> PipelineResult:
        """Execute the evaluation pipeline.
        
        Args:
            rag_input: The RAG input to evaluate
            
        Returns:
            PipelineResult with all metric results and performance data
        """
        start_time = time.time()
        
        # Build execution order if not already built
        if not self.execution_order:
            self.build_execution_order()
        
        # Validate metric requirements using dependency manager
        self._validate_metric_requirements(rag_input)
        
        # Initialize execution tracking
        executions = {m.name(): MetricExecution(metric=m) for m in self.metrics}
        
        # Initialize performance monitoring
        resource_monitor = SystemResourceMonitor()
        performance_analyzer = get_performance_analyzer()
        
        # Track performance
        performance_data = {
            "pipeline_start": start_time,
            "level_times": [],
            "metric_times": {},
            "parallel_efficiency": 0.0
        }
        
        # Start resource monitoring
        if self.config.enable_performance_monitoring:
            resource_monitor.start_monitoring()
        
        self.logger.info(f"Starting pipeline execution with {len(self.metrics)} metrics in {len(self.execution_order)} levels")
        
        try:
            if self.config.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                await self._execute_sequential(executions, rag_input, performance_data)
            elif self.config.execution_strategy == ExecutionStrategy.PARALLEL_BY_LEVEL:
                await self._execute_by_levels(executions, rag_input, performance_data)
            else:  # FULL_PARALLEL
                await self._execute_full_parallel(executions, rag_input, performance_data)
                
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            if self.config.degradation_mode == DegradationMode.FAIL_FAST:
                raise EvaluatorError(f"Pipeline execution failed: {str(e)}", original_error=e)
        
        # Stop resource monitoring
        peak_memory, avg_cpu = 0.0, 0.0
        if self.config.enable_performance_monitoring:
            peak_memory, avg_cpu = resource_monitor.stop_monitoring()
        
        # Compile results
        total_time = time.time() - start_time
        result = self._compile_results(executions, performance_data, total_time)
        
        # Store detailed performance data
        if self.config.enable_performance_monitoring:
            # Create detailed performance data
            metric_performances = []
            for metric_name, execution in executions.items():
                metric_perf = MetricPerformanceData(
                    metric_name=metric_name,
                    execution_time=execution.execution_time or 0.0,
                    memory_usage_mb=peak_memory / len(executions) if executions else 0.0,  # Approximate
                    cpu_usage_percent=avg_cpu,
                    success=execution.status == "completed",
                    error_message=str(execution.error) if execution.error else None,
                    cache_hit=False,  # Would need to track this from metrics
                    retry_count=execution.retry_count
                )
                metric_performances.append(metric_perf)
            
            # Create pipeline performance data
            pipeline_perf = PipelinePerformanceData(
                pipeline_id=f"pipeline_{int(start_time)}",
                total_execution_time=total_time,
                metrics_count=len(self.metrics),
                successful_metrics=len([e for e in executions.values() if e.status == "completed"]),
                failed_metrics=len([e for e in executions.values() if e.status == "failed"]),
                parallel_efficiency=performance_data.get("parallel_efficiency", 0.0),
                memory_peak_mb=peak_memory,
                cpu_average_percent=avg_cpu,
                execution_strategy=self.config.execution_strategy.value,
                metric_performances=metric_performances
            )
            
            # Add to performance analyzer
            performance_analyzer.add_pipeline_performance(pipeline_perf)
            
            # Store legacy performance history for backward compatibility
            self.performance_history.append({
                "timestamp": start_time,
                "total_time": total_time,
                "metrics_count": len(self.metrics),
                "successful_metrics": len(result.metric_results),
                "failed_metrics": len(result.failed_metrics),
                "execution_strategy": self.config.execution_strategy.value
            })
        
        self.logger.info(f"Pipeline execution completed in {total_time:.2f}s")
        return result
    
    async def _execute_by_levels(self, executions: Dict[str, MetricExecution], 
                               rag_input: RAGInput, performance_data: Dict[str, Any]):
        """Execute metrics level by level with parallelism within levels."""
        for level_idx, level_metrics in enumerate(self.execution_order):
            level_start = time.time()
            
            self.logger.debug(f"Executing level {level_idx + 1}/{len(self.execution_order)} with {len(level_metrics)} metrics")
            
            # Check dependencies for this level
            ready_metrics = []
            for metric_name in level_metrics:
                execution = executions[metric_name]
                if self._check_dependencies(metric_name, executions):
                    ready_metrics.append(metric_name)
                    execution.dependencies_met = True
                else:
                    execution.status = "skipped"
                    execution.error = Exception("Dependencies not met")
            
            if not ready_metrics:
                continue
            
            # Execute ready metrics in parallel
            tasks = []
            for metric_name in ready_metrics:
                task = self._execute_single_metric(executions[metric_name], rag_input)
                tasks.append(task)
            
            # Wait for level completion
            await asyncio.gather(*tasks, return_exceptions=True)
            
            level_time = time.time() - level_start
            performance_data["level_times"].append({
                "level": level_idx,
                "metrics": ready_metrics,
                "execution_time": level_time,
                "parallel_count": len(ready_metrics)
            })
    
    async def _execute_sequential(self, executions: Dict[str, MetricExecution], 
                                rag_input: RAGInput, performance_data: Dict[str, Any]):
        """Execute metrics sequentially in dependency order."""
        for level_metrics in self.execution_order:
            for metric_name in level_metrics:
                execution = executions[metric_name]
                if self._check_dependencies(metric_name, executions):
                    await self._execute_single_metric(execution, rag_input)
                else:
                    execution.status = "skipped"
                    execution.error = Exception("Dependencies not met")
    
    async def _execute_full_parallel(self, executions: Dict[str, MetricExecution], 
                                   rag_input: RAGInput, performance_data: Dict[str, Any]):
        """Execute all metrics in parallel, ignoring dependencies."""
        tasks = []
        for execution in executions.values():
            execution.dependencies_met = True  # Ignore dependencies
            task = self._execute_single_metric(execution, rag_input)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_single_metric(self, execution: MetricExecution, rag_input: RAGInput):
        """Execute a single metric with timeout and retry logic."""
        metric = execution.metric
        metric_name = metric.name()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                execution.status = "running"
                execution.start_time = time.time()
                execution.retry_count = attempt
                
                self.logger.debug(f"Executing metric {metric_name} (attempt {attempt + 1})")
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    metric.evaluate_with_cache(rag_input),
                    timeout=self.config.metric_timeout
                )
                
                execution.end_time = time.time()
                execution.result = result
                execution.status = "completed"
                
                self.logger.debug(f"Metric {metric_name} completed successfully in {execution.execution_time:.2f}s")
                return result
                
            except asyncio.TimeoutError:
                execution.error = Exception(f"Metric {metric_name} timed out after {self.config.metric_timeout}s")
                self.logger.warning(f"Metric {metric_name} timed out (attempt {attempt + 1})")
            except Exception as e:
                execution.error = e
                self.logger.warning(f"Metric {metric_name} failed: {str(e)} (attempt {attempt + 1})")
            
            # Don't retry if not configured
            if not self.config.retry_failed_metrics:
                break
        
        # All attempts failed
        execution.end_time = time.time()
        execution.status = "failed"
        
        # Handle degradation
        if self.config.degradation_mode == DegradationMode.GRACEFUL:
            # Return a degraded result
            execution.result = MetricResult(
                score=0.0,
                explanation=f"Metric {metric_name} failed after {execution.retry_count + 1} attempts: {str(execution.error)}",
                details={"degraded_mode": True, "error": str(execution.error), "attempts": execution.retry_count + 1}
            )
        
        return execution.result
    
    def _check_dependencies(self, metric_name: str, executions: Dict[str, MetricExecution]) -> bool:
        """Check if all dependencies for a metric are satisfied."""
        if metric_name not in self.dependencies:
            return True
        
        dependency = self.dependencies[metric_name]
        
        for dep_name in dependency.depends_on:
            if dep_name not in executions:
                if not dependency.optional:
                    return False
                continue
            
            dep_execution = executions[dep_name]
            if dep_execution.status not in ["completed"] and not dependency.optional:
                return False
        
        return True
    
    def _compile_results(self, executions: Dict[str, MetricExecution], 
                        performance_data: Dict[str, Any], total_time: float) -> PipelineResult:
        """Compile final pipeline results."""
        metric_results = {}
        failed_metrics = []
        skipped_metrics = []
        execution_stats = {}
        
        for metric_name, execution in executions.items():
            if execution.status == "completed" and execution.result:
                metric_results[metric_name] = execution.result
            elif execution.status == "failed":
                failed_metrics.append(metric_name)
                if execution.result:  # Degraded result
                    metric_results[metric_name] = execution.result
            elif execution.status == "skipped":
                skipped_metrics.append(metric_name)
            
            execution_stats[metric_name] = {
                "status": execution.status,
                "execution_time": execution.execution_time,
                "retry_count": execution.retry_count,
                "dependencies_met": execution.dependencies_met
            }
            
            if execution.execution_time:
                performance_data["metric_times"][metric_name] = execution.execution_time
        
        # Calculate parallel efficiency
        if performance_data["level_times"]:
            total_metric_time = sum(performance_data["metric_times"].values())
            performance_data["parallel_efficiency"] = total_metric_time / total_time if total_time > 0 else 0.0
        
        pipeline_success = len(failed_metrics) == 0 or self.config.degradation_mode == DegradationMode.GRACEFUL
        
        return PipelineResult(
            metric_results=metric_results,
            execution_stats=execution_stats,
            performance_metrics=performance_data,
            failed_metrics=failed_metrics,
            skipped_metrics=skipped_metrics,
            total_execution_time=total_time,
            pipeline_success=pipeline_success
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from execution history."""
        if not self.performance_history:
            return {"message": "No execution history available"}
        
        recent_executions = self.performance_history[-10:]  # Last 10 executions
        
        avg_time = sum(e["total_time"] for e in recent_executions) / len(recent_executions)
        avg_success_rate = sum(1 for e in recent_executions if e["failed_metrics"] == 0) / len(recent_executions)
        
        return {
            "total_executions": len(self.performance_history),
            "average_execution_time": avg_time,
            "average_success_rate": avg_success_rate,
            "recent_executions": recent_executions,
            "current_config": {
                "execution_strategy": self.config.execution_strategy.value,
                "degradation_mode": self.config.degradation_mode.value,
                "max_parallel_metrics": self.config.max_parallel_metrics,
                "metric_timeout": self.config.metric_timeout
            }
        }
    
    def _validate_metric_requirements(self, rag_input: RAGInput):
        """Validate that all metric requirements are met."""
        from .dependencies import get_dependency_manager
        
        dependency_manager = get_dependency_manager()
        validation_errors = []
        
        for metric in self.metrics:
            errors = dependency_manager.validate_metric_requirements(metric.name(), rag_input)
            validation_errors.extend(errors)
        
        if validation_errors and self.config.degradation_mode == DegradationMode.FAIL_FAST:
            error_msg = f"Metric requirement validation failed: {'; '.join(validation_errors)}"
            self.logger.error(error_msg)
            raise EvaluatorError(error_msg)
        elif validation_errors:
            self.logger.warning(f"Some metric requirements not met: {'; '.join(validation_errors)}")
    
    def get_performance_analysis(self, detailed: bool = False) -> Dict[str, Any]:
        """Get comprehensive performance analysis.
        
        Args:
            detailed: Whether to include detailed historical data
            
        Returns:
            Performance analysis results
        """
        if not self.config.enable_performance_monitoring:
            return {"error": "Performance monitoring is disabled"}
        
        performance_analyzer = get_performance_analyzer()
        return performance_analyzer.generate_performance_report(detailed=detailed)
    
    def get_metric_performance_analysis(self, metric_name: str) -> Dict[str, Any]:
        """Get performance analysis for a specific metric.
        
        Args:
            metric_name: Name of the metric to analyze
            
        Returns:
            Metric-specific performance analysis
        """
        if not self.config.enable_performance_monitoring:
            return {"error": "Performance monitoring is disabled"}
        
        performance_analyzer = get_performance_analyzer()
        return performance_analyzer.analyze_metric_performance(metric_name)
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on performance analysis.
        
        Returns:
            List of optimization suggestions
        """
        if not self.config.enable_performance_monitoring:
            return ["Enable performance monitoring to get optimization suggestions"]
        
        analysis = self.get_performance_analysis()
        return analysis.get("pipeline_summary", {}).get("optimization_suggestions", [])
    
    def clear_performance_history(self):
        """Clear performance history."""
        self.performance_history.clear()
        
        if self.config.enable_performance_monitoring:
            performance_analyzer = get_performance_analyzer()
            performance_analyzer.pipeline_history.clear()
            performance_analyzer.metric_history.clear()
        
        self.logger.info("Performance history cleared") 