"""Advanced performance monitoring and analytics for RAG evaluation pipelines."""

import time
import psutil
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict, deque
import statistics
import json

from .logging import get_logger
from .types import MetricResult


class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"
    PARALLEL_EFFICIENCY = "parallel_efficiency"
    CACHE_HIT_RATE = "cache_hit_rate"


class PerformanceLevel(Enum):
    """Performance level classification."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class MetricPerformanceData:
    """Performance data for a single metric execution."""
    metric_name: str
    execution_time: float
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    cache_hit: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "execution_time": self.execution_time,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "cache_hit": self.cache_hit,
            "retry_count": self.retry_count
        }


@dataclass
class PipelinePerformanceData:
    """Performance data for a complete pipeline execution."""
    pipeline_id: str
    total_execution_time: float
    metrics_count: int
    successful_metrics: int
    failed_metrics: int
    parallel_efficiency: float
    memory_peak_mb: float = 0.0
    cpu_average_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)
    execution_strategy: str = "unknown"
    metric_performances: List[MetricPerformanceData] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.metrics_count == 0:
            return 0.0
        return self.successful_metrics / self.metrics_count
    
    @property
    def throughput(self) -> float:
        """Calculate throughput (metrics per second)."""
        if self.total_execution_time == 0:
            return 0.0
        return self.metrics_count / self.total_execution_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pipeline_id": self.pipeline_id,
            "total_execution_time": self.total_execution_time,
            "metrics_count": self.metrics_count,
            "successful_metrics": self.successful_metrics,
            "failed_metrics": self.failed_metrics,
            "success_rate": self.success_rate,
            "throughput": self.throughput,
            "parallel_efficiency": self.parallel_efficiency,
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_average_percent": self.cpu_average_percent,
            "timestamp": self.timestamp,
            "execution_strategy": self.execution_strategy,
            "metric_performances": [mp.to_dict() for mp in self.metric_performances]
        }


@dataclass
class PerformanceTrend:
    """Performance trend analysis."""
    metric_type: PerformanceMetricType
    trend_direction: str  # "improving", "degrading", "stable"
    change_percentage: float
    confidence: float  # 0.0 to 1.0
    sample_size: int
    
    def __str__(self) -> str:
        direction_emoji = {"improving": "📈", "degrading": "📉", "stable": "📊"}
        emoji = direction_emoji.get(self.trend_direction, "📊")
        return f"{emoji} {self.metric_type.value}: {self.trend_direction} ({self.change_percentage:+.1f}%)"


class SystemResourceMonitor:
    """Monitors system resources during execution."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.samples: List[Tuple[float, float, float]] = []  # (timestamp, memory_mb, cpu_percent)
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.samples.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Tuple[float, float]:
        """Stop monitoring and return peak memory and average CPU."""
        if not self.monitoring:
            return 0.0, 0.0
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.samples:
            return 0.0, 0.0
        
        memory_values = [sample[1] for sample in self.samples]
        cpu_values = [sample[2] for sample in self.samples]
        
        peak_memory = max(memory_values) if memory_values else 0.0
        avg_cpu = statistics.mean(cpu_values) if cpu_values else 0.0
        
        return peak_memory, avg_cpu
    
    def _monitor_loop(self):
        """Resource monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                timestamp = time.time()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                self.samples.append((timestamp, memory_mb, cpu_percent))
                
                time.sleep(self.sampling_interval)
            except Exception:
                # Ignore monitoring errors
                pass


class PerformanceAnalyzer:
    """Analyzes performance data and provides insights."""
    
    def __init__(self, history_limit: int = 1000):
        self.logger = get_logger("performance")
        self.history_limit = history_limit
        self.pipeline_history: deque = deque(maxlen=history_limit)
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_limit))
        
    def add_pipeline_performance(self, performance_data: PipelinePerformanceData):
        """Add pipeline performance data to history."""
        self.pipeline_history.append(performance_data)
        
        # Also add individual metric performances
        for metric_perf in performance_data.metric_performances:
            self.metric_history[metric_perf.metric_name].append(metric_perf)
    
    def analyze_metric_performance(self, metric_name: str, 
                                 recent_count: int = 10) -> Dict[str, Any]:
        """Analyze performance for a specific metric."""
        if metric_name not in self.metric_history:
            return {"error": f"No performance data for metric: {metric_name}"}
        
        history = list(self.metric_history[metric_name])
        if not history:
            return {"error": f"No performance data for metric: {metric_name}"}
        
        recent_data = history[-recent_count:] if len(history) >= recent_count else history
        
        # Calculate statistics
        execution_times = [d.execution_time for d in recent_data if d.success]
        success_rate = sum(1 for d in recent_data if d.success) / len(recent_data)
        cache_hit_rate = sum(1 for d in recent_data if d.cache_hit) / len(recent_data)
        
        analysis = {
            "metric_name": metric_name,
            "sample_size": len(recent_data),
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "performance_level": self._classify_performance_level(execution_times, success_rate)
        }
        
        if execution_times:
            analysis.update({
                "avg_execution_time": statistics.mean(execution_times),
                "median_execution_time": statistics.median(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times),
                "std_execution_time": statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
            })
        
        # Detect trends
        trends = self._detect_trends(metric_name, recent_count)
        analysis["trends"] = trends
        
        return analysis
    
    def analyze_pipeline_performance(self, recent_count: int = 10) -> Dict[str, Any]:
        """Analyze overall pipeline performance."""
        if not self.pipeline_history:
            return {"error": "No pipeline performance data available"}
        
        recent_data = list(self.pipeline_history)[-recent_count:]
        
        # Calculate statistics
        execution_times = [d.total_execution_time for d in recent_data]
        success_rates = [d.success_rate for d in recent_data]
        throughputs = [d.throughput for d in recent_data]
        parallel_efficiencies = [d.parallel_efficiency for d in recent_data]
        
        analysis = {
            "sample_size": len(recent_data),
            "avg_execution_time": statistics.mean(execution_times),
            "avg_success_rate": statistics.mean(success_rates),
            "avg_throughput": statistics.mean(throughputs),
            "avg_parallel_efficiency": statistics.mean(parallel_efficiencies),
            "performance_level": self._classify_pipeline_performance_level(recent_data)
        }
        
        if len(execution_times) > 1:
            analysis.update({
                "execution_time_std": statistics.stdev(execution_times),
                "success_rate_std": statistics.stdev(success_rates),
                "throughput_std": statistics.stdev(throughputs)
            })
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(recent_data)
        analysis["bottlenecks"] = bottlenecks
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(analysis, recent_data)
        analysis["optimization_suggestions"] = suggestions
        
        return analysis
    
    def _detect_trends(self, metric_name: str, window_size: int = 10) -> List[PerformanceTrend]:
        """Detect performance trends for a metric."""
        trends = []
        
        if metric_name not in self.metric_history:
            return trends
        
        history = list(self.metric_history[metric_name])
        if len(history) < window_size:
            return trends
        
        recent_data = history[-window_size:]
        older_data = history[-window_size*2:-window_size] if len(history) >= window_size*2 else []
        
        if not older_data:
            return trends
        
        # Analyze execution time trend
        recent_times = [d.execution_time for d in recent_data if d.success]
        older_times = [d.execution_time for d in older_data if d.success]
        
        if recent_times and older_times:
            recent_avg = statistics.mean(recent_times)
            older_avg = statistics.mean(older_times)
            change_pct = ((recent_avg - older_avg) / older_avg) * 100
            
            if abs(change_pct) > 5:  # Significant change threshold
                direction = "degrading" if change_pct > 0 else "improving"
                confidence = min(1.0, abs(change_pct) / 20.0)  # Higher change = higher confidence
                
                trends.append(PerformanceTrend(
                    metric_type=PerformanceMetricType.EXECUTION_TIME,
                    trend_direction=direction,
                    change_percentage=change_pct,
                    confidence=confidence,
                    sample_size=len(recent_times)
                ))
        
        # Analyze success rate trend
        recent_success_rate = sum(1 for d in recent_data if d.success) / len(recent_data)
        older_success_rate = sum(1 for d in older_data if d.success) / len(older_data)
        
        success_change_pct = ((recent_success_rate - older_success_rate) / older_success_rate) * 100
        
        if abs(success_change_pct) > 10:  # Significant change threshold for success rate
            direction = "improving" if success_change_pct > 0 else "degrading"
            confidence = min(1.0, abs(success_change_pct) / 30.0)
            
            trends.append(PerformanceTrend(
                metric_type=PerformanceMetricType.SUCCESS_RATE,
                trend_direction=direction,
                change_percentage=success_change_pct,
                confidence=confidence,
                sample_size=len(recent_data)
            ))
        
        return trends
    
    def _classify_performance_level(self, execution_times: List[float], 
                                  success_rate: float) -> PerformanceLevel:
        """Classify performance level for a metric."""
        if not execution_times or success_rate < 0.8:
            return PerformanceLevel.POOR
        
        avg_time = statistics.mean(execution_times)
        
        if success_rate >= 0.95 and avg_time <= 2.0:
            return PerformanceLevel.EXCELLENT
        elif success_rate >= 0.9 and avg_time <= 5.0:
            return PerformanceLevel.GOOD
        elif success_rate >= 0.8 and avg_time <= 10.0:
            return PerformanceLevel.FAIR
        else:
            return PerformanceLevel.POOR
    
    def _classify_pipeline_performance_level(self, 
                                           pipeline_data: List[PipelinePerformanceData]) -> PerformanceLevel:
        """Classify overall pipeline performance level."""
        if not pipeline_data:
            return PerformanceLevel.POOR
        
        avg_success_rate = statistics.mean([d.success_rate for d in pipeline_data])
        avg_parallel_efficiency = statistics.mean([d.parallel_efficiency for d in pipeline_data])
        avg_throughput = statistics.mean([d.throughput for d in pipeline_data])
        
        if avg_success_rate >= 0.95 and avg_parallel_efficiency >= 0.8 and avg_throughput >= 2.0:
            return PerformanceLevel.EXCELLENT
        elif avg_success_rate >= 0.9 and avg_parallel_efficiency >= 0.6 and avg_throughput >= 1.0:
            return PerformanceLevel.GOOD
        elif avg_success_rate >= 0.8 and avg_parallel_efficiency >= 0.4:
            return PerformanceLevel.FAIR
        else:
            return PerformanceLevel.POOR
    
    def _detect_bottlenecks(self, pipeline_data: List[PipelinePerformanceData]) -> List[str]:
        """Detect performance bottlenecks."""
        bottlenecks = []
        
        if not pipeline_data:
            return bottlenecks
        
        # Analyze parallel efficiency
        avg_parallel_efficiency = statistics.mean([d.parallel_efficiency for d in pipeline_data])
        if avg_parallel_efficiency < 0.5:
            bottlenecks.append("Low parallel efficiency - consider optimizing metric dependencies")
        
        # Analyze individual metric performance
        metric_times = defaultdict(list)
        for pipeline in pipeline_data:
            for metric_perf in pipeline.metric_performances:
                if metric_perf.success:
                    metric_times[metric_perf.metric_name].append(metric_perf.execution_time)
        
        # Find slow metrics
        for metric_name, times in metric_times.items():
            if times:
                avg_time = statistics.mean(times)
                if avg_time > 10.0:  # Threshold for slow metrics
                    bottlenecks.append(f"Slow metric: {metric_name} (avg: {avg_time:.1f}s)")
        
        # Analyze memory usage
        memory_peaks = [d.memory_peak_mb for d in pipeline_data if d.memory_peak_mb > 0]
        if memory_peaks:
            avg_memory = statistics.mean(memory_peaks)
            if avg_memory > 1000:  # 1GB threshold
                bottlenecks.append(f"High memory usage: {avg_memory:.0f}MB average")
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self, analysis: Dict[str, Any], 
                                         pipeline_data: List[PipelinePerformanceData]) -> List[str]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []
        
        # Parallel efficiency suggestions
        parallel_eff = analysis.get("avg_parallel_efficiency", 0)
        if parallel_eff < 0.5:
            suggestions.append("Consider reducing metric dependencies to improve parallelization")
        elif parallel_eff < 0.7:
            suggestions.append("Review metric execution order for better parallel efficiency")
        
        # Success rate suggestions
        success_rate = analysis.get("avg_success_rate", 0)
        if success_rate < 0.9:
            suggestions.append("Investigate failing metrics and implement better error handling")
        
        # Execution time suggestions
        exec_time = analysis.get("avg_execution_time", 0)
        if exec_time > 30:
            suggestions.append("Consider enabling metric caching to reduce execution time")
        elif exec_time > 60:
            suggestions.append("Consider using faster execution strategy or reducing metric count")
        
        # Memory suggestions
        if analysis.get("bottlenecks"):
            memory_bottlenecks = [b for b in analysis["bottlenecks"] if "memory" in b.lower()]
            if memory_bottlenecks:
                suggestions.append("Consider processing metrics in smaller batches to reduce memory usage")
        
        return suggestions
    
    def generate_performance_report(self, detailed: bool = False) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "pipeline_summary": self.analyze_pipeline_performance(),
            "metric_summaries": {},
            "overall_health": "unknown"
        }
        
        # Analyze each metric
        for metric_name in self.metric_history.keys():
            metric_analysis = self.analyze_metric_performance(metric_name)
            report["metric_summaries"][metric_name] = metric_analysis
        
        # Determine overall health
        pipeline_level = report["pipeline_summary"].get("performance_level", PerformanceLevel.POOR)
        if isinstance(pipeline_level, PerformanceLevel):
            report["overall_health"] = pipeline_level.value
        
        # Add detailed data if requested
        if detailed:
            report["detailed_history"] = {
                "pipeline_history": [p.to_dict() for p in list(self.pipeline_history)[-10:]],
                "metric_history": {
                    name: [m.to_dict() for m in list(history)[-10:]]
                    for name, history in self.metric_history.items()
                }
            }
        
        return report


# Global instance
_performance_analyzer = None

def get_performance_analyzer() -> PerformanceAnalyzer:
    """Get the global performance analyzer instance."""
    global _performance_analyzer
    if _performance_analyzer is None:
        _performance_analyzer = PerformanceAnalyzer()
    return _performance_analyzer 