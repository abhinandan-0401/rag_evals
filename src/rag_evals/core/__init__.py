"""Core components for RAG evaluation."""

from .types import RAGInput, MetricResult, EvaluationResult
from .base_metric import BaseMetric
from .logging import get_logger, MetricLogger, EvaluationContext, configure_logging
from .exceptions import (
    RAGEvalError, 
    LLMError, LLMConnectionError, LLMConfigurationError, LLMRateLimitError, LLMResponseError,
    MetricError, MetricConfigurationError, MetricInputError, MetricEvaluationError,
    EvaluatorError, EvaluatorConfigurationError, BatchEvaluationError,
    PromptError, DependencyError, ValidationError,
    handle_llm_error, handle_metric_error
)
from .error_handler import (
    ErrorSeverity, ErrorCategory, ErrorContext, RecoveryStrategy,
    ErrorClassifier, ErrorRecoveryManager, get_recovery_manager,
    with_error_recovery, error_context, sync_error_context
)
from .pipeline import (
    EvaluationPipeline, PipelineConfig, PipelineResult,
    ExecutionStrategy, DegradationMode, MetricDependency, MetricExecution
)
from .dependencies import (
    MetricDependencyManager, MetricProfile, MetricRequirement, MetricDependencyRule,
    DependencyType, RequirementType, get_dependency_manager
)
from .performance import (
    PerformanceAnalyzer, MetricPerformanceData, PipelinePerformanceData,
    PerformanceTrend, SystemResourceMonitor, PerformanceMetricType, PerformanceLevel,
    get_performance_analyzer
)
 
__all__ = [
    # Core types
    "RAGInput", 
    "MetricResult", 
    "EvaluationResult", 
    "BaseMetric",
    
    # Logging
    "get_logger",
    "MetricLogger", 
    "EvaluationContext",
    "configure_logging",
    
    # Exception handling - Base classes
    "RAGEvalError",
    
    # LLM exceptions
    "LLMError",
    "LLMConnectionError", 
    "LLMConfigurationError",
    "LLMRateLimitError",
    "LLMResponseError",
    
    # Metric exceptions
    "MetricError",
    "MetricConfigurationError",
    "MetricInputError",
    "MetricEvaluationError",
    
    # Evaluator exceptions
    "EvaluatorError",
    "EvaluatorConfigurationError",
    "BatchEvaluationError",
    
    # Other exceptions
    "PromptError",
    "DependencyError",
    "ValidationError",
    
    # Error handlers
    "handle_llm_error",
    "handle_metric_error",
    
    # Advanced error handling
    "ErrorSeverity",
    "ErrorCategory", 
    "ErrorContext",
    "RecoveryStrategy",
    "ErrorClassifier",
    "ErrorRecoveryManager",
    "get_recovery_manager",
    "with_error_recovery",
    "error_context",
    "sync_error_context",
    
    # Pipeline components
    "EvaluationPipeline",
    "PipelineConfig", 
    "PipelineResult",
    "ExecutionStrategy",
    "DegradationMode",
    "MetricDependency",
    "MetricExecution",
    
    # Dependency management
    "MetricDependencyManager",
    "MetricProfile",
    "MetricRequirement", 
    "MetricDependencyRule",
    "DependencyType",
    "RequirementType",
    "get_dependency_manager",
    
    # Performance monitoring
    "PerformanceAnalyzer",
    "MetricPerformanceData",
    "PipelinePerformanceData",
    "PerformanceTrend",
    "SystemResourceMonitor",
    "PerformanceMetricType",
    "PerformanceLevel",
    "get_performance_analyzer"
] 