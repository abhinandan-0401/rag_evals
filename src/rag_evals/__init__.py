"""RAG Evaluation Library with LangChain Integration.

A comprehensive library for evaluating Retrieval-Augmented Generation (RAG) systems
using LangChain's prompt management and LLM interfaces. Includes enterprise-grade
logging, error handling, and Azure OpenAI integration.
"""

from .evaluator import RAGEvaluator
from .core.types import RAGInput, MetricResult, EvaluationResult
from .core.logging import get_logger, configure_logging, EvaluationContext
from .core.exceptions import (
    RAGEvalError, LLMError, MetricError, EvaluatorError,
    LLMConfigurationError, MetricConfigurationError, EvaluatorConfigurationError
)
from .core.error_handler import (
    ErrorSeverity, ErrorCategory, ErrorContext, RecoveryStrategy,
    ErrorClassifier, ErrorRecoveryManager, get_recovery_manager,
    with_error_recovery, error_context, sync_error_context
)
from .core.pipeline import (
    EvaluationPipeline, PipelineConfig, PipelineResult,
    ExecutionStrategy, DegradationMode, MetricDependency, MetricExecution
)
from .core.dependencies import (
    MetricDependencyManager, MetricProfile, MetricRequirement, MetricDependencyRule,
    DependencyType, RequirementType, get_dependency_manager
)
from .core.performance import (
    PerformanceAnalyzer, MetricPerformanceData, PipelinePerformanceData,
    PerformanceTrend, SystemResourceMonitor, PerformanceMetricType, PerformanceLevel,
    get_performance_analyzer
)
from .llm import create_llm, ChatOpenAI, AzureChatOpenAI, LLMManager, get_llm_manager, get_managed_llm

# Individual metrics (for advanced users)
from .metrics import (
    # Generation metrics
    Faithfulness, AnswerRelevance, AnswerCorrectness, Completeness, Coherence, Helpfulness,
    # Retrieval metrics  
    ContextRecall, ContextRelevance, ContextPrecision,
    # Composite metrics
    LLMJudge, RAGCertainty, TrustScore
)

__version__ = "0.0.1"
__all__ = [
    # Main interface
    "RAGEvaluator",
    
    # Core types
    "RAGInput",
    "MetricResult", 
    "EvaluationResult",
    
    # LLM utilities
    "create_llm",
    "ChatOpenAI", 
    "AzureChatOpenAI",
    "LLMManager",
    "get_llm_manager",
    "get_managed_llm",
    
    # Logging and monitoring
    "get_logger",
    "configure_logging", 
    "EvaluationContext",
    
    # Exception handling
    "RAGEvalError",
    "LLMError",
    "MetricError", 
    "EvaluatorError",
    "LLMConfigurationError",
    "MetricConfigurationError",
    "EvaluatorConfigurationError",
    
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
    "get_performance_analyzer",
    
    # Generation metrics
    "Faithfulness",
    "AnswerRelevance", 
    "AnswerCorrectness",
    "Completeness",
    "Coherence",
    "Helpfulness",
    
    # Retrieval metrics
    "ContextRecall",
    "ContextRelevance",
    "ContextPrecision",
    
    # Composite metrics
    "LLMJudge",
    "RAGCertainty",
    "TrustScore",
    
    # Version
    "__version__"
]

# Configure default logging when package is imported
configure_logging()

# Package metadata
__author__ = "RAG Evals Team"
__email__ = "contact@ragevals.dev"
__description__ = "Comprehensive RAG evaluation library with Azure OpenAI integration"
__url__ = "https://github.com/ragevals/rag_evals" 