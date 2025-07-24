"""Standardized exception classes for RAG evaluation system.

This module defines a hierarchy of exceptions with consistent error handling,
recovery strategies, and detailed error information for debugging.
"""

from typing import Dict, Any, Optional, List
import traceback


class RAGEvalError(Exception):
    """Base exception class for all RAG evaluation errors.
    
    Provides consistent error handling with detailed context information,
    recovery suggestions, and proper error categorization.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize RAG evaluation error.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for programmatic handling
            context: Additional context information
            recovery_suggestions: List of suggested recovery actions
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.original_error = original_error
        self.stack_trace = traceback.format_exc() if original_error else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "error_message": self.message,  # Changed from "message" to avoid logging conflict
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions,
            "original_error": str(self.original_error) if self.original_error else None,
            "stack_trace": self.stack_trace
        }
    
    def __str__(self) -> str:
        """String representation with context."""
        base_msg = f"{self.error_code}: {self.message}"
        
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            base_msg += f" (Context: {context_str})"
        
        if self.recovery_suggestions:
            suggestions = "; ".join(self.recovery_suggestions)
            base_msg += f" [Suggestions: {suggestions}]"
        
        return base_msg


class LLMError(RAGEvalError):
    """Errors related to LLM operations."""
    
    def __init__(self, message: str, llm_provider: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.llm_provider = llm_provider
        if llm_provider:
            self.context["llm_provider"] = llm_provider


class LLMConnectionError(LLMError):
    """LLM connection or network-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="LLM_CONNECTION_ERROR",
            recovery_suggestions=[
                "Check your internet connection",
                "Verify API credentials are correct",
                "Check if the LLM service is available",
                "Try again with exponential backoff"
            ],
            **kwargs
        )


class LLMConfigurationError(LLMError):
    """LLM configuration and setup errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="LLM_CONFIG_ERROR",
            recovery_suggestions=[
                "Check required environment variables are set",
                "Verify Azure/OpenAI configuration parameters",
                "Ensure API keys have proper permissions",
                "Check deployment/model names are correct"
            ],
            **kwargs
        )


class LLMRateLimitError(LLMError):
    """LLM rate limiting errors."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        self.retry_after = retry_after
        super().__init__(
            message,
            error_code="LLM_RATE_LIMIT",
            recovery_suggestions=[
                f"Wait {retry_after} seconds before retrying" if retry_after else "Wait before retrying",
                "Reduce batch size or concurrent requests",
                "Implement exponential backoff",
                "Check rate limit quotas"
            ],
            **kwargs
        )
        if retry_after:
            self.context["retry_after"] = retry_after


class LLMResponseError(LLMError):
    """LLM response parsing or validation errors."""
    
    def __init__(self, message: str, response_content: str = None, **kwargs):
        super().__init__(
            message,
            error_code="LLM_RESPONSE_ERROR",
            recovery_suggestions=[
                "Check prompt formatting and instructions",
                "Verify response parsing logic",
                "Try with different temperature settings",
                "Add more specific output format instructions"
            ],
            **kwargs
        )
        if response_content:
            self.context["response_content"] = response_content[:500]  # Truncate for logging


class MetricError(RAGEvalError):
    """Base class for metric evaluation errors."""
    
    def __init__(self, message: str, metric_name: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.metric_name = metric_name
        if metric_name:
            self.context["metric_name"] = metric_name


class MetricConfigurationError(MetricError):
    """Metric configuration and initialization errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="METRIC_CONFIG_ERROR",
            recovery_suggestions=[
                "Check metric initialization parameters",
                "Verify required dependencies are installed",
                "Ensure LLM provider is properly configured",
                "Check metric-specific requirements"
            ],
            **kwargs
        )


class MetricInputError(MetricError):
    """Invalid input data for metric evaluation."""
    
    def __init__(self, message: str, input_type: str = None, **kwargs):
        super().__init__(
            message,
            error_code="METRIC_INPUT_ERROR",
            recovery_suggestions=[
                "Check required input fields are provided",
                "Verify input data format and types",
                "Ensure contexts are provided for retrieval metrics",
                "Check answer is provided for generation metrics"
            ],
            **kwargs
        )
        if input_type:
            self.context["input_type"] = input_type


class MetricEvaluationError(MetricError):
    """Errors during metric evaluation process."""
    
    def __init__(self, message: str, evaluation_stage: str = None, **kwargs):
        super().__init__(
            message,
            error_code="METRIC_EVAL_ERROR",
            recovery_suggestions=[
                "Retry evaluation with different parameters",
                "Check input data quality",
                "Verify LLM is responding correctly",
                "Try with simpler evaluation approach"
            ],
            **kwargs
        )
        if evaluation_stage:
            self.context["evaluation_stage"] = evaluation_stage


class EvaluatorError(RAGEvalError):
    """Errors in the main RAGEvaluator class."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class EvaluatorConfigurationError(EvaluatorError):
    """RAGEvaluator configuration errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="EVALUATOR_CONFIG_ERROR",
            recovery_suggestions=[
                "Check metric names are valid",
                "Verify LLM configuration",
                "Ensure required parameters are provided",
                "Check available metrics list"
            ],
            **kwargs
        )


class BatchEvaluationError(EvaluatorError):
    """Errors during batch evaluation."""
    
    def __init__(self, message: str, batch_size: int = None, failed_count: int = None, **kwargs):
        super().__init__(
            message,
            error_code="BATCH_EVAL_ERROR",
            recovery_suggestions=[
                "Reduce batch size",
                "Check input data format",
                "Implement error recovery for individual items",
                "Review failed items separately"
            ],
            **kwargs
        )
        if batch_size:
            self.context["batch_size"] = batch_size
        if failed_count:
            self.context["failed_count"] = failed_count


class PromptError(RAGEvalError):
    """Errors related to prompt management and templates."""
    
    def __init__(self, message: str, prompt_name: str = None, **kwargs):
        super().__init__(
            message,
            error_code="PROMPT_ERROR",
            recovery_suggestions=[
                "Check prompt template exists",
                "Verify variable names in template",
                "Ensure proper prompt formatting",
                "Check YAML syntax for prompt files"
            ],
            **kwargs
        )
        if prompt_name:
            self.context["prompt_name"] = prompt_name


class DependencyError(RAGEvalError):
    """Missing or incompatible dependency errors."""
    
    def __init__(self, message: str, dependency_name: str = None, **kwargs):
        super().__init__(
            message,
            error_code="DEPENDENCY_ERROR",
            recovery_suggestions=[
                "Install missing dependencies with pip",
                "Check version compatibility",
                "Update to latest supported versions",
                "Check optional dependencies for features"
            ],
            **kwargs
        )
        if dependency_name:
            self.context["dependency_name"] = dependency_name


class ValidationError(RAGEvalError):
    """Data validation errors."""
    
    def __init__(self, message: str, field_name: str = None, expected_type: str = None, **kwargs):
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            recovery_suggestions=[
                "Check data types match expected format",
                "Verify required fields are provided",
                "Ensure data is not empty or None",
                "Check data format specifications"
            ],
            **kwargs
        )
        if field_name:
            self.context["field_name"] = field_name
        if expected_type:
            self.context["expected_type"] = expected_type


def handle_llm_error(original_error: Exception, llm_provider: str = None) -> LLMError:
    """Convert generic exceptions to appropriate LLM error types.
    
    Args:
        original_error: Original exception from LLM operation
        llm_provider: Name of LLM provider
        
    Returns:
        Appropriate LLMError subclass
    """
    error_message = str(original_error)
    
    # Connection-related errors
    if any(keyword in error_message.lower() for keyword in [
        "connection", "network", "timeout", "unreachable", "refused"
    ]):
        return LLMConnectionError(
            f"Failed to connect to LLM service: {error_message}",
            llm_provider=llm_provider,
            original_error=original_error
        )
    
    # Rate limiting
    if any(keyword in error_message.lower() for keyword in [
        "rate limit", "quota", "too many requests", "429"
    ]):
        return LLMRateLimitError(
            f"Rate limit exceeded: {error_message}",
            llm_provider=llm_provider,
            original_error=original_error
        )
    
    # Configuration errors
    if any(keyword in error_message.lower() for keyword in [
        "authentication", "api key", "unauthorized", "forbidden", "401", "403"
    ]):
        return LLMConfigurationError(
            f"LLM authentication/configuration error: {error_message}",
            llm_provider=llm_provider,
            original_error=original_error
        )
    
    # Generic LLM error
    return LLMError(
        f"LLM operation failed: {error_message}",
        llm_provider=llm_provider,
        original_error=original_error
    )


def handle_metric_error(
    original_error: Exception, 
    metric_name: str = None,
    evaluation_stage: str = None
) -> MetricError:
    """Convert generic exceptions to appropriate metric error types.
    
    Args:
        original_error: Original exception from metric evaluation
        metric_name: Name of the metric
        evaluation_stage: Stage where error occurred
        
    Returns:
        Appropriate MetricError subclass
    """
    error_message = str(original_error)
    
    # Input validation errors
    if any(keyword in error_message.lower() for keyword in [
        "missing", "required", "empty", "invalid", "none"
    ]):
        return MetricInputError(
            f"Invalid input for metric evaluation: {error_message}",
            metric_name=metric_name,
            original_error=original_error
        )
    
    # Configuration errors
    if any(keyword in error_message.lower() for keyword in [
        "not configured", "missing llm", "no provider"
    ]):
        return MetricConfigurationError(
            f"Metric configuration error: {error_message}",
            metric_name=metric_name,
            original_error=original_error
        )
    
    # Generic evaluation error
    return MetricEvaluationError(
        f"Metric evaluation failed: {error_message}",
        metric_name=metric_name,
        evaluation_stage=evaluation_stage,
        original_error=original_error
    )


__all__ = [
    # Base classes
    "RAGEvalError",
    
    # LLM errors
    "LLMError",
    "LLMConnectionError", 
    "LLMConfigurationError",
    "LLMRateLimitError",
    "LLMResponseError",
    
    # Metric errors
    "MetricError",
    "MetricConfigurationError",
    "MetricInputError",
    "MetricEvaluationError",
    
    # Evaluator errors
    "EvaluatorError",
    "EvaluatorConfigurationError",
    "BatchEvaluationError",
    
    # Other errors
    "PromptError",
    "DependencyError",
    "ValidationError",
    
    # Error handlers
    "handle_llm_error",
    "handle_metric_error"
] 