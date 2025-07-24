"""Advanced error handling framework for RAG evaluation system.

This module provides:
- Consistent error recovery strategies
- Enhanced error context and reporting
- Graceful degradation capabilities
- Error categorization and routing
- Performance impact tracking
"""

import time
import traceback
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Union, Type
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
import asyncio
import logging

from .exceptions import RAGEvalError, LLMError, MetricError, EvaluatorError
from .logging import get_logger


class ErrorSeverity(Enum):
    """Error severity levels for handling strategies."""
    LOW = "low"           # Continue with degraded functionality
    MEDIUM = "medium"     # Retry with backoff, then degrade
    HIGH = "high"         # Immediate retry, then fail
    CRITICAL = "critical" # Immediate failure, no retry


class ErrorCategory(Enum):
    """Error categories for specialized handling."""
    NETWORK = "network"           # Connection, timeout issues
    AUTHENTICATION = "auth"       # API key, permission issues
    RATE_LIMIT = "rate_limit"     # Rate limiting, quota exceeded
    VALIDATION = "validation"     # Input validation, format issues
    CONFIGURATION = "config"      # Setup, configuration issues
    RESOURCE = "resource"         # Memory, disk, CPU issues
    DEPENDENCY = "dependency"     # Missing dependencies, import issues
    UNKNOWN = "unknown"          # Unclassified errors


@dataclass
class ErrorContext:
    """Rich error context for better debugging and recovery."""
    error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    component: str  # Which component failed
    operation: str  # What operation failed
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "category": self.category.value,
            "severity": self.severity.value,
            "component": self.component,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata
        }


@dataclass
class RecoveryStrategy:
    """Defines how to handle different types of errors."""
    retry_enabled: bool = True
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    base_delay: float = 1.0
    fallback_enabled: bool = True
    fallback_value: Any = None
    degraded_mode: bool = False  # Continue with reduced functionality
    
    def get_delay(self, retry_count: int) -> float:
        """Calculate delay for retry attempt."""
        return self.base_delay * (self.backoff_multiplier ** retry_count)


class ErrorClassifier:
    """Classifies errors into categories and assigns severity levels."""
    
    def __init__(self):
        self.logger = get_logger("error_classifier")
        
        # Error patterns for classification
        self.patterns = {
            ErrorCategory.NETWORK: [
                "connection", "timeout", "network", "unreachable", "refused"
            ],
            ErrorCategory.AUTHENTICATION: [
                "authentication", "unauthorized", "forbidden", "api key", "401", "403"
            ],
            ErrorCategory.RATE_LIMIT: [
                "rate limit", "quota", "too many requests", "429", "throttle"
            ],
            ErrorCategory.VALIDATION: [
                "validation", "invalid", "format", "parse", "schema"
            ],
            ErrorCategory.CONFIGURATION: [
                "configuration", "config", "missing", "not found", "setup"
            ],
            ErrorCategory.RESOURCE: [
                "memory", "disk", "cpu", "resource", "limit exceeded"
            ],
            ErrorCategory.DEPENDENCY: [
                "import", "module", "dependency", "not installed"
            ]
        }
        
        # Severity mapping by error type
        self.severity_map = {
            "LLMConnectionError": ErrorSeverity.MEDIUM,
            "LLMRateLimitError": ErrorSeverity.LOW,
            "LLMConfigurationError": ErrorSeverity.HIGH,
            "MetricConfigurationError": ErrorSeverity.HIGH,
            "MetricInputError": ErrorSeverity.MEDIUM,
            "EvaluatorConfigurationError": ErrorSeverity.CRITICAL,
            "ValidationError": ErrorSeverity.MEDIUM,
            "DependencyError": ErrorSeverity.CRITICAL
        }
    
    def classify(self, error: Exception, component: str = "unknown") -> ErrorContext:
        """Classify an error and create context."""
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Determine category
        category = ErrorCategory.UNKNOWN
        for cat, patterns in self.patterns.items():
            if any(pattern in error_message for pattern in patterns):
                category = cat
                break
        
        # Determine severity
        severity = self.severity_map.get(error_type, ErrorSeverity.MEDIUM)
        
        # Create context
        context = ErrorContext(
            error=error,
            category=category,
            severity=severity,
            component=component,
            operation="unknown"
        )
        
        self.logger.debug(
            f"Classified error: {error_type} -> {category.value}/{severity.value}",
            extra=context.to_dict()
        )
        
        return context


class ErrorRecoveryManager:
    """Manages error recovery strategies and execution."""
    
    def __init__(self):
        self.logger = get_logger("error_recovery")
        self.classifier = ErrorClassifier()
        
        # Default recovery strategies by category
        self.strategies = {
            ErrorCategory.NETWORK: RecoveryStrategy(
                retry_enabled=True,
                max_retries=3,
                backoff_multiplier=2.0,
                base_delay=1.0,
                fallback_enabled=True,
                degraded_mode=True
            ),
            ErrorCategory.RATE_LIMIT: RecoveryStrategy(
                retry_enabled=True,
                max_retries=5,
                backoff_multiplier=3.0,
                base_delay=5.0,
                fallback_enabled=True,
                degraded_mode=True
            ),
            ErrorCategory.AUTHENTICATION: RecoveryStrategy(
                retry_enabled=False,
                fallback_enabled=False,
                degraded_mode=False
            ),
            ErrorCategory.VALIDATION: RecoveryStrategy(
                retry_enabled=False,
                fallback_enabled=True,
                fallback_value=0.0,
                degraded_mode=True
            ),
            ErrorCategory.CONFIGURATION: RecoveryStrategy(
                retry_enabled=False,
                fallback_enabled=False,
                degraded_mode=False
            ),
            ErrorCategory.UNKNOWN: RecoveryStrategy(
                retry_enabled=True,
                max_retries=2,
                fallback_enabled=True,
                degraded_mode=True
            )
        }
    
    async def handle_async(
        self,
        operation: Callable,
        component: str,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Handle async operation with error recovery."""
        context = None
        
        for attempt in range(3):  # Max attempts
            try:
                result = await operation(*args, **kwargs)
                
                if context:  # Log successful recovery
                    self.logger.info(
                        f"Operation recovered successfully: {component}.{operation_name}",
                        extra={
                            "component": component,
                            "operation": operation_name,
                            "attempt": attempt + 1,
                            "recovery": True
                        }
                    )
                
                return result
                
            except Exception as error:
                context = self.classifier.classify(error, component)
                context.operation = operation_name
                context.retry_count = attempt
                
                strategy = self.strategies.get(context.category, self.strategies[ErrorCategory.UNKNOWN])
                
                self.logger.warning(
                    f"Operation failed: {component}.{operation_name}",
                    extra=context.to_dict()
                )
                
                # Check if we should retry
                if (strategy.retry_enabled and 
                    attempt < strategy.max_retries and 
                    context.severity != ErrorSeverity.CRITICAL):
                    
                    delay = strategy.get_delay(attempt)
                    self.logger.info(
                        f"Retrying in {delay:.1f}s: {component}.{operation_name}",
                        extra={"delay": delay, "attempt": attempt + 1}
                    )
                    await asyncio.sleep(delay)
                    continue
                
                # Handle fallback or failure
                return await self._handle_failure(context, strategy)
    
    def handle_sync(
        self,
        operation: Callable,
        component: str,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Handle sync operation with error recovery."""
        context = None
        
        for attempt in range(3):  # Max attempts
            try:
                result = operation(*args, **kwargs)
                
                if context:  # Log successful recovery
                    self.logger.info(
                        f"Operation recovered successfully: {component}.{operation_name}",
                        extra={
                            "component": component,
                            "operation": operation_name,
                            "attempt": attempt + 1,
                            "recovery": True
                        }
                    )
                
                return result
                
            except Exception as error:
                context = self.classifier.classify(error, component)
                context.operation = operation_name
                context.retry_count = attempt
                
                strategy = self.strategies.get(context.category, self.strategies[ErrorCategory.UNKNOWN])
                
                self.logger.warning(
                    f"Operation failed: {component}.{operation_name}",
                    extra=context.to_dict()
                )
                
                # Check if we should retry
                if (strategy.retry_enabled and 
                    attempt < strategy.max_retries and 
                    context.severity != ErrorSeverity.CRITICAL):
                    
                    delay = strategy.get_delay(attempt)
                    self.logger.info(
                        f"Retrying in {delay:.1f}s: {component}.{operation_name}",
                        extra={"delay": delay, "attempt": attempt + 1}
                    )
                    time.sleep(delay)
                    continue
                
                # Handle fallback or failure
                return self._handle_failure_sync(context, strategy)
    
    async def _handle_failure(
        self, 
        context: ErrorContext, 
        strategy: RecoveryStrategy
    ) -> Any:
        """Handle operation failure with fallback strategies."""
        if strategy.fallback_enabled:
            self.logger.warning(
                f"Using fallback for {context.component}.{context.operation}",
                extra=context.to_dict()
            )
            return strategy.fallback_value
        
        if strategy.degraded_mode:
            self.logger.error(
                f"Entering degraded mode for {context.component}.{context.operation}",
                extra=context.to_dict()
            )
            # Return a degraded result that indicates partial failure
            return None
        
        # Re-raise the original error if no recovery possible
        self.logger.error(
            f"No recovery possible for {context.component}.{context.operation}",
            extra=context.to_dict()
        )
        raise context.error
    
    def _handle_failure_sync(
        self, 
        context: ErrorContext, 
        strategy: RecoveryStrategy
    ) -> Any:
        """Handle sync operation failure with fallback strategies."""
        if strategy.fallback_enabled:
            self.logger.warning(
                f"Using fallback for {context.component}.{context.operation}",
                extra=context.to_dict()
            )
            return strategy.fallback_value
        
        if strategy.degraded_mode:
            self.logger.error(
                f"Entering degraded mode for {context.component}.{context.operation}",
                extra=context.to_dict()
            )
            # Return a degraded result that indicates partial failure
            return None
        
        # Re-raise the original error if no recovery possible
        self.logger.error(
            f"No recovery possible for {context.component}.{context.operation}",
            extra=context.to_dict()
        )
        raise context.error


# Global error recovery manager
_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = ErrorRecoveryManager()
    return _recovery_manager


# Convenient decorator for error handling
def with_error_recovery(component: str, operation: str = None):
    """Decorator for automatic error recovery on functions."""
    def decorator(func):
        op_name = operation or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                recovery_manager = get_recovery_manager()
                return await recovery_manager.handle_async(
                    func, component, op_name, *args, **kwargs
                )
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                recovery_manager = get_recovery_manager()
                return recovery_manager.handle_sync(
                    func, component, op_name, *args, **kwargs
                )
            return sync_wrapper
    
    return decorator


# Context managers for error handling
@asynccontextmanager
async def error_context(component: str, operation: str):
    """Async context manager for error handling."""
    recovery_manager = get_recovery_manager()
    logger = get_logger(f"error_context.{component}")
    
    start_time = time.time()
    logger.debug(f"Starting operation: {component}.{operation}")
    
    try:
        yield recovery_manager
        duration = time.time() - start_time
        logger.debug(
            f"Operation completed: {component}.{operation}",
            extra={"duration": duration, "success": True}
        )
    except Exception as e:
        duration = time.time() - start_time
        context = recovery_manager.classifier.classify(e, component)
        context.operation = operation
        
        logger.error(
            f"Operation failed: {component}.{operation}",
            extra={**context.to_dict(), "duration": duration}
        )
        raise


@contextmanager
def sync_error_context(component: str, operation: str):
    """Sync context manager for error handling."""
    recovery_manager = get_recovery_manager()
    logger = get_logger(f"error_context.{component}")
    
    start_time = time.time()
    logger.debug(f"Starting operation: {component}.{operation}")
    
    try:
        yield recovery_manager
        duration = time.time() - start_time
        logger.debug(
            f"Operation completed: {component}.{operation}",
            extra={"duration": duration, "success": True}
        )
    except Exception as e:
        duration = time.time() - start_time
        context = recovery_manager.classifier.classify(e, component)
        context.operation = operation
        
        logger.error(
            f"Operation failed: {component}.{operation}",
            extra={**context.to_dict(), "duration": duration}
        )
        raise 