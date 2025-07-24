"""Centralized logging configuration for RAG evaluation system.

This module provides structured logging capabilities with consistent formatting,
performance monitoring, and error tracking across all RAG evaluation components.
"""

import logging
import logging.config
import time
import functools
import traceback
from typing import Dict, Any, Optional, Callable, Union
from pathlib import Path
from datetime import datetime
import sys
import os


class RAGEvalLogger:
    """Centralized logger for RAG evaluation system with structured logging."""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            self._initialized = True
    
    def _setup_logging(self):
        """Setup logging configuration with structured format."""
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Determine log level from environment
        log_level = os.getenv("RAG_EVALS_LOG_LEVEL", "INFO").upper()
        
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "simple": {
                    "format": "%(levelname)s | %(name)s | %(message)s"
                },
                "json": {
                    "format": "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(lineno)d | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "simple",
                    "stream": sys.stdout
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": log_dir / "rag_evals.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": log_dir / "rag_evals_errors.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                }
            },
            "loggers": {
                "rag_evals": {
                    "level": "DEBUG",
                    "handlers": ["console", "file", "error_file"],
                    "propagate": False
                },
                "rag_evals.metrics": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "rag_evals.evaluator": {
                    "level": "DEBUG", 
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "rag_evals.llm": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            },
            "root": {
                "level": log_level,
                "handlers": ["console"]
            }
        }
        
        logging.config.dictConfig(logging_config)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for the given name.
        
        Args:
            name: Logger name (typically module name)
            
        Returns:
            Configured logger instance
        """
        if name not in self._loggers:
            logger = logging.getLogger(f"rag_evals.{name}")
            self._loggers[name] = logger
        
        return self._loggers[name]


# Global logger instance
_logger_instance = RAGEvalLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module.
    
    Args:
        name: Module or component name
        
    Returns:
        Configured logger instance
    """
    return _logger_instance.get_logger(name)


class MetricLogger:
    """Specialized logger for metric evaluation with performance tracking."""
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.logger = get_logger(f"metrics.{metric_name}")
        self._start_time = None
        self._evaluation_count = 0
    
    def start_evaluation(self, rag_input_info: Dict[str, Any]) -> None:
        """Log the start of metric evaluation.
        
        Args:
            rag_input_info: Information about the RAG input being evaluated
        """
        self._start_time = time.time()
        self._evaluation_count += 1
        
        self.logger.info(
            f"Starting {self.metric_name} evaluation #{self._evaluation_count}",
            extra={
                "metric": self.metric_name,
                "evaluation_id": self._evaluation_count,
                "query_length": rag_input_info.get("query_length", 0),
                "contexts_count": rag_input_info.get("contexts_count", 0),
                "answer_length": rag_input_info.get("answer_length", 0)
            }
        )
    
    def end_evaluation(self, score: float, success: bool = True, error: Optional[str] = None) -> None:
        """Log the end of metric evaluation.
        
        Args:
            score: Evaluation score
            success: Whether evaluation was successful
            error: Error message if evaluation failed
        """
        end_time = time.time()
        duration = end_time - self._start_time if self._start_time else 0
        
        if success:
            self.logger.info(
                f"Completed {self.metric_name} evaluation #{self._evaluation_count}",
                extra={
                    "metric": self.metric_name,
                    "evaluation_id": self._evaluation_count,
                    "score": score,
                    "duration_seconds": round(duration, 3),
                    "success": True
                }
            )
        else:
            self.logger.error(
                f"Failed {self.metric_name} evaluation #{self._evaluation_count}",
                extra={
                    "metric": self.metric_name,
                    "evaluation_id": self._evaluation_count,
                    "duration_seconds": round(duration, 3),
                    "success": False,
                    "error": error
                }
            )
        
        self._start_time = None
    
    def log_llm_call(self, prompt_length: int, response_length: int, duration: float) -> None:
        """Log LLM API call details.
        
        Args:
            prompt_length: Length of the prompt sent to LLM
            response_length: Length of LLM response
            duration: Time taken for the API call
        """
        self.logger.debug(
            f"LLM call for {self.metric_name}",
            extra={
                "metric": self.metric_name,
                "llm_call": True,
                "prompt_length": prompt_length,
                "response_length": response_length,
                "duration_seconds": round(duration, 3)
            }
        )


def log_function_call(logger: Optional[logging.Logger] = None):
    """Decorator to log function calls with timing and error handling.
    
    Args:
        logger: Logger instance to use (if None, creates one based on module)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__.split('.')[-1])
            start_time = time.time()
            
            func_logger.debug(
                f"Starting {func.__name__}",
                extra={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                func_logger.debug(
                    f"Completed {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "duration_seconds": round(duration, 3),
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                func_logger.error(
                    f"Failed {func.__name__}: {str(e)}",
                    extra={
                        "function": func.__name__,
                        "duration_seconds": round(duration, 3),
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__.split('.')[-1])
            start_time = time.time()
            
            func_logger.debug(
                f"Starting {func.__name__}",
                extra={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                func_logger.debug(
                    f"Completed {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "duration_seconds": round(duration, 3),
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                func_logger.error(
                    f"Failed {func.__name__}: {str(e)}",
                    extra={
                        "function": func.__name__,
                        "duration_seconds": round(duration, 3),
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                raise
        
        # Return appropriate wrapper based on whether function is async
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class EvaluationContext:
    """Context manager for tracking evaluation sessions with logging."""
    
    def __init__(self, session_name: str, logger: Optional[logging.Logger] = None):
        self.session_name = session_name
        self.logger = logger or get_logger("evaluator")
        self.start_time = None
        self.metrics_evaluated = []
        self.errors = []
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(
            f"Starting evaluation session: {self.session_name}",
            extra={
                "session": self.session_name,
                "event": "session_start"
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed evaluation session: {self.session_name}",
                extra={
                    "session": self.session_name,
                    "event": "session_complete",
                    "duration_seconds": round(duration, 3),
                    "metrics_evaluated": len(self.metrics_evaluated),
                    "errors_count": len(self.errors),
                    "success": True
                }
            )
        else:
            self.logger.error(
                f"Failed evaluation session: {self.session_name} - {str(exc_val)}",
                extra={
                    "session": self.session_name,
                    "event": "session_failed",
                    "duration_seconds": round(duration, 3),
                    "error": str(exc_val),
                    "error_type": exc_type.__name__ if exc_type else None,
                    "success": False
                }
            )
    
    def add_metric(self, metric_name: str, score: float, success: bool = True) -> None:
        """Add a metric result to the session tracking.
        
        Args:
            metric_name: Name of the evaluated metric
            score: Metric score
            success: Whether metric evaluation was successful
        """
        self.metrics_evaluated.append({
            "metric": metric_name,
            "score": score,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
        if success:
            self.logger.debug(
                f"Metric completed in session {self.session_name}: {metric_name} = {score}",
                extra={
                    "session": self.session_name,
                    "metric": metric_name,
                    "score": score,
                    "event": "metric_complete"
                }
            )
    
    def add_error(self, error: Exception, context: str = "") -> None:
        """Add an error to the session tracking.
        
        Args:
            error: Exception that occurred
            context: Additional context about the error
        """
        error_info = {
            "error": str(error),
            "error_type": type(error).__name__,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        self.errors.append(error_info)
        
        self.logger.warning(
            f"Error in session {self.session_name}: {str(error)}",
            extra={
                "session": self.session_name,
                "error": str(error),
                "error_type": type(error).__name__,
                "context": context,
                "event": "session_error"
            }
        )


def configure_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> None:
    """Configure global logging settings.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to files
        log_dir: Directory for log files
    """
    os.environ["RAG_EVALS_LOG_LEVEL"] = log_level.upper()
    
    if log_to_file:
        Path(log_dir).mkdir(exist_ok=True)
    
    # Reinitialize logger with new settings
    global _logger_instance
    _logger_instance = RAGEvalLogger()


__all__ = [
    "get_logger",
    "MetricLogger", 
    "log_function_call",
    "EvaluationContext",
    "configure_logging"
] 