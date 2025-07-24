"""Abstract base class for all evaluation metrics."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from .types import RAGInput, MetricResult, MetricType
from .logging import get_logger, MetricLogger
from .exceptions import (
    MetricError, MetricInputError, MetricConfigurationError, 
    MetricEvaluationError, handle_metric_error
)
from ..llm import generate_with_langchain, LLMResponse
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from .error_handler import with_error_recovery, error_context


class BaseMetric(ABC):
    """Abstract base class for all RAG evaluation metrics.
    
    All metrics must implement the evaluate method and provide a name.
    Metrics can optionally specify their type, requirements, and configuration.
    Includes comprehensive logging and error handling capabilities.
    """

    def __init__(
        self,
        llm: Optional[Union[ChatOpenAI, AzureChatOpenAI, Any]] = None,
        cache_enabled: bool = True,
        **kwargs: Any
    ):
        """Initialize the metric.
        
        Args:
            llm: LangChain LLM instance for LLM-based metrics
            cache_enabled: Whether to enable caching of results
            **kwargs: Additional configuration parameters
        """
        # Standardized LLM parameter handling
        self.llm = llm
        self.cache_enabled = cache_enabled
        self.config = kwargs
        self._cache: Dict[str, MetricResult] = {}
        
        # Initialize logging
        self.logger = get_logger(f"metrics.{self.__class__.__name__.lower()}")
        self.metric_logger = MetricLogger(self.name())
        
        # Log initialization
        self.logger.info(
            f"Initialized {self.name()} metric",
            extra={
                "metric": self.name(),
                "metric_type": self.metric_type.value,
                "cache_enabled": cache_enabled,
                "llm_configured": self.llm is not None,
                "config": self.config
            }
        )

    async def generate_llm_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate LLM response using standardized interface.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens for response
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            MetricConfigurationError: If no LLM is configured
            MetricEvaluationError: If generation fails
        """
        start_time = time.time()
        
        try:
            if self.llm is not None:
                # Use new LangChain interface
                generation_kwargs = {"temperature": temperature, **kwargs}
                if max_tokens:
                    generation_kwargs["max_tokens"] = max_tokens
                    
                self.logger.debug(
                    f"Generating LLM response for {self.name()}",
                    extra={
                        "metric": self.name(),
                        "prompt_length": len(prompt),
                        "has_system_prompt": system_prompt is not None,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                )
                
                response = await generate_with_langchain(
                    self.llm, prompt, system_prompt, **generation_kwargs
                )
                
                duration = time.time() - start_time
                self.metric_logger.log_llm_call(
                    prompt_length=len(prompt),
                    response_length=len(response.content),
                    duration=duration
                )
                
                return response
                
            else:
                raise MetricConfigurationError(
                    f"No LLM configured for metric {self.name()}. Please provide 'llm' parameter "
                    f"with a LangChain LLM instance (ChatOpenAI or AzureChatOpenAI).",
                    metric_name=self.name()
                )
                
        except MetricError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Convert other errors to metric errors
            raise handle_metric_error(
                e, 
                metric_name=self.name(),
                evaluation_stage="llm_generation"
            )

    @abstractmethod
    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate the RAG input and return a metric result.
        
        Args:
            rag_input: The RAG input to evaluate
            
        Returns:
            MetricResult containing score and explanation
            
        Raises:
            MetricInputError: If input is invalid
            MetricEvaluationError: If evaluation fails
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of this metric."""
        pass

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        pass

    def validate_input(self, rag_input: RAGInput) -> None:
        """Validate the RAG input for this metric.
        
        Args:
            rag_input: The RAG input to validate
            
        Raises:
            MetricInputError: If input is invalid for this metric
        """
        try:
            if not isinstance(rag_input, RAGInput):
                raise MetricInputError(
                    "Input must be a RAGInput instance",
                    metric_name=self.name(),
                    input_type=type(rag_input).__name__
                )
            
            if not rag_input.query or not rag_input.query.strip():
                raise MetricInputError(
                    "Query cannot be empty",
                    metric_name=self.name(),
                    input_type="query"
                )
            
            # Specific validation can be overridden by subclasses
            self._validate_metric_specific(rag_input)
            
            self.logger.debug(
                f"Input validation passed for {self.name()}",
                extra={
                    "metric": self.name(),
                    "query_length": len(rag_input.query),
                    "answer_length": len(rag_input.answer) if rag_input.answer else 0,
                    "contexts_count": len(rag_input.retrieved_contexts) if rag_input.retrieved_contexts is not None else 0
                }
            )
            
        except MetricInputError:
            # Re-raise our validation errors
            raise
        except Exception as e:
            # Convert other validation errors
            raise MetricInputError(
                f"Input validation failed: {str(e)}",
                metric_name=self.name(),
                original_error=e
            )

    def _validate_metric_specific(self, rag_input: RAGInput) -> None:
        """Override this method for metric-specific validation.
        
        Args:
            rag_input: The RAG input to validate
            
        Raises:
            MetricInputError: If metric-specific validation fails
        """
        pass

    def _get_cache_key(self, rag_input: RAGInput) -> str:
        """Generate cache key for the input.
        
        Args:
            rag_input: The RAG input
            
        Returns:
            Cache key string
        """
        # Simple cache key based on input hash
        import hashlib
        content = f"{rag_input.query}|{rag_input.answer}|{len(rag_input.retrieved_contexts or [])}"
        return hashlib.md5(content.encode()).hexdigest()

    async def evaluate_with_cache(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate with caching support, comprehensive logging, and error recovery.
        
        Args:
            rag_input: The RAG input to evaluate
            
        Returns:
            MetricResult (from cache or fresh evaluation)
            
        Raises:
            MetricError: If evaluation fails
        """
        # Prepare logging context
        rag_input_info = {
            "query_length": len(rag_input.query),
            "answer_length": len(rag_input.answer) if rag_input.answer else 0,
            "contexts_count": len(rag_input.retrieved_contexts) if rag_input.retrieved_contexts is not None else 0
        }
        
        async with error_context("metric", f"{self.name()}_evaluation"):
            try:
                self.metric_logger.start_evaluation(rag_input_info)
                
                if not self.cache_enabled:
                    result = await self.evaluate(rag_input)
                    self.metric_logger.end_evaluation(result.score, success=True)
                    return result

                cache_key = self._get_cache_key(rag_input)
                if cache_key in self._cache:
                    cached_result = self._cache[cache_key]
                    
                    self.logger.debug(
                        f"Cache hit for {self.name()}",
                        extra={
                            "metric": self.name(),
                            "cache_key": cache_key,
                            "cached_score": cached_result.score
                        }
                    )
                    
                    self.metric_logger.end_evaluation(cached_result.score, success=True)
                    return cached_result

                # Perform fresh evaluation with error recovery
                result = await self._evaluate_with_recovery(rag_input)
                
                if result is not None:
                    self._cache[cache_key] = result
                    
                    self.logger.debug(
                        f"Cached new result for {self.name()}",
                        extra={
                            "metric": self.name(),
                            "cache_key": cache_key,
                            "score": result.score
                        }
                    )
                    
                    self.metric_logger.end_evaluation(result.score, success=True)
                    return result
                else:
                    # Degraded mode - return default result
                    degraded_result = MetricResult(
                        score=0.0,
                        explanation=f"Metric {self.name()} evaluation failed - using degraded mode",
                        details={"degraded_mode": True, "error": "evaluation_failed"}
                    )
                    self.metric_logger.end_evaluation(0.0, success=False, error="degraded_mode")
                    return degraded_result

            except MetricError as e:
                # Re-raise our custom errors with logging
                error_msg = f"Metric evaluation failed: {str(e)}"
                self.metric_logger.end_evaluation(0.0, success=False, error=error_msg)
                raise
            except Exception as e:
                # Convert and log other errors
                error_msg = f"Unexpected error in metric evaluation: {str(e)}"
                self.metric_logger.end_evaluation(0.0, success=False, error=error_msg)
                
                metric_error = handle_metric_error(e, metric_name=self.name())
                raise metric_error
    
    async def _evaluate_with_recovery(self, rag_input: RAGInput) -> Optional[MetricResult]:
        """Evaluate metric with error recovery support."""
        from .error_handler import get_recovery_manager
        
        recovery_manager = get_recovery_manager()
        
        # Wrap the evaluation in error recovery
        result = await recovery_manager.handle_async(
            self.evaluate,
            "metric",
            f"{self.name()}_evaluate",
            rag_input
        )
        
        return result

    @property
    def requires_llm(self) -> bool:
        """Whether this metric requires an LLM for evaluation."""
        # Most metrics require an LLM unless they're simple rule-based metrics
        return True

    @property
    def requires_contexts(self) -> bool:
        """Whether this metric requires retrieved contexts."""
        # Override in subclasses if needed
        return self.metric_type in [MetricType.RETRIEVAL, MetricType.COMPOSITE]

    @property
    def requires_answer(self) -> bool:
        """Whether this metric requires a generated answer."""
        # Override in subclasses if needed
        return self.metric_type in [MetricType.GENERATION, MetricType.COMPOSITE]

    def clear_cache(self) -> None:
        """Clear the metric's cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        
        self.logger.info(
            f"Cleared cache for {self.name()}",
            extra={
                "metric": self.name(),
                "cleared_entries": cache_size
            }
        )

    def get_cache_size(self) -> int:
        """Get the number of cached results."""
        return len(self._cache)

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        return {
            "metric": self.name(),
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys()) if len(self._cache) < 10 else "too_many_to_list"
        }

    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.name()} ({self.metric_type.value})"

    def __repr__(self) -> str:
        """Detailed string representation of the metric."""
        return (f"{self.__class__.__name__}(name='{self.name()}', "
                f"type='{self.metric_type.value}', "
                f"requires_llm={self.requires_llm}, "
                f"cache_enabled={self.cache_enabled})") 