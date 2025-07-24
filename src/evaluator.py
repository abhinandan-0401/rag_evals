"""Unified RAG evaluation interface."""

import asyncio
import time
from typing import Dict, List, Optional, Union, Any
import warnings

from .llm import create_llm, LLMResponse, ChatOpenAI, AzureChatOpenAI, get_llm_manager
from .core.types import RAGInput, MetricResult, EvaluationResult
from .core.base_metric import BaseMetric
from .core.logging import get_logger, EvaluationContext
from .core.exceptions import (
    EvaluatorError, EvaluatorConfigurationError, BatchEvaluationError,
    MetricError, handle_llm_error
)
from .metrics import (
    Faithfulness, AnswerRelevance, AnswerCorrectness, Completeness, Coherence, Helpfulness,
    ContextRecall, ContextRelevance, ContextPrecision,
    LLMJudge, RAGCertainty, TrustScore
)


class RAGEvaluator:
    """Unified interface for evaluating RAG systems.
    
    This class provides a single entry point for evaluating RAG systems across
    multiple metrics including generation, retrieval, and composite metrics.
    
    Supports both simple string-based metric configuration and advanced 
    BaseMetric instance configuration for maximum flexibility.
    
    Primary interface uses Azure OpenAI by default for enterprise deployments.
    Includes comprehensive logging and error handling capabilities.
    """
    
    # Available metrics mapping
    AVAILABLE_METRICS = {
        # Generation metrics
        "faithfulness": Faithfulness,
        "answer_relevance": AnswerRelevance,
        "answer_correctness": AnswerCorrectness,
        "completeness": Completeness,
        "coherence": Coherence,
        "helpfulness": Helpfulness,
        
        # Retrieval metrics
        "context_recall": ContextRecall,
        "context_relevance": ContextRelevance,
        "context_precision": ContextPrecision,
        
        # Composite metrics
        "llm_judge": LLMJudge,
        "rag_certainty": RAGCertainty,
        "trust_score": TrustScore,
    }
    
    def __init__(
        self,
        metrics: List[Union[str, BaseMetric]] = None,
        llm: Optional[Union[ChatOpenAI, AzureChatOpenAI]] = None,
        provider: str = "azure",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        azure_config: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        use_llm_manager: bool = True,
        **kwargs
    ):
        """Initialize the RAG evaluator.
        
        Args:
            metrics: List of metrics to use (strings or BaseMetric instances)
            llm: Pre-configured LangChain LLM instance (overrides other LLM params)
            provider: LLM provider ("azure" or "openai"). Default "azure".
            model: Model name (e.g., "gpt-4", "gpt-35-turbo")
            api_key: API key for the LLM provider
            azure_config: Azure-specific configuration
            temperature: Sampling temperature for LLM responses
            max_tokens: Maximum tokens for LLM responses
            use_llm_manager: Whether to use LLM Manager for pooling/caching (default True)
            **kwargs: Additional configuration
            
        Raises:
            EvaluatorConfigurationError: If configuration is invalid
            EvaluatorError: If LLM initialization fails
        """
        # Initialize logging
        self.logger = get_logger("evaluator")
        
        # Set default metrics if none provided
        if metrics is None:
            metrics = ["faithfulness", "answer_relevance", "context_relevance"]
        
        self.logger.info(
            "Initializing RAGEvaluator",
            extra={
                "provider": provider,
                "model": model,
                "metrics_requested": [m if isinstance(m, str) else m.__class__.__name__ for m in metrics],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        
        # Initialize LLM
        if llm is not None:
            self.llm = llm
            self.use_llm_manager = False  # Track if we're using the manager
            self.logger.info("Using pre-configured LLM instance")
        else:
            try:
                if use_llm_manager:
                    # Use LLM Manager for connection pooling and caching
                    llm_manager = get_llm_manager()
                    self.llm = llm_manager.get_llm(
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        azure_config=azure_config,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                    self.use_llm_manager = True
                    self.llm_manager = llm_manager
                    self.logger.info(
                        f"Successfully initialized {provider} LLM with Manager",
                        extra={
                            "provider": provider,
                            "model": model,
                            "llm_type": type(self.llm).__name__,
                            "using_manager": True
                        }
                    )
                else:
                    # Use direct creation (legacy mode)
                    self.llm = create_llm(
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        azure_config=azure_config,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                    self.use_llm_manager = False
                    self.llm_manager = None
                    self.logger.info(
                        f"Successfully initialized {provider} LLM (direct)",
                        extra={
                            "provider": provider,
                            "model": model,
                            "llm_type": type(self.llm).__name__,
                            "using_manager": False
                        }
                    )
            except Exception as e:
                llm_error = handle_llm_error(e, provider)
                self.logger.error(
                    f"Failed to initialize LLM: {str(llm_error)}",
                    extra=llm_error.to_dict()
                )
                raise EvaluatorError(
                    f"Failed to initialize LLM: {str(llm_error)}",
                    original_error=llm_error
                )
        
        # Initialize metrics
        self.metrics = []
        failed_metrics = []
        
        for metric in metrics:
            try:
                if isinstance(metric, str):
                    if metric not in self.AVAILABLE_METRICS:
                        available = ", ".join(self.AVAILABLE_METRICS.keys())
                        raise EvaluatorConfigurationError(
                            f"Unknown metric '{metric}'. Available: {available}",
                            context={"requested_metric": metric, "available_metrics": list(self.AVAILABLE_METRICS.keys())}
                        )
                    
                    # Create metric instance with LLM
                    metric_class = self.AVAILABLE_METRICS[metric]
                    # Pass only llm parameter for consistency
                    metric_instance = metric_class(
                        llm=self.llm, 
                        **kwargs
                    )
                    self.metrics.append(metric_instance)
                    
                    self.logger.debug(
                        f"Successfully initialized metric: {metric}",
                        extra={"metric": metric, "metric_class": metric_class.__name__}
                    )
                    
                elif isinstance(metric, BaseMetric):
                    # Use provided metric instance, ensure it has LLM if needed
                    if metric.requires_llm and not metric.llm:
                        metric.llm = self.llm
                        self.logger.debug(
                            f"Assigned LLM to pre-configured metric: {metric.name()}",
                            extra={"metric": metric.name()}
                        )
                    
                    self.metrics.append(metric)
                    
                    self.logger.debug(
                        f"Using pre-configured metric: {metric.name()}",
                        extra={"metric": metric.name(), "metric_class": metric.__class__.__name__}
                    )
                else:
                    raise EvaluatorConfigurationError(
                        f"Invalid metric type: {type(metric)}. Use string or BaseMetric instance.",
                        context={"metric_type": type(metric).__name__}
                    )
                    
            except Exception as e:
                metric_name = metric if isinstance(metric, str) else getattr(metric, 'name', str(metric))
                failed_metrics.append(metric_name)
                
                self.logger.error(
                    f"Failed to initialize metric '{metric_name}': {str(e)}",
                    extra={
                        "metric": metric_name,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                
                # Continue with other metrics unless it's a critical configuration error
                if not isinstance(e, (EvaluatorConfigurationError, EvaluatorError)):
                    continue
                else:
                    raise
        
        if not self.metrics:
            raise EvaluatorConfigurationError(
                "No valid metrics configured",
                context={"failed_metrics": failed_metrics}
            )
        
        if failed_metrics:
            self.logger.warning(
                f"Some metrics failed to initialize: {failed_metrics}",
                extra={"failed_metrics": failed_metrics, "successful_metrics": [m.name() for m in self.metrics]}
            )
        
        self.config = kwargs
        
        # Log final configuration
        self.logger.info(
            "RAGEvaluator initialization complete",
            extra={
                "total_metrics": len(self.metrics),
                "metrics": [m.name() for m in self.metrics],
                "failed_metrics": failed_metrics,
                "llm_provider": "azure" if isinstance(self.llm, AzureChatOpenAI) else "openai",
                "using_llm_manager": getattr(self, 'use_llm_manager', False)
            }
        )

    @classmethod
    def with_azure_openai(
        cls,
        azure_endpoint: str,
        azure_deployment: str,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        metrics: List[Union[str, BaseMetric]] = None,
        **kwargs
    ) -> "RAGEvaluator":
        """Create RAGEvaluator with Azure OpenAI configuration.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            azure_deployment: Azure OpenAI deployment name
            api_key: Azure OpenAI API key (or use AZURE_OPENAI_API_KEY env var)
            api_version: Azure OpenAI API version
            metrics: List of metrics to use
            **kwargs: Additional configuration
            
        Returns:
            Configured RAGEvaluator instance
            
        Raises:
            EvaluatorConfigurationError: If Azure configuration is invalid
        """
        logger = get_logger("evaluator")
        logger.info(
            "Creating RAGEvaluator with Azure OpenAI",
            extra={
                "azure_endpoint": azure_endpoint,
                "azure_deployment": azure_deployment,
                "api_version": api_version,
                "metrics_count": len(metrics) if metrics else 0
            }
        )
        
        azure_config = {
            "azure_endpoint": azure_endpoint,
            "azure_deployment": azure_deployment,
            "api_key": api_key,
            "api_version": api_version,
        }
        
        return cls(
            metrics=metrics,
            provider="azure",
            api_key=api_key,
            azure_config=azure_config,
            **kwargs
        )

    @classmethod  
    def with_openai(
        cls,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        metrics: List[Union[str, BaseMetric]] = None,
        **kwargs
    ) -> "RAGEvaluator":
        """Create RAGEvaluator with OpenAI configuration.
        
        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: OpenAI model name
            metrics: List of metrics to use
            **kwargs: Additional configuration
            
        Returns:
            Configured RAGEvaluator instance
            
        Raises:
            EvaluatorConfigurationError: If OpenAI configuration is invalid
        """
        logger = get_logger("evaluator")
        logger.info(
            "Creating RAGEvaluator with OpenAI",
            extra={
                "model": model,
                "metrics_count": len(metrics) if metrics else 0
            }
        )
        
        return cls(
            metrics=metrics,
            provider="openai",
            model=model,
            api_key=api_key,
            **kwargs
        )

    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_contexts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate a single RAG response.
        
        Args:
            query: The user's query
            answer: The generated answer
            retrieved_contexts: List of retrieved context passages
            metadata: Optional metadata for the evaluation
            
        Returns:
            EvaluationResult with scores from all metrics
            
        Raises:
            EvaluatorError: If required inputs are missing or evaluation fails
        """
        session_id = f"single_eval_{int(time.time() * 1000)}"
        
        with EvaluationContext(session_id, self.logger) as session:
            try:
                # Validate inputs
                if not query or not query.strip():
                    raise EvaluatorError(
                        "Query cannot be empty",
                        context={"query_length": len(query) if query else 0}
                    )
                if not answer or not answer.strip():
                    raise EvaluatorError(
                        "Answer cannot be empty",
                        context={"answer_length": len(answer) if answer else 0}
                    )
                
                # Create RAGInput
                rag_input = RAGInput(
                    query=query,
                    answer=answer,
                    retrieved_contexts=retrieved_contexts or [],
                    metadata=metadata or {}
                )
                
                self.logger.info(
                    f"Starting evaluation: {session_id}",
                    extra={
                        "session_id": session_id,
                        "query_length": len(query),
                        "answer_length": len(answer),
                        # Fix: Use explicit None and length check instead of boolean evaluation
                        "contexts_count": len(retrieved_contexts) if retrieved_contexts is not None else 0,
                        "metrics_count": len(self.metrics)
                    }
                )
                
                # Evaluate all metrics
                start_time = time.time()
                metric_results = await self._evaluate_metrics(rag_input, session)
                evaluation_time = time.time() - start_time
                
                # Calculate overall score (average of all metric scores)
                scores = [result.score for result in metric_results.values() if result.score is not None]
                overall_score = sum(scores) / len(scores) if scores else 0.0
                
                result = EvaluationResult(
                    overall_score=overall_score,
                    metric_results=metric_results,
                    evaluation_time=evaluation_time,
                    metadata={
                        "model": getattr(self.llm, 'model_name', 'unknown'),
                        "provider": "azure" if isinstance(self.llm, AzureChatOpenAI) else "openai",
                        "metrics_evaluated": list(metric_results.keys()),
                        "session_id": session_id,
                        **(metadata or {})
                    }
                )
                
                self.logger.info(
                    f"Evaluation completed: {session_id}",
                    extra={
                        "session_id": session_id,
                        "overall_score": overall_score,
                        "evaluation_time": evaluation_time,
                        "metrics_completed": len(metric_results),
                        "successful_metrics": len([r for r in metric_results.values() if r.score is not None])
                    }
                )
                
                return result
                
            except EvaluatorError:
                # Re-raise our custom errors
                raise
            except Exception as e:
                error_msg = f"Evaluation failed: {str(e)}"
                self.logger.error(
                    error_msg,
                    extra={
                        "session_id": session_id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                raise EvaluatorError(error_msg, original_error=e)

    async def evaluate_batch(
        self,
        inputs: List[Dict[str, Any]],
        batch_size: int = 5,
        show_progress: bool = True
    ) -> List[EvaluationResult]:
        """Evaluate a batch of RAG responses.
        
        Args:
            inputs: List of input dictionaries with 'query', 'answer', and optional 'retrieved_contexts'
            batch_size: Number of evaluations to run concurrently
            show_progress: Whether to show progress information
            
        Returns:
            List of EvaluationResult objects
            
        Raises:
            BatchEvaluationError: If inputs are invalid or batch evaluation fails
        """
        session_id = f"batch_eval_{int(time.time() * 1000)}"
        
        with EvaluationContext(session_id, self.logger) as session:
            try:
                if not inputs:
                    raise BatchEvaluationError(
                        "No inputs provided for batch evaluation",
                        context={"inputs_count": 0}
                    )
                
                self.logger.info(
                    f"Starting batch evaluation: {session_id}",
                    extra={
                        "session_id": session_id,
                        "total_inputs": len(inputs),
                        "batch_size": batch_size,
                        "metrics_count": len(self.metrics)
                    }
                )
                
                results = []
                failed_count = 0
                
                # Process in batches
                for i in range(0, len(inputs), batch_size):
                    batch = inputs[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (len(inputs) + batch_size - 1) // batch_size
                    
                    if show_progress:
                        print(f"Processing batch {batch_num}/{total_batches}")
                    
                    self.logger.debug(
                        f"Processing batch {batch_num}/{total_batches}",
                        extra={
                            "session_id": session_id,
                            "batch_num": batch_num,
                            "batch_size": len(batch),
                            "total_batches": total_batches
                        }
                    )
                    
                    # Create evaluation tasks for the batch
                    tasks = []
                    for idx, input_data in enumerate(batch):
                        if not isinstance(input_data, dict):
                            raise BatchEvaluationError(
                                f"Each input must be a dictionary, got {type(input_data)}",
                                context={"input_index": i + idx, "input_type": type(input_data).__name__}
                            )
                        
                        tasks.append(self.evaluate(
                            query=input_data.get("query", ""),
                            answer=input_data.get("answer", ""),
                            retrieved_contexts=input_data.get("retrieved_contexts"),
                            metadata=input_data.get("metadata")
                        ))
                    
                    # Execute batch
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results and handle exceptions
                    for idx, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            failed_count += 1
                            # Create error result
                            error_result = EvaluationResult(
                                overall_score=0.0,
                                metric_results={},
                                evaluation_time=0.0,
                                metadata={
                                    "error": str(result),
                                    "error_type": type(result).__name__,
                                    "batch_index": i + idx,
                                    "session_id": session_id
                                }
                            )
                            results.append(error_result)
                            
                            session.add_error(result, f"batch_item_{i + idx}")
                            
                            self.logger.error(
                                f"Failed to evaluate item {i + idx}",
                                extra={
                                    "session_id": session_id,
                                    "item_index": i + idx,
                                    "error": str(result),
                                    "error_type": type(result).__name__
                                }
                            )
                        else:
                            results.append(result)
                
                self.logger.info(
                    f"Batch evaluation completed: {session_id}",
                    extra={
                        "session_id": session_id,
                        "total_processed": len(results),
                        "successful_count": len(results) - failed_count,
                        "failed_count": failed_count,
                        "success_rate": (len(results) - failed_count) / len(results) if results else 0
                    }
                )
                
                if failed_count > 0:
                    self.logger.warning(
                        f"Batch evaluation had {failed_count} failures out of {len(results)} items",
                        extra={"session_id": session_id, "failed_count": failed_count, "total_count": len(results)}
                    )
                
                return results
                
            except BatchEvaluationError:
                # Re-raise batch-specific errors
                raise
            except Exception as e:
                error_msg = f"Batch evaluation failed: {str(e)}"
                self.logger.error(
                    error_msg,
                    extra={
                        "session_id": session_id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                raise BatchEvaluationError(
                    error_msg,
                    batch_size=batch_size,
                    original_error=e
                )

    async def _evaluate_metrics(self, rag_input: RAGInput, session: EvaluationContext = None) -> Dict[str, MetricResult]:
        """Evaluate all configured metrics for the given input.
        
        Args:
            rag_input: The RAG input to evaluate
            session: Optional evaluation session for tracking
            
        Returns:
            Dictionary mapping metric names to their results
        """
        # Create evaluation tasks
        tasks = []
        metric_names = []
        
        for metric in self.metrics:
            tasks.append(self._evaluate_single_metric(metric, rag_input))
            metric_names.append(metric.name())
        
        # Execute all metrics in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        metric_results = {}
        for metric_name, result in zip(metric_names, results):
            if isinstance(result, Exception):
                # Create error result
                error_result = MetricResult(
                    score=0.0,
                    explanation=f"Metric evaluation failed: {str(result)}",
                    details={
                        "error": str(result),
                        "error_type": type(result).__name__
                    }
                )
                metric_results[metric_name] = error_result
                
                # Add to session tracking
                if session:
                    session.add_metric(metric_name, 0.0, success=False)
                    session.add_error(result, f"metric_{metric_name}")
                
                self.logger.error(
                    f"Metric {metric_name} evaluation failed",
                    extra={
                        "metric": metric_name,
                        "error": str(result),
                        "error_type": type(result).__name__
                    }
                )
            else:
                metric_results[metric_name] = result
                
                # Add to session tracking
                if session:
                    session.add_metric(metric_name, result.score, success=True)
                
                self.logger.debug(
                    f"Metric {metric_name} completed successfully",
                    extra={
                        "metric": metric_name,
                        "score": result.score
                    }
                )
        
        return metric_results

    async def _evaluate_single_metric(self, metric: BaseMetric, rag_input: RAGInput) -> MetricResult:
        """Evaluate a single metric with error handling.
        
        Args:
            metric: The metric to evaluate
            rag_input: The RAG input to evaluate
            
        Returns:
            MetricResult from the metric
            
        Raises:
            MetricError: If metric evaluation fails
        """
        try:
            # Validate input for this metric
            metric.validate_input(rag_input)
            
            # Check metric requirements
            if metric.requires_llm and not metric.llm:
                raise MetricError(
                    f"Metric {metric.name()} requires an LLM but none is configured",
                    metric_name=metric.name()
                )
            
            if metric.requires_contexts and (rag_input.retrieved_contexts is None or len(rag_input.retrieved_contexts) == 0):
                return MetricResult(
                    score=0.0,
                    explanation=f"Metric {metric.name()} requires retrieved contexts but none provided",
                    details={"error": "missing_contexts"}
                )
            
            if metric.requires_answer and not rag_input.answer:
                return MetricResult(
                    score=0.0,
                    explanation=f"Metric {metric.name()} requires an answer but none provided",
                    details={"error": "missing_answer"}
                )
            
            # Evaluate the metric
            return await metric.evaluate_with_cache(rag_input)
            
        except MetricError:
            # Re-raise metric errors
            raise
        except Exception as e:
            # Convert other errors to metric errors
            raise MetricError(
                f"Metric {metric.name()} evaluation failed: {str(e)}",
                metric_name=metric.name(),
                original_error=e
            )

    def list_metrics(self) -> List[str]:
        """Get list of configured metric names."""
        return [metric.name() for metric in self.metrics]

    def get_metric_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about configured metrics."""
        info = {}
        for metric in self.metrics:
            info[metric.name()] = {
                "type": metric.metric_type.value,
                "requires_llm": metric.requires_llm,
                "requires_contexts": metric.requires_contexts,
                "requires_answer": metric.requires_answer,
                "class": metric.__class__.__name__,
                "cache_enabled": metric.cache_enabled,
                "cache_size": metric.get_cache_size()
            }
        return info

    @classmethod
    def list_available_metrics(cls) -> Dict[str, str]:
        """Get list of all available metrics."""
        return {name: cls_.__name__ for name, cls_ in cls.AVAILABLE_METRICS.items()}

    def clear_all_caches(self) -> None:
        """Clear caches for all configured metrics and LLM manager."""
        # Clear metric caches
        for metric in self.metrics:
            metric.clear_cache()
        
        # Clear LLM manager cache if available
        llm_cache_cleared = False
        if hasattr(self, 'llm_manager') and self.llm_manager:
            self.llm_manager.clear_cache()
            llm_cache_cleared = True
        
        self.logger.info(
            "Cleared caches for all metrics and LLM manager",
            extra={
                "metrics_cleared": [m.name() for m in self.metrics],
                "llm_cache_cleared": llm_cache_cleared
            }
        )

    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the evaluator and its metrics."""
        total_cache_size = sum(metric.get_cache_size() for metric in self.metrics)
        
        stats = {
            "evaluator": {
                "total_metrics": len(self.metrics),
                "llm_provider": "azure" if isinstance(self.llm, AzureChatOpenAI) else "openai",
                "llm_model": getattr(self.llm, 'model_name', 'unknown'),
                "using_llm_manager": getattr(self, 'use_llm_manager', False)
            },
            "metrics": {metric.name(): metric.get_cache_info() for metric in self.metrics},
            "cache_summary": {
                "total_cache_size": total_cache_size,
                "cache_enabled_metrics": len([m for m in self.metrics if m.cache_enabled])
            }
        }
        
        # Add LLM Manager statistics if available
        if hasattr(self, 'llm_manager') and self.llm_manager:
            stats["llm_manager"] = self.llm_manager.get_stats()
        
        return stats

    def __str__(self) -> str:
        """String representation of the evaluator."""
        provider = "Azure" if isinstance(self.llm, AzureChatOpenAI) else "OpenAI"
        metrics = ", ".join(self.list_metrics())
        return f"RAGEvaluator({provider}, metrics=[{metrics}])"

    def __repr__(self) -> str:
        """Detailed string representation of the evaluator."""
        return f"RAGEvaluator(llm={self.llm}, metrics={len(self.metrics)})" 