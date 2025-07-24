"""Base class for composite evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import asyncio

from ...core.base_metric import BaseMetric
from ...core.types import RAGInput, MetricResult, MetricType
from langchain_openai import ChatOpenAI, AzureChatOpenAI


class BaseCompositeMetric(BaseMetric):
    """Abstract base class for composite evaluation metrics.
    
    Composite metrics combine multiple sub-metrics or use complex
    evaluation logic that may involve multiple LLM calls or steps.
    """

    def __init__(
        self,
        llm: Optional[Union[ChatOpenAI, AzureChatOpenAI, Any]] = None,
        sub_metrics: Optional[List[BaseMetric]] = None,
        weights: Optional[Dict[str, float]] = None,
        **kwargs: Any
    ):
        """Initialize the composite metric.
        
        Args:
            llm: LLM instance for LLM-based evaluations
            sub_metrics: List of sub-metrics to combine (optional)
            weights: Weights for combining sub-metric scores (optional)
            **kwargs: Additional configuration parameters
        """
        super().__init__(llm=llm, **kwargs)
        self.sub_metrics = sub_metrics or []
        self.weights = weights or {}
        
        # Validate weights if provided
        if self.weights and self.sub_metrics:
            sub_metric_names = {metric.name() for metric in self.sub_metrics}
            weight_names = set(self.weights.keys())
            if not weight_names.issubset(sub_metric_names):
                extra_weights = weight_names - sub_metric_names
                raise ValueError(f"Weights provided for unknown metrics: {extra_weights}")

    @property
    def metric_type(self) -> MetricType:
        """Return the type of this composite metric."""
        return MetricType.COMPOSITE

    async def evaluate_sub_metrics(self, rag_input: RAGInput) -> Dict[str, MetricResult]:
        """Evaluate all sub-metrics in parallel.
        
        Args:
            rag_input: The RAG input to evaluate
            
        Returns:
            Dictionary mapping metric names to their results
        """
        if not self.sub_metrics:
            return {}
        
        # Run all sub-metrics in parallel
        tasks = [
            self._evaluate_single_sub_metric(metric, rag_input)
            for metric in self.sub_metrics
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        sub_results = {}
        for metric, result in zip(self.sub_metrics, results):
            if isinstance(result, Exception):
                # Log error and continue with other metrics
                sub_results[metric.name()] = MetricResult(
                    score=0.0,
                    explanation=f"Sub-metric evaluation failed: {str(result)}",
                    details={"error": str(result)}
                )
            else:
                sub_results[metric.name()] = result
        
        return sub_results

    async def _evaluate_single_sub_metric(
        self, 
        metric: BaseMetric, 
        rag_input: RAGInput
    ) -> MetricResult:
        """Evaluate a single sub-metric with error handling.
        
        Args:
            metric: The metric to evaluate
            rag_input: The RAG input to evaluate
            
        Returns:
            MetricResult from the sub-metric
        """
        try:
            return await metric.evaluate(rag_input)
        except Exception as e:
            raise RuntimeError(f"Sub-metric {metric.name()} failed: {str(e)}")

    def calculate_weighted_score(
        self, 
        sub_results: Dict[str, MetricResult]
    ) -> float:
        """Calculate weighted average score from sub-metric results.
        
        Args:
            sub_results: Dictionary of sub-metric results
            
        Returns:
            Weighted average score
        """
        if not sub_results:
            return 0.0
        
        if not self.weights:
            # Equal weights if no weights specified
            scores = [result.score for result in sub_results.values()]
            return sum(scores) / len(scores)
        
        # Weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, result in sub_results.items():
            weight = self.weights.get(metric_name, 0.0)
            total_score += result.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

    @abstractmethod
    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate the RAG input using the composite metric logic.
        
        This method must be implemented by concrete composite metrics.
        
        Args:
            rag_input: The RAG input to evaluate
            
        Returns:
            MetricResult containing the composite evaluation
        """
        pass 