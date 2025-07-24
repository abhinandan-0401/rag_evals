"""Core type definitions for RAG evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class MetricType(Enum):
    """Types of evaluation metrics."""
    RETRIEVAL = "retrieval"
    GENERATION = "generation"  
    COMPOSITE = "composite"


@dataclass
class RAGInput:
    """Input data for RAG evaluation.
    
    Attributes:
        query: The user's original question/query
        retrieved_contexts: List of retrieved text chunks/documents
        answer: The generated answer from the RAG system
        metadata: Optional additional metadata (timestamps, scores, etc.)
    """
    query: str
    retrieved_contexts: List[str]
    answer: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate input data."""
        if not self.query.strip():
            raise ValueError("Query cannot be empty")
        if not self.answer.strip():
            raise ValueError("Answer cannot be empty")
        # Handle both lists and numpy arrays safely
        if self.retrieved_contexts is None or len(self.retrieved_contexts) == 0:
            raise ValueError("Retrieved contexts cannot be empty")
        # Convert to list if it's a numpy array to avoid iteration issues
        if hasattr(self.retrieved_contexts, 'tolist'):
            self.retrieved_contexts = self.retrieved_contexts.tolist()
        # Validate individual contexts
        if any(not str(ctx).strip() for ctx in self.retrieved_contexts):
            raise ValueError("Retrieved contexts cannot contain empty strings")


@dataclass
class MetricResult:
    """Result of a metric evaluation.
    
    Attributes:
        score: Numerical score (typically 0.0 to 1.0)
        explanation: Human-readable explanation of the score
        details: Additional structured details about the evaluation
        metadata: Metadata about the evaluation process
    """
    score: float
    explanation: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __init__(self, score: float, explanation: Optional[str] = None, reason: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Initialize MetricResult with backward compatibility for 'reason' parameter."""
        self.score = score
        # Handle both 'explanation' and 'reason' for backward compatibility
        self.explanation = explanation or reason
        self.details = details
        self.metadata = metadata or {}
        
        # Call validation
        self.__post_init__()

    def __post_init__(self):
        """Validate metric result."""
        if not isinstance(self.score, (int, float)):
            raise ValueError("Score must be a number")
        if self.score < 0 or self.score > 1:
            raise ValueError("Score must be between 0 and 1")


@dataclass
class EvaluationResult:
    """Complete evaluation result containing multiple metric results.
    
    Attributes:
        overall_score: Optional aggregated score across all metrics
        metric_results: Dictionary mapping metric names to their results
        evaluation_time: Time taken for evaluation in seconds
        metadata: Additional metadata about the evaluation
    """
    overall_score: Optional[float] = None
    metric_results: Dict[str, MetricResult] = field(default_factory=dict)
    evaluation_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def get_metric_score(self, metric_name: str) -> Optional[float]:
        """Get score for a specific metric."""
        if metric_name in self.metric_results:
            return self.metric_results[metric_name].score
        return None

    def get_average_score(self) -> float:
        """Calculate average score across all metrics."""
        if not self.metric_results:
            return 0.0
        scores = [result.score for result in self.metric_results.values()]
        return sum(scores) / len(scores)


# Type aliases for better code readability
ScoreType = Union[int, float]
MetricName = str
ContextList = List[str] 