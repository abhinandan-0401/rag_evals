"""Context precision metric for evaluating ranking quality."""

import re
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio

from ...core.base_metric import BaseMetric
from ...core.types import RAGInput, MetricResult, MetricType
from ...prompts import get_prompt, format_contexts

# Simple utility function to replace the deleted parse_llm_score
def parse_llm_score(response_text: str) -> Tuple[float, str]:
    """Parse LLM response to extract score and explanation."""
    score_patterns = [
        r'Score:\s*([0-9]*\.?[0-9]+)',
        r'([0-9]*\.?[0-9]+)',
    ]
    score = 0.0
    for pattern in score_patterns:
        match = re.search(pattern, response_text)
        if match:
            try:
                score = float(match.group(1))
                break
            except ValueError:
                continue
    return score, response_text


class ContextPrecision(BaseMetric):
    """Evaluates the precision and ranking quality of retrieved contexts.
    
    Context precision measures whether the most relevant contexts appear first
    in the ranking, optimizing for precision at the top ranks.
    
    Score interpretation:
    - 1.0: All contexts are relevant and perfectly ranked
    - 0.5: Some contexts are relevant but ranking could be improved
    - 0.0: Few contexts are relevant or ranking is poor
    """

    def name(self) -> str:
        """Return the name of this metric."""
        return "context_precision"

    @property
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        return MetricType.RETRIEVAL

    def _validate_metric_specific(self, rag_input: RAGInput) -> None:
        """Validate that required inputs are present for context precision evaluation."""
        if rag_input.retrieved_contexts is None or len(rag_input.retrieved_contexts) == 0:
            raise ValueError("Context precision metric requires retrieved contexts")

    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate context precision using modular YAML-based prompts.
        
        Args:
            rag_input: RAG input containing query and contexts to evaluate
            
        Returns:
            MetricResult with precision score and reasoning
        """
        try:
            # Validate inputs
            self._validate_metric_specific(rag_input)
            
            # Quick check - if no contexts, score is 0
            if rag_input.retrieved_contexts is None or len(rag_input.retrieved_contexts) == 0:
                return MetricResult(
                    score=0.0,
                    reason="No contexts provided for precision evaluation",
                    details={"contexts_count": 0}
                )

            self.logger.info(f"Starting {self.name()} evaluation")

            # For now, use simple ranking evaluation to ensure it works
            return await self._simple_precision_evaluation(rag_input)
            
        except Exception as e:
            self.logger.error(f"Failed {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate context precision: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _simple_precision_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform simple precision evaluation using ranking assessment.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with precision assessment
        """
        try:
            # Get the ranking evaluation prompt from YAML
            ranking_prompt = get_prompt("context_precision", "ranking_evaluation")
            
            # Format contexts for the prompt
            contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
            
            # Format the prompt with variables
            prompt = ranking_prompt.format(
                query=rag_input.query,
                ranked_contexts=contexts_str
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("context_precision", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None

            response = await self.generate_llm_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=500
            )
            
            # Parse the response
            score, explanation = parse_llm_score(response.content)
            score = max(0.0, min(1.0, score))  # Ensure score is in valid range
            
            self.logger.info(f"Completed {self.name()} evaluation")
            
            return MetricResult(
                score=score,
                reason=explanation.strip(),
                details={
                    "evaluation_type": "simple_ranking",
                    "evaluation_prompt": "ranking_evaluation",
                    "contexts_count": len(rag_input.retrieved_contexts),
                    "query_length": len(rag_input.query)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed simple {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate context precision: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _comprehensive_precision_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform comprehensive precision evaluation with single and multiple context handling.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with detailed precision analysis
        """
        try:
            # Special case: single context
            if len(rag_input.retrieved_contexts) == 1:
                # Evaluate single context relevance
                single_score = await self._evaluate_single_context_relevance(
                    rag_input.query, rag_input.retrieved_contexts[0]
                )
                return MetricResult(
                    score=single_score,
                    reason=f"Single context evaluated with relevance score: {single_score:.3f}",
                    details={
                        "contexts_count": 1,
                        "single_context_score": single_score,
                        "evaluation_method": "single_context"
                    }
                )

            # Multiple contexts - evaluate ranking precision
            return await self._evaluate_ranking_precision(rag_input)
            
        except Exception as e:
            self.logger.error(f"Failed comprehensive {self.name()} evaluation: {str(e)}")
            # Fallback to simple evaluation
            return await self._simple_precision_evaluation(rag_input)

    async def _evaluate_single_context_relevance(self, query: str, context: str) -> float:
        """Evaluate relevance of a single context.
        
        Args:
            query: The query string
            context: The context to evaluate
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            relevance_prompt = get_prompt("context_precision", "relevance_assessment")

            formatted_prompt = relevance_prompt.format(
                query=query,
                context=context
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("context_precision", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None

            response = await self.generate_llm_response(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=200
            )

            score, _ = parse_llm_score(response.content)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Failed single context relevance evaluation: {str(e)}")
            return 0.0

    async def _evaluate_ranking_precision(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate precision of context ranking.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with ranking precision assessment
        """
        try:
            ranking_prompt = get_prompt("context_precision", "ranking_evaluation")

            contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)

            formatted_prompt = ranking_prompt.format(
                query=rag_input.query,
                ranked_contexts=contexts_str
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("context_precision", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None

            response = await self.generate_llm_response(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=500
            )

            score, explanation = parse_llm_score(response.content)
            score = max(0.0, min(1.0, score))

            # Calculate additional metrics
            contexts_count = len(rag_input.retrieved_contexts)

            return MetricResult(
                score=score,
                reason=explanation.strip(),
                details={
                    "contexts_count": contexts_count,
                    "evaluation_method": "ranking_precision",
                    "prompt_used": "context_precision"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed ranking precision evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate ranking precision: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            ) 