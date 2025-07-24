"""Context Relevance metric for evaluating retrieved context quality."""

import re
from typing import Any, List, Optional, Tuple

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


class ContextRelevance(BaseMetric):
    """Evaluates relevance of retrieved contexts to the query.
    
    This metric assesses whether the retrieved contexts contain information
    that is relevant to answering the user's query.
    
    Score interpretation:
    - 1.0: All contexts are highly relevant to the query
    - 0.5: Some contexts are relevant, others are not
    - 0.0: Contexts are not relevant to the query
    """

    def name(self) -> str:
        """Return the name of this metric."""
        return "context_relevance"

    @property
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        return MetricType.RETRIEVAL

    def _validate_metric_specific(self, rag_input: RAGInput) -> None:
        """Validate that required inputs are present for context relevance evaluation."""
        if rag_input.retrieved_contexts is None or len(rag_input.retrieved_contexts) == 0:
            raise ValueError("Context relevance metric requires retrieved contexts")

    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate context relevance using modular YAML-based prompts.
        
        Args:
            rag_input: RAG input containing query and contexts to evaluate
            
        Returns:
            MetricResult with relevance score and reasoning
        """
        try:
            # Validate inputs
            self._validate_metric_specific(rag_input)
            
            # Quick check - if no contexts, score is 0
            if rag_input.retrieved_contexts is None or len(rag_input.retrieved_contexts) == 0:
                return MetricResult(
                    score=0.0,
                    reason="No contexts provided for relevance evaluation",
                    details={"contexts_count": 0}
                )

            self.logger.info(f"Starting {self.name()} evaluation")

            # Use simple relevance evaluation approach
            return await self._simple_relevance_evaluation(rag_input)
            
        except Exception as e:
            self.logger.error(f"Failed {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate context relevance: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _simple_relevance_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform simple relevance evaluation using the main prompt template.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with relevance assessment
        """
        try:
            # Get the main relevance evaluation prompt from YAML
            relevance_prompt = get_prompt("context_relevance", "template")
            
            # Format contexts for the prompt
            contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
            
            # Format the prompt with variables
            prompt = relevance_prompt.format(
                query=rag_input.query,
                contexts=contexts_str
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("context_relevance", "system_prompt")
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
                    "evaluation_type": "simple_relevance",
                    "evaluation_prompt": "template",
                    "contexts_count": len(rag_input.retrieved_contexts),
                    "query_length": len(rag_input.query)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed simple {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate context relevance: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _detailed_relevance_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform detailed relevance evaluation with per-context analysis.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with detailed relevance analysis
        """
        try:
            # Get the detailed analysis prompt from YAML
            detailed_prompt = get_prompt("context_relevance", "detailed_analysis")
            
            # Format contexts for the prompt
            contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
            
            # Format the prompt with variables
            prompt = detailed_prompt.format(
                query=rag_input.query,
                contexts=contexts_str
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("context_relevance", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None

            response = await self.generate_llm_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=700
            )
            
            # Parse the response
            score, explanation = parse_llm_score(response.content)
            score = max(0.0, min(1.0, score))  # Ensure score is in valid range
            
            return MetricResult(
                score=score,
                reason=explanation.strip(),
                details={
                    "evaluation_type": "detailed_analysis",
                    "evaluation_prompt": "detailed_analysis",
                    "contexts_count": len(rag_input.retrieved_contexts),
                    "query_length": len(rag_input.query)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed detailed {self.name()} evaluation: {str(e)}")
            # Fallback to simple evaluation
            return await self._simple_relevance_evaluation(rag_input)

    async def _binary_relevance_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform binary relevance evaluation for each context individually.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with binary relevance assessment
        """
        try:
            # Get the binary relevance prompt from YAML
            binary_prompt = get_prompt("context_relevance", "binary_relevance")
            
            relevant_contexts = 0
            total_contexts = len(rag_input.retrieved_contexts)
            context_evaluations = []
            
            # Get system prompt
            try:
                system_prompt_template = get_prompt("context_relevance", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None

            # Evaluate each context individually
            for i, context in enumerate(rag_input.retrieved_contexts):
                prompt = binary_prompt.format(
                    query=rag_input.query,
                    context=context
                )

                response = await self.generate_llm_response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=300
                )
                
                # Check for relevance indicators
                is_relevant = any(keyword in response.content.lower() 
                                for keyword in ['relevant', 'yes', 'useful', 'helpful'])
                
                if is_relevant:
                    relevant_contexts += 1
                
                context_evaluations.append({
                    "context_index": i + 1,
                    "is_relevant": is_relevant,
                    "evaluation": response.content.strip()
                })
            
            # Calculate overall relevance score
            relevance_score = relevant_contexts / total_contexts if total_contexts > 0 else 0.0
            
            # Generate reasoning
            reason = f"Binary relevance analysis: {relevant_contexts}/{total_contexts} contexts are relevant"
            
            return MetricResult(
                score=relevance_score,
                reason=reason,
                details={
                    "evaluation_type": "binary_relevance",
                    "evaluation_prompt": "binary_relevance",
                    "contexts_count": total_contexts,
                    "relevant_contexts": relevant_contexts,
                    "context_evaluations": context_evaluations
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed binary {self.name()} evaluation: {str(e)}")
            # Fallback to simple evaluation
            return await self._simple_relevance_evaluation(rag_input) 