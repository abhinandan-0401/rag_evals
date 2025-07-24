"""Answer correctness metric for evaluating factual accuracy and logical consistency."""

from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import re

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


class AnswerCorrectness(BaseMetric):
    """
    Evaluates the correctness of generated answers through multiple dimensions.
    
    This metric assesses correctness by evaluating:
    1. Factual accuracy - Are the facts stated correct?
    2. Context consistency - Does the answer align with provided contexts?
    3. Logical consistency - Is the reasoning sound and coherent?
    
    Score interpretation:
    - 1.0: Answer is completely correct across all dimensions
    - 0.5: Answer is partially correct with some issues
    - 0.0: Answer contains significant errors or inaccuracies
    """
    
    def __init__(
        self, 
        llm=None,
        evaluate_factual_accuracy: bool = True,
        evaluate_context_consistency: bool = True,
        evaluate_logical_consistency: bool = True,
        strictness_level: str = "moderate",
        **kwargs: Any
    ):
        """Initialize the Answer Correctness metric.
        
        Args:
            llm: LLM instance for evaluation
            evaluate_factual_accuracy: Whether to check factual accuracy
            evaluate_context_consistency: Whether to check consistency with contexts
            evaluate_logical_consistency: Whether to check logical coherence
            strictness_level: Evaluation strictness ("strict", "moderate", "lenient")
            **kwargs: Additional configuration parameters
        """
        super().__init__(llm=llm, **kwargs)
        self.evaluate_factual_accuracy = evaluate_factual_accuracy
        self.evaluate_context_consistency = evaluate_context_consistency
        self.evaluate_logical_consistency = evaluate_logical_consistency
        self.strictness_level = strictness_level
    
    def name(self) -> str:
        """Return the name of this metric."""
        return "answer_correctness"

    @property
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        return MetricType.GENERATION

    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate the correctness of the answer.
        
        Args:
            rag_input: RAG input containing query, answer, and contexts
            
        Returns:
            MetricResult with correctness score and detailed analysis
        """
        if not rag_input.answer or not rag_input.answer.strip():
            return MetricResult(
                score=0.0,
                reason="Empty answer provided",
                details={"error": "empty_answer"}
            )
        
        # For now, use simple accuracy evaluation to ensure it works
        return await self._simple_accuracy_evaluation(rag_input)

    async def _simple_accuracy_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform simple accuracy evaluation using factual accuracy check.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with accuracy assessment
        """
        try:
            # Get the accuracy prompt from YAML
            accuracy_prompt = get_prompt("answer_correctness", "accuracy_prompt")
            
            # Format contexts for the prompt
            contexts_str = ""
            if rag_input.retrieved_contexts is not None and len(rag_input.retrieved_contexts) > 0:
                contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
            
            # Format the prompt with variables
            prompt = accuracy_prompt.format(
                context=contexts_str,
                answer=rag_input.answer
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("answer_correctness", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None

            self.logger.info(f"Starting {self.name()} evaluation")

            response = await self.generate_llm_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=400
            )
            
            # Parse the response
            score, explanation = parse_llm_score(response.content)
            score = max(0.0, min(1.0, score))  # Ensure score is in valid range
            
            self.logger.info(f"Completed {self.name()} evaluation")
            
            return MetricResult(
                score=score,
                reason=explanation.strip(),
                details={
                    "evaluation_type": "simple_accuracy",
                    "evaluation_prompt": "factual_accuracy_assessment",
                    "contexts_count": len(rag_input.retrieved_contexts) if rag_input.retrieved_contexts is not None else 0,
                    "answer_length": len(rag_input.answer),
                    "query_length": len(rag_input.query),
                    "strictness_level": self.strictness_level
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate answer correctness: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _comprehensive_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform comprehensive evaluation with multiple dimensions.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with detailed correctness analysis
        """
        try:
            # Collect evaluation tasks
            evaluation_tasks = []
            
            if self.evaluate_factual_accuracy:
                evaluation_tasks.append(("factual_accuracy", self._evaluate_factual_accuracy(
                    rag_input.query, rag_input.answer, rag_input.retrieved_contexts
                )))
            
            if self.evaluate_context_consistency and rag_input.retrieved_contexts is not None and len(rag_input.retrieved_contexts) > 0:
                evaluation_tasks.append(("context_consistency", self._evaluate_context_consistency(
                    rag_input.query, rag_input.answer, rag_input.retrieved_contexts
                )))
            
            if self.evaluate_logical_consistency:
                evaluation_tasks.append(("logical_consistency", self._evaluate_logical_consistency(
                    rag_input.query, rag_input.answer
                )))
            
            if not evaluation_tasks:
                return MetricResult(
                    score=0.0,
                    reason="No evaluation dimensions enabled",
                    details={"error": "no_dimensions"}
                )
            
            # Run evaluations in parallel
            results = {}
            for dimension_name, task in evaluation_tasks:
                score, details = await task
                results[dimension_name] = {
                    "score": score,
                    "details": details
                }
            
            # Calculate overall correctness score (average of all dimensions)
            dimension_scores = [results[dim]["score"] for dim in results.keys()]
            overall_score = sum(dimension_scores) / len(dimension_scores)
            
            # Generate explanation
            passed_dimensions = [dim for dim, data in results.items() if data["score"] >= 0.7]
            
            if overall_score >= 0.8:
                reason = f"Answer demonstrates strong correctness across {len(passed_dimensions)}/{len(results)} dimensions"
            elif overall_score >= 0.5:
                reason = f"Answer shows moderate correctness with {len(passed_dimensions)}/{len(results)} dimensions performing well"
            else:
                reason = f"Answer has correctness issues with only {len(passed_dimensions)}/{len(results)} dimensions performing adequately"
            
            return MetricResult(
                score=overall_score,
                reason=reason,
                details={
                    "overall_score": overall_score,
                    "dimensions_evaluated": list(results.keys()),
                    "dimension_scores": {dim: data["score"] for dim, data in results.items()},
                    "dimension_details": results,
                    "evaluation_summary": {
                        "total_dimensions": len(results),
                        "passed_dimensions": len(passed_dimensions),
                        "average_score": overall_score
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed comprehensive {self.name()} evaluation: {str(e)}")
            # Fallback to simple evaluation
            return await self._simple_accuracy_evaluation(rag_input)

    async def _evaluate_factual_accuracy(
        self, 
        query: str, 
        answer: str, 
        contexts: Optional[List[str]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate the factual accuracy of the answer.
        
        Args:
            query: The original query
            answer: The generated answer
            contexts: Optional retrieved contexts for reference
            
        Returns:
            Tuple of (accuracy_score, details_dict)
        """
        try:
            # Format contexts
            contexts_str = ""
            if contexts is not None and len(contexts) > 0:
                contexts_str = format_contexts(contexts, numbered=True)
            
            # Get the accuracy prompt from YAML
            accuracy_prompt = get_prompt("answer_correctness", "accuracy_prompt")
            
            # Format the prompt with variables
            prompt = accuracy_prompt.format(
                context=contexts_str,
                answer=answer
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("answer_correctness", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None

            response = await self.generate_llm_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=400
            )
            
            score, explanation = parse_llm_score(response.content)
            score = max(0.0, min(1.0, score))
            
            return score, {
                "score": score,
                "explanation": explanation,
                "evaluation_prompt": "factual_accuracy_assessment",
                "contexts_used": bool(contexts)
            }
            
        except Exception as e:
            self.logger.error(f"Failed factual accuracy evaluation: {str(e)}")
            return 0.0, {
                "score": 0.0,
                "explanation": f"Failed to parse evaluation: {str(e)}",
                "error": "parsing_failed"
            }

    async def _evaluate_context_consistency(
        self, 
        query: str, 
        answer: str, 
        contexts: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate how consistent the answer is with provided contexts.
        
        Args:
            query: The original query
            answer: The generated answer
            contexts: The retrieved contexts
            
        Returns:
            Tuple of (consistency_score, details_dict)
        """
        try:
            # Get the consistency prompt from YAML
            consistency_prompt = get_prompt("answer_correctness", "consistency_prompt")
            
            # Format the prompt with variables
            prompt = consistency_prompt.format(answer=answer)

            # Get system prompt
            try:
                system_prompt_template = get_prompt("answer_correctness", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None

            response = await self.generate_llm_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=400
            )
            
            score, explanation = parse_llm_score(response.content)
            score = max(0.0, min(1.0, score))
            
            return score, {
                "score": score,
                "explanation": explanation,
                "evaluation_prompt": "context_consistency_assessment",
                "contexts_count": len(contexts)
            }
            
        except Exception as e:
            self.logger.error(f"Failed context consistency evaluation: {str(e)}")
            return 0.0, {
                "score": 0.0,
                "explanation": f"Failed to parse evaluation: {str(e)}",
                "error": "parsing_failed"
            }

    async def _evaluate_logical_consistency(
        self, 
        query: str, 
        answer: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate the logical consistency of the answer.
        
        Args:
            query: The original query
            answer: The generated answer
            
        Returns:
            Tuple of (logical_score, details_dict)
        """
        try:
            # Get the logic prompt from YAML
            logic_prompt = get_prompt("answer_correctness", "logic_prompt")
            
            # Format the prompt with variables
            prompt = logic_prompt.format(
                query=query,
                answer=answer
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("answer_correctness", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None

            response = await self.generate_llm_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=400
            )
            
            score, explanation = parse_llm_score(response.content)
            score = max(0.0, min(1.0, score))
            
            return score, {
                "score": score,
                "explanation": explanation,
                "evaluation_prompt": "logical_consistency_assessment"
            }
            
        except Exception as e:
            self.logger.error(f"Failed logical consistency evaluation: {str(e)}")
            return 0.0, {
                "score": 0.0,
                "explanation": f"Failed to parse evaluation: {str(e)}",
                "error": "parsing_failed"
            } 