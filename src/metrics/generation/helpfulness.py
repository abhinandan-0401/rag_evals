"""Helpfulness metric for evaluating RAG system responses."""

from typing import Optional, Any, List, Dict, Tuple
import asyncio
import re

from ...core.base_metric import BaseMetric
from ...core.types import RAGInput, MetricResult, MetricType
from ...prompts import get_prompt, format_contexts

# Simple utility function to replace the deleted parse_llm_score
def parse_llm_score(response_text: str) -> Tuple[float, str]:
    """Parse LLM response to extract score and explanation.
    
    Args:
        response_text: Raw LLM response text
        
    Returns:
        Tuple of (score, explanation)
    """
    # Try to find score patterns like "Score: 0.8" or "0.8" 
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


class Helpfulness(BaseMetric):
    """
    Evaluates how helpful and useful the generated answer is from a user perspective.
    
    This metric focuses on practical utility and user experience by assessing:
    1. Actionability - Can the user act on the information provided?
    2. Clarity - Is the answer easy to understand and follow?
    3. Practical value - Does the answer solve the user's problem?
    4. Usability - Is the information presented in a user-friendly way?
    
    Score interpretation:
    - 1.0: Extremely helpful - directly solves user's problem with clear guidance
    - 0.5: Moderately helpful - provides useful information but may lack clarity/actionability
    - 0.0: Not helpful - fails to provide practical value to the user
    """
    
    def __init__(
        self, 
        llm: Any = None,
        assess_actionability: bool = True,
        check_clarity: bool = True,
        evaluate_practical_value: bool = True,
        **kwargs: Any
    ):
        """Initialize the Helpfulness metric.
        
        Args:
            llm: The LLM instance to use for evaluation
            assess_actionability: Whether to evaluate if answer provides actionable guidance
            check_clarity: Whether to assess clarity and understandability
            evaluate_practical_value: Whether to evaluate practical problem-solving value
            **kwargs: Additional configuration parameters
        """
        super().__init__(llm=llm, **kwargs)
        self.assess_actionability = assess_actionability
        self.check_clarity = check_clarity
        self.evaluate_practical_value = evaluate_practical_value
    
    def name(self) -> str:
        """Return the name of this metric."""
        return "helpfulness"

    @property
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        return MetricType.GENERATION

    def _validate_metric_specific(self, rag_input: RAGInput) -> None:
        """Validate that required inputs are present for helpfulness evaluation."""
        if not rag_input.answer or not rag_input.answer.strip():
            raise ValueError("Helpfulness metric requires a non-empty answer")

    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """
        Evaluate the helpfulness of the generated answer.
        
        Args:
            rag_input: The RAG input containing query, contexts, and answer
            
        Returns:
            MetricResult with helpfulness score and detailed breakdown
        """
        try:
            # Validate inputs
            self._validate_metric_specific(rag_input)
            
            # Check if answer is empty
            if not rag_input.answer.strip():
                return MetricResult(
                    score=0.0,
                    reason="Empty answer provided - cannot assess helpfulness",
                    details={"error": "empty_answer"}
                )

            self.logger.info(f"Starting {self.name()} evaluation")

            # Use simple helpfulness evaluation for reliability
            return await self._simple_helpfulness_evaluation(rag_input)
            
        except Exception as e:
            self.logger.error(f"Failed {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate helpfulness: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _simple_helpfulness_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Simple helpfulness evaluation using the main prompt template.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with helpfulness assessment
        """
        try:
            # Get helpfulness evaluation prompt from YAML
            helpfulness_prompt = get_prompt("helpfulness", "template")
            
            # Format the prompt with variables
            prompt = helpfulness_prompt.format(
                query=rag_input.query,
                answer=rag_input.answer
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("helpfulness", "system_prompt")
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
                    "evaluation_type": "simple_helpfulness",
                    "evaluation_prompt": "template",
                    "query_length": len(rag_input.query),
                    "answer_length": len(rag_input.answer)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed simple {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate helpfulness: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _practical_utility_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Practical utility evaluation with context information.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with practical utility assessment
        """
        try:
            # Get practical utility evaluation prompt from YAML
            utility_prompt = get_prompt("helpfulness", "practical_utility")
            
            # Format contexts if available
            contexts_str = ""
            if rag_input.retrieved_contexts is not None and len(rag_input.retrieved_contexts) > 0:
                contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
            
            # Format the prompt with variables
            prompt = utility_prompt.format(
                query=rag_input.query,
                answer=rag_input.answer,
                context=contexts_str
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("helpfulness", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None

            response = await self.generate_llm_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=600
            )
            
            # Parse the response
            score, explanation = parse_llm_score(response.content)
            score = max(0.0, min(1.0, score))  # Ensure score is in valid range
            
            return MetricResult(
                score=score,
                reason=explanation.strip(),
                details={
                    "evaluation_type": "practical_utility",
                    "evaluation_prompt": "practical_utility",
                    "query_length": len(rag_input.query),
                    "answer_length": len(rag_input.answer),
                    "contexts_count": len(rag_input.retrieved_contexts) if rag_input.retrieved_contexts is not None else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed practical utility {self.name()} evaluation: {str(e)}")
            # Fallback to simple evaluation
            return await self._simple_helpfulness_evaluation(rag_input)

    async def _user_experience_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """User experience evaluation focusing on answer quality.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with user experience assessment
        """
        try:
            # Get user experience evaluation prompt from YAML
            ux_prompt = get_prompt("helpfulness", "user_experience")
            
            # Format the prompt with variables
            prompt = ux_prompt.format(
                answer=rag_input.answer
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("helpfulness", "system_prompt")
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
            
            return MetricResult(
                score=score,
                reason=explanation.strip(),
                details={
                    "evaluation_type": "user_experience",
                    "evaluation_prompt": "user_experience",
                    "answer_length": len(rag_input.answer),
                    "answer_word_count": len(rag_input.answer.split())
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed user experience {self.name()} evaluation: {str(e)}")
            # Fallback to simple evaluation
            return await self._simple_helpfulness_evaluation(rag_input)

    @property
    def requires_llm(self) -> bool:
        """This metric requires an LLM for helpfulness evaluation."""
        return True

    @property
    def requires_contexts(self) -> bool:
        """Helpfulness evaluation doesn't require contexts."""
        return False

    @property
    def requires_answer(self) -> bool:
        """Helpfulness requires a generated answer."""
        return True 