"""Coherence metric for evaluating answer structure and flow."""

import re
from typing import Optional

from ...core.base_metric import BaseMetric
from ...core.types import RAGInput, MetricResult, MetricType
from ...prompts import get_prompt


class Coherence(BaseMetric):
    """Evaluates how well-structured and logically organized the generated answer is.
    
    Coherence measures whether the answer:
    1. Has a logical flow and structure
    2. Uses appropriate transitions between ideas
    3. Maintains consistency throughout the response
    4. Is easy to follow and understand
    
    Score interpretation:
    - 1.0: Highly coherent, well-structured answer
    - 0.5: Somewhat coherent with minor structural issues
    - 0.0: Incoherent, poorly structured answer
    
    Uses YAML-based prompts for modularity and easy customization.
    """

    def name(self) -> str:
        """Return the name of this metric."""
        return "coherence"

    @property
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        return MetricType.GENERATION

    def _validate_metric_specific(self, rag_input: RAGInput) -> None:
        """Validate that required inputs are present for coherence evaluation."""
        if not rag_input.answer or not rag_input.answer.strip():
            raise ValueError("Coherence metric requires a non-empty answer")

    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate answer coherence using modular YAML-based prompts.
        
        Args:
            rag_input: RAG input containing the answer to evaluate
            
        Returns:
            MetricResult with coherence score and reasoning
        """
        try:
            # Validate inputs
            self._validate_metric_specific(rag_input)
            
            self.logger.info(f"Starting {self.name()} evaluation")

            # Use simple coherence evaluation for reliability
            return await self._simple_coherence_evaluation(rag_input)
            
        except Exception as e:
            self.logger.error(f"Failed {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate coherence: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _simple_coherence_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Simple coherence evaluation using the main prompt template.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with coherence assessment
        """
        try:
            # Get coherence evaluation prompt from YAML
            coherence_prompt = get_prompt("coherence", "template")
            
            # Format the prompt with variables
            prompt = coherence_prompt.format(
                answer=rag_input.answer
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("coherence", "system_prompt")
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
            score = self._parse_score_from_response(response.content)
            
            self.logger.info(f"Completed {self.name()} evaluation")
            
            return MetricResult(
                score=score,
                reason=response.content.strip(),
                details={
                    "evaluation_type": "simple_coherence",
                    "evaluation_prompt": "template",
                    "answer_length": len(rag_input.answer),
                    "answer_word_count": len(rag_input.answer.split())
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed simple {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate coherence: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _detailed_coherence_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Detailed coherence evaluation with query context.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with detailed coherence assessment
        """
        try:
            # Get detailed evaluation prompt from YAML
            evaluation_prompt = get_prompt("coherence", "evaluation_prompt")
            
            # Format the prompt with variables
            prompt = evaluation_prompt.format(
                query=rag_input.query,
                answer=rag_input.answer
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("coherence", "system_prompt")
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
            score = self._parse_score_from_response(response.content)
            
            return MetricResult(
                score=score,
                reason=response.content.strip(),
                details={
                    "evaluation_type": "detailed_coherence",
                    "evaluation_prompt": "evaluation_prompt",
                    "answer_length": len(rag_input.answer),
                    "answer_word_count": len(rag_input.answer.split()),
                    "query_length": len(rag_input.query)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed detailed {self.name()} evaluation: {str(e)}")
            # Fallback to simple evaluation
            return await self._simple_coherence_evaluation(rag_input)
    
    async def _fallback_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Fallback evaluation when YAML prompts are not available.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with coherence assessment
        """
        try:
            # Default coherence evaluation prompt
            fallback_prompt = f"""You are an expert evaluator. Please evaluate how coherent and well-structured the given answer is.

Answer: {rag_input.answer}

Rate the coherence on a scale of 0.0 to 1.0, where:
- 1.0: The answer is very well-structured and flows logically
- 0.5: The answer has some structure but could be improved
- 0.0: The answer lacks structure and is difficult to follow

Score: [Your score from 0.0 to 1.0]
Reasoning: [Explain your reasoning]"""
            
            response = await self.generate_llm_response(
                prompt=fallback_prompt,
                temperature=0.0,
                max_tokens=300
            )
            
            score = self._parse_score_from_response(response.content)
            
            return MetricResult(
                score=score,
                reason=response.content.strip(),
                details={
                    "evaluation_type": "fallback_hardcoded",
                    "answer_length": len(rag_input.answer),
                    "evaluation_method": "fallback"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed fallback {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate coherence: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
    
    def _parse_score_from_response(self, response_content: str) -> float:
        """Parse score from LLM response.
        
        Args:
            response_content: Raw LLM response
            
        Returns:
            Parsed score between 0.0 and 1.0
        """
        # Try multiple score extraction patterns
        score_patterns = [
            r"Score:\s*([0-9]*\.?[0-9]+)",
            r"score:\s*([0-9]*\.?[0-9]+)",
            r"([0-9]*\.?[0-9]+)/1\.0",
            r"([0-9]*\.?[0-9]+)\s*out\s*of\s*1",
            r"([01]\.?[0-9]*)"
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response_content, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except (ValueError, IndexError):
                    continue
        
        # Default score if parsing fails
        return 0.5

    @property
    def requires_contexts(self) -> bool:
        """Coherence evaluation doesn't require contexts."""
        return False

    @property
    def requires_answer(self) -> bool:
        """Coherence requires a generated answer."""
        return True 