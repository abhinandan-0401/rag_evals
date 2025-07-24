"""LLM Judge metric for comprehensive quality evaluation."""

import re
from typing import Any, List, Dict, Optional, Union, Tuple
from enum import Enum

from .base_composite import BaseCompositeMetric
from ...core.types import RAGInput, MetricResult, MetricType
from ...core.exceptions import MetricError
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


class JudgmentCriteria(Enum):
    """Criteria for LLM-based judgment evaluation."""
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    HELPFULNESS = "helpfulness"
    CLARITY = "clarity"
    FACTUAL_CONSISTENCY = "factual_consistency"  # Add missing criterion


class LLMJudge(BaseCompositeMetric):
    """Uses an LLM as a judge to evaluate RAG responses across multiple criteria.
    
    This metric leverages the reasoning capabilities of large language models to
    provide comprehensive evaluation of RAG outputs across various quality dimensions.
    
    Score interpretation:
    - 1.0: Excellent across all evaluated criteria
    - 0.5: Satisfactory with some areas for improvement  
    - 0.0: Poor quality across multiple criteria
    """

    def __init__(
        self,
        llm=None,
        criteria: List[JudgmentCriteria] = None,
        use_scoring_rubric: bool = True,
        detailed_feedback: bool = True,
        **kwargs
    ):
        """Initialize the LLM Judge metric.
        
        Args:
            llm: LLM instance for evaluation
            criteria: List of criteria to evaluate (defaults to all)
            use_scoring_rubric: Whether to use structured scoring rubric
            detailed_feedback: Whether to provide detailed explanations
        """
        super().__init__(llm=llm, **kwargs)
        self.criteria = criteria or [
            JudgmentCriteria.RELEVANCE,
            JudgmentCriteria.ACCURACY,
            JudgmentCriteria.COMPLETENESS,
            JudgmentCriteria.COHERENCE
        ]
        self.use_scoring_rubric = use_scoring_rubric
        self.detailed_feedback = detailed_feedback

    def name(self) -> str:
        """Return the name of this metric."""
        return "llm_judge"

    @property
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        return MetricType.COMPOSITE

    def _validate_metric_specific(self, rag_input: RAGInput) -> None:
        """Validate that required inputs are present."""
        if not rag_input.answer or not rag_input.answer.strip():
            raise ValueError("LLM Judge metric requires a non-empty answer")

    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate using LLM judge with modular YAML-based prompts.
        
        Args:
            rag_input: RAG input containing query and answer to evaluate
            
        Returns:
            MetricResult with judge score and reasoning
        """
        # Validate inputs
        self._validate_metric_specific(rag_input)
        
        if self.use_scoring_rubric:
            return await self._structured_evaluation(rag_input)
        else:
            return await self._holistic_evaluation(rag_input)

    async def _structured_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform structured evaluation using scoring rubric."""
        
        # Get prompt template from YAML
        try:
            prompt_template = get_prompt("llm_judge", "template")
        except Exception as e:
            self.logger.warning(f"Could not load llm_judge template: {e}")
            # Fallback with explicit error logging
            try:
                prompt_template = get_prompt("llm_judge", "template", "composite")
            except Exception as e2:
                self.logger.error(f"Fallback prompt loading also failed: {e2}")
                raise MetricError(f"Could not load LLM judge prompt template: {e}") from e
        
        # Format contexts
        contexts_str = ""
        # Fix: Use explicit None and length check instead of boolean evaluation
        if rag_input.retrieved_contexts is not None and len(rag_input.retrieved_contexts) > 0:
            contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
        
        # Prepare prompt variables
        criteria_text = self._format_criteria_for_prompt()
        
        prompt_variables = {
            "query": rag_input.query,
            "answer": rag_input.answer,
            "criteria": criteria_text
        }
        
        # Add contexts if template expects it
        if "context" in prompt_template.input_variables and contexts_str:
            prompt_variables["context"] = contexts_str
        
        formatted_prompt = prompt_template.format(**prompt_variables)
        
        # Get LLM response
        try:
            response = await self.generate_llm_response(
                prompt=formatted_prompt,
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=600
            )
        except Exception as e:
            self.logger.error(f"LLM response generation failed for llm_judge: {e}")
            raise MetricError(f"LLM response generation failed: {e}") from e
        
        # Parse structured response
        try:
            scores, reasoning = self._parse_structured_response(response.content)
        except Exception as e:
            self.logger.error(f"Failed to parse LLM judge response: {e}")
            self.logger.debug(f"Raw LLM response: {response.content}")
            # Try fallback parsing
            score, reason = parse_llm_score(response.content)
            scores = {"overall": score}
            reasoning = reason
        
        # Calculate weighted average
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        
        return MetricResult(
            score=overall_score,
            reason=reasoning,
            details={
                "individual_scores": scores,
                "criteria_evaluated": [c.value for c in self.criteria],
                # Fix: Use explicit None and length check instead of boolean evaluation
                "contexts_count": len(rag_input.retrieved_contexts) if rag_input.retrieved_contexts is not None else 0,
                "evaluation_method": "structured_rubric"
            }
        )

    async def _holistic_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform holistic evaluation without structured rubric."""
        
        # Get holistic prompt from YAML
        try:
            prompt_template = get_prompt("llm_judge", "holistic_prompt")
        except Exception as e:
            self.logger.warning(f"Could not load llm_judge holistic_prompt, using main template: {e}")
            # Fallback to main template
            try:
                prompt_template = get_prompt("llm_judge", "template")
            except Exception as e2:
                self.logger.error(f"Could not load any llm_judge prompt template: {e2}")
                raise MetricError(f"Could not load LLM judge prompt template: {e}") from e
        
        # Format contexts
        contexts_str = ""
        # Fix: Use explicit None and length check instead of boolean evaluation
        if rag_input.retrieved_contexts is not None and len(rag_input.retrieved_contexts) > 0:
            contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
        
        # Prepare prompt variables
        prompt_variables = {
            "query": rag_input.query,
            "answer": rag_input.answer
        }
        
        if "context" in prompt_template.input_variables and contexts_str:
            prompt_variables["context"] = contexts_str
        
        formatted_prompt = prompt_template.format(**prompt_variables)
        
        try:
            response = await self.generate_llm_response(
                prompt=formatted_prompt,
                temperature=0.1,
                max_tokens=500
            )
        except Exception as e:
            self.logger.error(f"LLM response generation failed for llm_judge (holistic): {e}")
            raise MetricError(f"LLM response generation failed: {e}") from e
        
        try:
            score, reasoning = parse_llm_score(response.content)
        except Exception as e:
            self.logger.error(f"Failed to parse LLM judge holistic response: {e}")
            self.logger.debug(f"Raw LLM response: {response.content}")
            # Fallback to default score
            score, reasoning = 0.0, "Failed to parse LLM response"
        score = max(0.0, min(1.0, score))
        
        return MetricResult(
            score=score,
            reason=reasoning,
            details={
                "evaluation_method": "holistic",
                # Fix: Use explicit None and length check instead of boolean evaluation
                "contexts_count": len(rag_input.retrieved_contexts) if rag_input.retrieved_contexts is not None else 0
            }
        )

    async def _comparative_evaluation(self, rag_input: RAGInput, reference_answer: str) -> MetricResult:
        """Perform comparative evaluation against reference answer."""
        
        try:
            prompt_template = get_prompt("llm_judge", "comparative_prompt")
        except Exception:
            prompt_template = get_prompt("llm_judge", "template", "composite")
        
        # Format contexts
        contexts_str = ""
        # Fix: Use explicit None and length check instead of boolean evaluation
        if rag_input.retrieved_contexts is not None and len(rag_input.retrieved_contexts) > 0:
            contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
        
        prompt_variables = {
            "query": rag_input.query,
            "answer": rag_input.answer,
            "reference_answer": reference_answer
        }
        
        if "context" in prompt_template.input_variables and contexts_str:
            prompt_variables["context"] = contexts_str
        
        formatted_prompt = prompt_template.format(**prompt_variables)
        
        response = await self.generate_llm_response(
            prompt=formatted_prompt,
            temperature=0.1,
            max_tokens=600
        )
        
        score, reasoning = parse_llm_score(response.content)
        score = max(0.0, min(1.0, score))
        
        return MetricResult(
            score=score,
            reason=reasoning,
            details={
                "evaluation_method": "comparative",
                "reference_provided": True,
                "contexts_count": len(rag_input.retrieved_contexts) if rag_input.retrieved_contexts is not None else 0
            }
        )

    def _format_criteria_for_prompt(self) -> str:
        """Format evaluation criteria for inclusion in prompts."""
        criteria_descriptions = {
            JudgmentCriteria.RELEVANCE: "How well does the answer address the specific question asked?",
            JudgmentCriteria.ACCURACY: "How factually correct is the information provided?",
            JudgmentCriteria.COMPLETENESS: "How thoroughly does the answer cover the topic?",
            JudgmentCriteria.COHERENCE: "How well-structured and logically consistent is the answer?",
            JudgmentCriteria.FACTUAL_CONSISTENCY: "How consistent is the answer with provided contexts?",
            JudgmentCriteria.HELPFULNESS: "How useful is this answer for the user?"
        }
        
        criteria_list = []
        for i, criterion in enumerate(self.criteria, 1):
            description = criteria_descriptions.get(criterion, criterion.value)
            criteria_list.append(f"{i}. {criterion.value.title()}: {description}")
        
        return "\n".join(criteria_list)

    def _parse_structured_response(self, response_text: str) -> Tuple[Dict[str, float], str]:
        """Parse structured LLM response to extract individual scores."""
        scores = {}
        
        # Extract scores for each criterion
        for criterion in self.criteria:
            pattern = rf"{criterion.value}[:\s]*([0-9]*\.?[0-9]+)"
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    scores[criterion.value] = max(0.0, min(1.0, score))
                except ValueError:
                    scores[criterion.value] = 0.0
            else:
                scores[criterion.value] = 0.0
        
        # Extract overall reasoning
        reasoning_patterns = [
            r"reasoning[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)",
            r"explanation[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)",
            r"overall[:\s]*(.+?)(?=\n\n|\n[A-Z]|$)"
        ]
        
        reasoning = response_text.strip()
        for pattern in reasoning_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                break
        
        return scores, reasoning

    @property
    def requires_contexts(self) -> bool:
        """LLM Judge can work with or without contexts."""
        return False 