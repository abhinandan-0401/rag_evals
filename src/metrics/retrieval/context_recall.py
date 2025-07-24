"""Context Recall metric for evaluating context completeness."""

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


class ContextRecall(BaseMetric):
    """Evaluates how well the retrieved contexts cover the necessary information.
    
    Context recall measures whether the retrieved contexts contain all the information
    needed to properly answer the query, based on the ground truth answer.
    
    Score interpretation:
    - 1.0: Contexts contain all necessary information for a complete answer
    - 0.5: Contexts contain some but not all necessary information  
    - 0.0: Contexts contain little to no necessary information
    """

    def name(self) -> str:
        """Return the name of this metric."""
        return "context_recall"

    @property
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        return MetricType.RETRIEVAL

    def _validate_metric_specific(self, rag_input: RAGInput) -> None:
        """Validate that required inputs are present for context recall evaluation."""
        if rag_input.retrieved_contexts is None or len(rag_input.retrieved_contexts) == 0:
            raise ValueError("Context recall metric requires retrieved contexts")
        if not rag_input.answer or not rag_input.answer.strip():
            raise ValueError("Context recall metric requires a non-empty answer")

    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate context recall using modular YAML-based prompts.
        
        Args:
            rag_input: RAG input containing query, answer, and contexts to evaluate
            
        Returns:
            MetricResult with recall score and reasoning
        """
        try:
            # Validate inputs
            self._validate_metric_specific(rag_input)
            
            # Quick checks for edge cases
            if rag_input.retrieved_contexts is None or len(rag_input.retrieved_contexts) == 0:
                return MetricResult(
                    score=0.0,
                    reason="No contexts provided for recall evaluation",
                    details={"contexts_count": 0}
                )

            if not rag_input.answer or not rag_input.answer.strip():
                return MetricResult(
                    score=0.0,
                    reason="No answer provided - cannot extract claims for recall evaluation",
                    details={"answer_length": 0}
                )

            self.logger.info(f"Starting {self.name()} evaluation")

            # Use simple recall evaluation approach
            return await self._simple_recall_evaluation(rag_input)
            
        except Exception as e:
            self.logger.error(f"Failed {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate context recall: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _simple_recall_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform simple recall evaluation using the main prompt template.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with recall assessment
        """
        try:
            # Get the main recall evaluation prompt from YAML
            recall_prompt = get_prompt("context_recall", "template")
            
            # Format contexts for the prompt
            contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
            
            # Format the prompt with variables
            prompt = recall_prompt.format(
                query=rag_input.query,
                answer=rag_input.answer,
                contexts=contexts_str
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("context_recall", "system_prompt")
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
                    "evaluation_type": "simple_recall",
                    "evaluation_prompt": "template",
                    "contexts_count": len(rag_input.retrieved_contexts),
                    "query_length": len(rag_input.query),
                    "answer_length": len(rag_input.answer)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed simple {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate context recall: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _comprehensive_recall_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform comprehensive recall evaluation with claim extraction and verification.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with detailed recall analysis
        """
        try:
            # Extract claims from the answer
            claims = await self._extract_claims_from_answer(rag_input)
            
            if not claims:
                return MetricResult(
                    score=0.0,
                    reason="No extractable claims found in answer for recall evaluation",
                    details={"claims_extracted": 0}
                )

            # Verify claims against contexts
            supported_claims, total_claims = await self._verify_claims_in_contexts(
                claims, rag_input.retrieved_contexts
            )

            # Calculate recall score
            recall_score = supported_claims / total_claims if total_claims > 0 else 0.0

            # Generate reasoning
            if recall_score >= 0.8:
                reason = f"Excellent recall: {supported_claims}/{total_claims} claims supported by contexts"
            elif recall_score >= 0.6:
                reason = f"Good recall: {supported_claims}/{total_claims} claims supported by contexts"
            elif recall_score >= 0.4:
                reason = f"Moderate recall: {supported_claims}/{total_claims} claims supported by contexts"
            else:
                reason = f"Poor recall: {supported_claims}/{total_claims} claims supported by contexts"

            return MetricResult(
                score=recall_score,
                reason=reason,
                details={
                    "claims_extracted": total_claims,
                    "claims_supported": supported_claims,
                    "contexts_count": len(rag_input.retrieved_contexts),
                    "extracted_claims": claims,
                    "evaluation_method": "comprehensive_claim_analysis"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed comprehensive {self.name()} evaluation: {str(e)}")
            # Fallback to simple evaluation
            return await self._simple_recall_evaluation(rag_input)

    async def _extract_claims_from_answer(self, rag_input: RAGInput) -> List[str]:
        """Extract factual claims from the answer.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            List of extracted claims
        """
        try:
            extraction_prompt = get_prompt("context_recall", "extraction_prompt")

            formatted_prompt = extraction_prompt.format(
                query=rag_input.query,
                ground_truth=rag_input.answer  # Note: using ground_truth as per YAML
            )
            
            # Get system prompt
            try:
                system_prompt_template = get_prompt("context_recall", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None
            
            response = await self.generate_llm_response(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=500
            )
            
            # Simple parsing - split by newlines and filter
            claims = [line.strip() for line in response.content.split('\n') 
                     if line.strip() and not line.strip().startswith('#')]
            
            return claims
            
        except Exception as e:
            self.logger.error(f"Failed to extract claims: {str(e)}")
            return []

    async def _verify_claims_in_contexts(self, claims: List[str], contexts: List[str]) -> Tuple[int, int]:
        """Verify which claims are supported by the contexts.
        
        Args:
            claims: List of claims to verify
            contexts: List of context strings
            
        Returns:
            Tuple of (supported_claims_count, total_claims_count)
        """
        try:
            verification_prompt = get_prompt("context_recall", "verification_prompt")

            contexts_str = format_contexts(contexts, numbered=True)
            
            supported_claims = 0
            
            for claim in claims:
                formatted_prompt = verification_prompt.format(
                    element=claim,  # Note: using element as per YAML
                    contexts=contexts_str
                )
                
                # Get system prompt
                try:
                    system_prompt_template = get_prompt("context_recall", "system_prompt")
                    system_prompt = system_prompt_template.format()
                except:
                    system_prompt = None
                
                response = await self.generate_llm_response(
                    prompt=formatted_prompt,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=200
                )
                
                # Check for presence indicators
                if any(keyword in response.content.lower() for keyword in ['present', 'supported', 'yes', 'true']):
                    supported_claims += 1

            return supported_claims, len(claims)
            
        except Exception as e:
            self.logger.error(f"Failed to verify claims: {str(e)}")
            return 0, len(claims) 