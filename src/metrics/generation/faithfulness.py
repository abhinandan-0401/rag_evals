"""Faithfulness metric for evaluating answer grounding in contexts."""

import re
from typing import List, Optional, Dict, Any

from ...core.base_metric import BaseMetric
from ...core.types import RAGInput, MetricResult, MetricType
from ...prompts import get_prompt, format_contexts


class Faithfulness(BaseMetric):
    """Evaluates whether the generated answer is faithful to the retrieved contexts.
    
    Faithfulness measures how well the answer is grounded in the provided contexts by:
    1. Checking if all claims in the answer are supported by the contexts
    2. Detecting contradictions between answer and contexts
    3. Identifying unsupported or hallucinated information
    
    Score interpretation:
    - 1.0: All claims are fully supported by contexts
    - 0.5: Some claims are supported, others are not
    - 0.0: Most/all claims are unsupported or contradictory
    
    Uses YAML-based prompts for modularity and easy customization.
    """

    def __init__(self, llm=None, detailed_evaluation: bool = False, **kwargs):
        """Initialize faithfulness metric.
        
        Args:
            llm: LangChain LLM instance
            detailed_evaluation: Whether to use detailed claim-by-claim evaluation
            **kwargs: Additional configuration
        """
        super().__init__(llm=llm, **kwargs)
        self.detailed_evaluation = detailed_evaluation

    def name(self) -> str:
        """Return the name of this metric."""
        return "faithfulness"

    @property
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        return MetricType.GENERATION

    def _validate_metric_specific(self, rag_input: RAGInput) -> None:
        """Validate that required inputs are present for faithfulness evaluation."""
        if rag_input.retrieved_contexts is None or len(rag_input.retrieved_contexts) == 0:
            raise ValueError("Faithfulness metric requires retrieved contexts")
        if not rag_input.answer or not rag_input.answer.strip():
            raise ValueError("Faithfulness metric requires a non-empty answer")

    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate faithfulness using modular YAML-based prompts.
        
        Args:
            rag_input: RAG input containing query, answer, and contexts
            
        Returns:
            MetricResult with faithfulness score and detailed reasoning
        """
        try:
            # Validate inputs
            self._validate_metric_specific(rag_input)
            
            self.logger.info(f"Starting {self.name()} evaluation")

            # Use simple faithfulness evaluation for reliability
            return await self._simple_faithfulness_evaluation(rag_input)
            
        except Exception as e:
            self.logger.error(f"Failed {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate faithfulness: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _simple_faithfulness_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Simple faithfulness evaluation using a single prompt.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with faithfulness score
        """
        try:
            # Get faithfulness evaluation prompt from YAML
            faithfulness_prompt = get_prompt("faithfulness", "template")
            
            # Format contexts for the prompt  
            contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
            
            # Format the prompt with variables (note: YAML uses 'context' singular)
            prompt = faithfulness_prompt.format(
                context=contexts_str,  # Note: using 'context' as per YAML
                answer=rag_input.answer
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("faithfulness", "system_prompt")
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
                    "evaluation_type": "simple_faithfulness",
                    "evaluation_prompt": "template",
                    "contexts_count": len(rag_input.retrieved_contexts),
                    "answer_length": len(rag_input.answer)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed simple {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate faithfulness: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _detailed_faithfulness_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Detailed faithfulness evaluation with claim-by-claim analysis.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with detailed faithfulness analysis
        """
        try:
            # Step 1: Extract claims from the answer
            claims = await self._extract_claims(rag_input.answer)
            
            if not claims:
                # Fallback to simple evaluation if claim extraction fails
                return await self._simple_faithfulness_evaluation(rag_input)
            
            # Step 2: Verify each claim against contexts
            claim_results = []
            contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
            
            for claim in claims:
                support_score = await self._verify_claim_support(claim, contexts_str)
                claim_results.append({
                    "claim": claim,
                    "support_score": support_score
                })
            
            # Step 3: Calculate overall faithfulness score
            support_scores = [result["support_score"] for result in claim_results]
            overall_score = sum(support_scores) / len(support_scores) if support_scores else 0.0
            
            # Generate detailed reasoning
            supported_claims = len([s for s in support_scores if s >= 0.7])
            partially_supported = len([s for s in support_scores if 0.3 <= s < 0.7])
            unsupported_claims = len([s for s in support_scores if s < 0.3])
            
            reasoning = f"Detailed claim analysis: {supported_claims} fully supported, {partially_supported} partially supported, {unsupported_claims} unsupported out of {len(claims)} total claims."
            
            return MetricResult(
                score=overall_score,
                reason=reasoning,
                details={
                    "evaluation_method": "detailed_claims",
                    "claims_analysis": claim_results,
                    "total_claims": len(claims),
                    "supported_claims": supported_claims,
                    "partially_supported_claims": partially_supported,
                    "unsupported_claims": unsupported_claims,
                    "contexts_count": len(rag_input.retrieved_contexts)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed detailed {self.name()} evaluation: {str(e)}")
            # Fallback to simple evaluation
            return await self._simple_faithfulness_evaluation(rag_input)

    async def _extract_claims(self, answer: str) -> List[str]:
        """Extract individual claims from the answer.
        
        Args:
            answer: The generated answer
            
        Returns:
            List of extracted claims
        """
        try:
            # Get claim extraction prompt from YAML
            extraction_prompt = get_prompt("faithfulness", "claim_extraction")

            formatted_prompt = extraction_prompt.format(answer=answer)
            
            # Get system prompt
            try:
                system_prompt_template = get_prompt("faithfulness", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None
            
            response = await self.generate_llm_response(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=300
            )
            
            # Parse claims from response
            claims = []
            for line in response.content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                    # Remove numbering/bullets and extract claim
                    claim = re.sub(r'^[-•\d.\s]+', '', line).strip()
                    if claim and len(claim) > 10:  # Filter out very short claims
                        claims.append(claim)
            
            return claims[:10]  # Limit to 10 claims for efficiency
            
        except Exception as e:
            self.logger.error(f"Failed to extract claims: {str(e)}")
            # Use simple rule-based extraction as fallback
            return self._simple_claim_extraction(answer)

    def _simple_claim_extraction(self, answer: str) -> List[str]:
        """Simple rule-based claim extraction as fallback.
        
        Args:
            answer: The generated answer
            
        Returns:
            List of sentences as claims
        """
        try:
            # Split by sentences and filter
            sentences = re.split(r'[.!?]+', answer)
            claims = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and not sentence.lower().startswith(('however', 'but', 'although')):
                    claims.append(sentence)
            
            return claims[:5]  # Limit to 5 claims
            
        except Exception as e:
            self.logger.error(f"Failed simple claim extraction: {str(e)}")
            return []

    async def _verify_claim_support(self, claim: str, contexts: str) -> float:
        """Verify if a claim is supported by the contexts.
        
        Args:
            claim: Individual claim to verify
            contexts: Formatted context string
            
        Returns:
            Support score between 0.0 and 1.0
        """
        try:
            # Get claim verification prompt from YAML
            verification_prompt = get_prompt("faithfulness", "claim_verification")

            formatted_prompt = verification_prompt.format(
                context=contexts,  # Note: using 'context' as per YAML
                claim=claim
            )
            
            # Get system prompt
            try:
                system_prompt_template = get_prompt("faithfulness", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None
            
            response = await self.generate_llm_response(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=200
            )
            
            return self._parse_score_from_response(response.content)
            
        except Exception as e:
            self.logger.error(f"Failed to verify claim support: {str(e)}")
            return 0.0

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
        return 0.0

    @property
    def requires_contexts(self) -> bool:
        """Faithfulness requires retrieved contexts."""
        return True

    @property
    def requires_answer(self) -> bool:
        """Faithfulness requires a generated answer."""
        return True 