"""RAG certainty metric for evaluating confidence and uncertainty in responses."""

from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import re

from .base_composite import BaseCompositeMetric
from ...core.types import RAGInput, MetricResult, MetricType
from ...prompts import get_prompt

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


class RAGCertainty(BaseCompositeMetric):
    """
    Composite metric that evaluates the certainty/confidence of a RAG system's response.
    
    This metric is crucial for production deployment as it combines:
    1. Generation Confidence - How confident the model is in its answer
    2. Retrieval Quality - How well the retrieved contexts support the answer
    3. Answer Consistency - How consistent the answer is internally
    4. Knowledge Alignment - How well the answer aligns with retrieved knowledge
    
    Score interpretation:
    - 1.0: Very high certainty - suitable for automated deployment
    - 0.7-0.9: High certainty - suitable with light human oversight
    - 0.4-0.7: Moderate certainty - requires human review
    - 0.0-0.4: Low certainty - requires significant human intervention
    """
    
    def __init__(
        self, 
        llm: Any = None,
        assess_generation_confidence: bool = True,
        evaluate_retrieval_quality: bool = True,
        check_answer_consistency: bool = True,
        verify_knowledge_alignment: bool = True,
        **kwargs: Any
    ):
        """Initialize the RAG Certainty metric.
        
        Args:
            llm: The LLM instance to use for evaluation
            assess_generation_confidence: Whether to assess model confidence in generation
            evaluate_retrieval_quality: Whether to evaluate quality of retrieved contexts
            check_answer_consistency: Whether to check internal consistency of answer
            verify_knowledge_alignment: Whether to verify alignment with retrieved knowledge
            **kwargs: Additional configuration parameters
        """
        super().__init__(llm=llm, **kwargs)
        self.assess_generation_confidence = assess_generation_confidence
        self.evaluate_retrieval_quality = evaluate_retrieval_quality
        self.check_answer_consistency = check_answer_consistency
        self.verify_knowledge_alignment = verify_knowledge_alignment
    
    def name(self) -> str:
        """Return the name of this metric."""
        return "rag_certainty"

    @property
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        return MetricType.COMPOSITE

    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """
        Evaluate the RAG system's certainty in its response.
        
        Args:
            rag_input: The RAG input containing query, contexts, and answer
            
        Returns:
            MetricResult with certainty score and detailed breakdown
        """
        self.validate_input(rag_input)
        
        # Check if answer is empty
        if not rag_input.answer.strip():
            return MetricResult(
                score=0.0,
                explanation="Empty answer provided - zero certainty",
                details={"error": "empty_answer"}
            )

        try:
            return await self._evaluate_rag_certainty(rag_input)
        except Exception as e:
            raise RuntimeError(f"RAG certainty evaluation failed: {str(e)}")

    async def _evaluate_rag_certainty(self, rag_input: RAGInput) -> MetricResult:
        """
        Perform the RAG certainty evaluation.
        
        Args:
            rag_input: The RAG input to evaluate
            
        Returns:
            MetricResult with detailed certainty assessment
        """
        evaluation_components = []
        
        # Component 1: Generation confidence assessment
        if self.assess_generation_confidence:
            gen_confidence, gen_details = await self._assess_generation_confidence(
                rag_input.query, rag_input.answer
            )
            evaluation_components.append(("generation_confidence", gen_confidence, gen_details))
        
        # Component 2: Retrieval quality assessment
        # Fix: Use explicit None and length check instead of boolean evaluation
        if self.evaluate_retrieval_quality and rag_input.retrieved_contexts is not None and len(rag_input.retrieved_contexts) > 0:
            retrieval_quality, retrieval_details = await self._evaluate_retrieval_quality(
                rag_input.query, rag_input.retrieved_contexts
            )
            evaluation_components.append(("retrieval_quality", retrieval_quality, retrieval_details))
        
        # Component 3: Answer consistency assessment
        if self.check_answer_consistency:
            consistency_score, consistency_details = await self._check_answer_consistency(
                rag_input.query, rag_input.answer
            )
            evaluation_components.append(("answer_consistency", consistency_score, consistency_details))
        
        # Component 4: Knowledge alignment assessment
        # Fix: Use explicit None and length check instead of boolean evaluation
        if self.verify_knowledge_alignment and rag_input.retrieved_contexts is not None and len(rag_input.retrieved_contexts) > 0:
            alignment_score, alignment_details = await self._verify_knowledge_alignment(
                rag_input.query, rag_input.answer, rag_input.retrieved_contexts
            )
            evaluation_components.append(("knowledge_alignment", alignment_score, alignment_details))
        
        # Calculate weighted certainty score
        certainty_score = self._calculate_certainty_score(evaluation_components)
        
        # Determine deployment recommendation
        deployment_recommendation = self._get_deployment_recommendation(certainty_score)
        
        # Generate explanation
        explanation = self._generate_explanation(certainty_score, evaluation_components, deployment_recommendation)
        
        # Prepare detailed results
        details = {
            "certainty_score": certainty_score,
            "deployment_recommendation": deployment_recommendation,
            "components_evaluated": len(evaluation_components),
            "evaluation_breakdown": {
                name: {"score": score, "details": details}
                for name, score, details in evaluation_components
            },
            "production_readiness": self._assess_production_readiness(certainty_score)
        }
        
        return MetricResult(
            score=certainty_score,
            explanation=explanation,
            details=details
        )

    async def _assess_generation_confidence(
        self, 
        query: str, 
        answer: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Assess the confidence level of the generated answer.
        
        Args:
            query: The user query
            answer: The generated answer
            
        Returns:
            Tuple of (confidence_score, details_dict)
        """
        confidence_prompt = f"""Assess how confident a language model would be in providing this answer to the given question.

Question: {query}
Answer: {answer}

Please evaluate the generation confidence by considering:
1. Does the answer show clear certainty or uncertainty indicators?
2. Are there hedging words (maybe, possibly, might) that indicate low confidence?
3. Does the answer provide specific, definitive information?
4. Would a model be confident about the factual claims made?
5. Does the answer avoid speculative or uncertain language?

Rate the generation confidence on a scale of 0.0 to 1.0:
- 1.0: Very high confidence - definitive, certain language with specific facts
- 0.8: High confidence - mostly certain with minimal hedging
- 0.6: Moderate confidence - some certainty but includes qualifiers
- 0.4: Low confidence - frequent hedging and uncertainty indicators
- 0.2: Very low confidence - mostly speculative and uncertain language
- 0.0: No confidence - completely uncertain or speculative

Provide your evaluation as:
Score: [0.0-1.0]
Explanation: [Detailed explanation of confidence assessment]

Response:"""
        
        response = await self.generate_llm_response(
            prompt=confidence_prompt,
            temperature=0.0,
            max_tokens=400
        )
        
        score, explanation = parse_llm_score(response.content)
        
        details = {
            "evaluation_prompt": "generation_confidence_assessment",
            "llm_explanation": explanation
        }
        
        return score, details

    async def _evaluate_retrieval_quality(
        self, 
        query: str, 
        contexts: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the quality and relevance of retrieved contexts.
        
        Args:
            query: The user query
            contexts: Retrieved contexts
            
        Returns:
            Tuple of (quality_score, details_dict)
        """
        contexts_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        quality_prompt = f"""Evaluate the quality and relevance of the retrieved contexts for answering the question.

Question: {query}

Retrieved Contexts:
{contexts_text}

Please assess the retrieval quality by considering:
1. How relevant are the contexts to the question?
2. Do the contexts contain information needed to answer the question?
3. Are the contexts comprehensive and informative?
4. Would these contexts enable a confident, accurate answer?
5. Do the contexts complement each other or provide redundant information?

Rate the retrieval quality on a scale of 0.0 to 1.0:
- 1.0: Excellent quality - highly relevant, comprehensive, and informative contexts
- 0.8: Good quality - relevant contexts with sufficient information
- 0.6: Moderate quality - somewhat relevant but may lack some information
- 0.4: Poor quality - limited relevance or insufficient information
- 0.2: Very poor quality - mostly irrelevant or unhelpful contexts
- 0.0: No quality - completely irrelevant or no useful information

Provide your evaluation as:
Score: [0.0-1.0]
Explanation: [Detailed explanation of retrieval quality assessment]

Response:"""
        
        response = await self.generate_llm_response(
            prompt=quality_prompt,
            temperature=0.0,
            max_tokens=400
        )
        
        score, explanation = parse_llm_score(response.content)
        
        details = {
            "evaluation_prompt": "retrieval_quality_assessment",
            "llm_explanation": explanation,
            "contexts_count": len(contexts)
        }
        
        return score, details

    async def _check_answer_consistency(
        self, 
        query: str, 
        answer: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Check the internal consistency of the answer.
        
        Args:
            query: The user query
            answer: The generated answer
            
        Returns:
            Tuple of (consistency_score, details_dict)
        """
        consistency_prompt = f"""Evaluate the internal consistency of this answer - whether all parts of the answer are coherent and non-contradictory.

Question: {query}
Answer: {answer}

Please assess the answer consistency by considering:
1. Are all statements in the answer mutually consistent?
2. Does the answer avoid contradicting itself?
3. Do different parts of the answer support the same conclusion?
4. Is the reasoning coherent throughout the answer?
5. Are there any logical inconsistencies or contradictions?

Rate the answer consistency on a scale of 0.0 to 1.0:
- 1.0: Perfectly consistent - all parts coherent and mutually supporting
- 0.8: Highly consistent - mostly coherent with minor inconsistencies
- 0.6: Moderately consistent - generally coherent but some unclear areas
- 0.4: Somewhat inconsistent - noticeable contradictions or incoherence
- 0.2: Highly inconsistent - significant contradictions throughout
- 0.0: Completely inconsistent - major contradictions and incoherence

Provide your evaluation as:
Score: [0.0-1.0]
Explanation: [Detailed explanation of consistency assessment]

Response:"""
        
        response = await self.generate_llm_response(
            prompt=consistency_prompt,
            temperature=0.0,
            max_tokens=400
        )
        
        score, explanation = parse_llm_score(response.content)
        
        details = {
            "evaluation_prompt": "answer_consistency_assessment",
            "llm_explanation": explanation
        }
        
        return score, details

    async def _verify_knowledge_alignment(
        self, 
        query: str, 
        answer: str, 
        contexts: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Verify how well the answer aligns with the retrieved knowledge.
        
        Args:
            query: The user query
            answer: The generated answer
            contexts: Retrieved contexts
            
        Returns:
            Tuple of (alignment_score, details_dict)
        """
        contexts_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        alignment_prompt = f"""Evaluate how well the answer aligns with and is supported by the retrieved knowledge.

Question: {query}
Answer: {answer}

Retrieved Knowledge:
{contexts_text}

Please assess the knowledge alignment by considering:
1. Does the answer accurately reflect the information in the contexts?
2. Are the claims in the answer well-supported by the retrieved knowledge?
3. Does the answer stay within the bounds of what the contexts support?
4. Is there good alignment between answer content and available knowledge?
5. Does the answer avoid making unsupported claims beyond the contexts?

Rate the knowledge alignment on a scale of 0.0 to 1.0:
- 1.0: Perfect alignment - answer fully supported and grounded in contexts
- 0.8: Strong alignment - answer well-supported with minor extrapolations
- 0.6: Good alignment - answer mostly supported but some unsupported elements
- 0.4: Weak alignment - answer partially supported with notable gaps
- 0.2: Poor alignment - answer poorly supported by available knowledge
- 0.0: No alignment - answer contradicts or ignores retrieved knowledge

Provide your evaluation as:
Score: [0.0-1.0]
Explanation: [Detailed explanation of knowledge alignment assessment]

Response:"""
        
        response = await self.generate_llm_response(
            prompt=alignment_prompt,
            temperature=0.0,
            max_tokens=400
        )
        
        score, explanation = parse_llm_score(response.content)
        
        details = {
            "evaluation_prompt": "knowledge_alignment_assessment",
            "llm_explanation": explanation,
            "contexts_count": len(contexts)
        }
        
        return score, details

    def _calculate_certainty_score(
        self, 
        evaluation_components: List[Tuple[str, float, Dict[str, Any]]]
    ) -> float:
        """
        Calculate weighted certainty score based on evaluation components.
        
        Args:
            evaluation_components: List of evaluation components with scores
            
        Returns:
            Weighted certainty score
        """
        if not evaluation_components:
            return 0.0
        
        # Weight the components based on importance for production deployment
        weights = {
            "generation_confidence": 0.3,
            "retrieval_quality": 0.2,
            "answer_consistency": 0.2,
            "knowledge_alignment": 0.3
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for name, score, _ in evaluation_components:
            weight = weights.get(name, 0.25)  # Default weight if not specified
            total_weighted_score += score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _get_deployment_recommendation(self, certainty_score: float) -> Dict[str, Any]:
        """
        Get deployment recommendation based on certainty score.
        
        Args:
            certainty_score: The calculated certainty score
            
        Returns:
            Deployment recommendation dictionary
        """
        if certainty_score >= 0.9:
            return {
                "level": "automated",
                "description": "Very high certainty - suitable for automated deployment",
                "oversight": "minimal",
                "confidence": "very_high"
            }
        elif certainty_score >= 0.7:
            return {
                "level": "supervised",
                "description": "High certainty - suitable with light human oversight",
                "oversight": "light",
                "confidence": "high"
            }
        elif certainty_score >= 0.4:
            return {
                "level": "review_required",
                "description": "Moderate certainty - requires human review before deployment",
                "oversight": "moderate",
                "confidence": "moderate"
            }
        else:
            return {
                "level": "manual_intervention",
                "description": "Low certainty - requires significant human intervention",
                "oversight": "high",
                "confidence": "low"
            }

    def _assess_production_readiness(self, certainty_score: float) -> Dict[str, Any]:
        """
        Assess production readiness based on certainty score.
        
        Args:
            certainty_score: The calculated certainty score
            
        Returns:
            Production readiness assessment
        """
        ready = certainty_score >= 0.7
        risk_level = "low" if certainty_score >= 0.8 else "medium" if certainty_score >= 0.5 else "high"
        
        return {
            "ready_for_production": ready,
            "risk_level": risk_level,
            "recommended_testing": "minimal" if certainty_score >= 0.9 else "standard" if certainty_score >= 0.7 else "extensive",
            "monitoring_level": "standard" if certainty_score >= 0.8 else "enhanced"
        }

    def _generate_explanation(
        self, 
        certainty_score: float, 
        evaluation_components: List[Tuple[str, float, Dict[str, Any]]],
        deployment_recommendation: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable explanation of the certainty score.
        
        Args:
            certainty_score: The calculated certainty score
            evaluation_components: List of evaluation components with scores
            deployment_recommendation: Deployment recommendation
            
        Returns:
            Explanation string
        """
        # Determine overall certainty level
        if certainty_score >= 0.9:
            certainty_level = "very high"
        elif certainty_score >= 0.7:
            certainty_level = "high"
        elif certainty_score >= 0.5:
            certainty_level = "moderate"
        elif certainty_score >= 0.3:
            certainty_level = "low"
        else:
            certainty_level = "very low"
        
        explanation = f"RAG system certainty is {certainty_level} ({certainty_score:.2f}). "
        
        # Add deployment recommendation
        explanation += f"Deployment recommendation: {deployment_recommendation['description']}. "
        
        # Add component insights
        if evaluation_components:
            strong_components = [name for name, score, _ in evaluation_components if score >= 0.7]
            weak_components = [name for name, score, _ in evaluation_components if score < 0.5]
            
            if strong_components:
                explanation += f"Strong areas: {', '.join(strong_components)}. "
            if weak_components:
                explanation += f"Needs improvement: {', '.join(weak_components)}. "
        
        return explanation

    @property
    def requires_llm(self) -> bool:
        """This metric requires an LLM for certainty evaluation."""
        return True 