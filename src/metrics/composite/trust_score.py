"""TrustScore composite metric implementation."""

import re
import random
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .base_composite import BaseCompositeMetric
from ...core.types import RAGInput, MetricResult


class FactualConsistency(Enum):
    """Factual consistency assessment results."""
    SUPPORT = "support"
    CONTRADICT = "contradict"
    NEUTRAL = "neutral"


class TrustScore(BaseCompositeMetric):
    """TrustScore composite metric combining behavioral and factual consistency.
    
    TrustScore evaluates trustworthiness through two components:
    1. TrustBC (Behavioral Consistency): Tests model confidence via multiple-choice questions
    2. TrustFC (Factual Consistency): Validates answer against retrieved contexts
    
    The final score combines both assessments according to a predefined rubric.
    """

    def __init__(
        self,
        llm: Any = None,
        max_checks: int = 10,
        num_distractors: int = 3,
        **kwargs: Any
    ):
        """Initialize TrustScore metric.
        
        Args:
            llm: LLM instance for evaluations
            max_checks: Maximum number of behavioral consistency checks
            num_distractors: Number of distractor options to generate
            **kwargs: Additional configuration parameters
        """
        super().__init__(llm=llm, **kwargs)
        self.max_checks = max_checks
        self.num_distractors = num_distractors

    def name(self) -> str:
        """Return the name of this metric."""
        return "trust_score"

    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate TrustScore by combining behavioral and factual consistency.
        
        Args:
            rag_input: The RAG input to evaluate
            
        Returns:
            MetricResult containing the trust score and detailed breakdown
        """
        self.validate_input(rag_input)
        
        try:
            # Evaluate both components in parallel
            bc_score, bc_details = await self._evaluate_behavioral_consistency(rag_input)
            fc_result, fc_details = await self._evaluate_factual_consistency(rag_input)
            
            # Calculate overall trust score using the rubric
            overall_score = self._calculate_trust_score(bc_score, fc_result)
            
            # Create detailed explanation
            explanation = self._create_explanation(bc_score, fc_result, overall_score)
            
            details = {
                "behavioral_consistency": {
                    "score": bc_score,
                    "details": bc_details
                },
                "factual_consistency": {
                    "result": fc_result.value,
                    "details": fc_details
                },
                "trust_rubric": self._get_trust_rubric_explanation(bc_score, fc_result)
            }
            
            return MetricResult(
                score=overall_score,
                explanation=explanation,
                details=details
            )
            
        except Exception as e:
            return MetricResult(
                score=0.0,
                explanation=f"TrustScore evaluation failed: {str(e)}",
                details={"error": str(e)}
            )

    async def _evaluate_behavioral_consistency(
        self, 
        rag_input: RAGInput
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate behavioral consistency using multiple-choice questions.
        
        Args:
            rag_input: The RAG input to evaluate
            
        Returns:
            Tuple of (consistency_score, details_dict)
        """
        if not rag_input.answer.strip():
            return 0.0, {"reason": "empty_answer", "checks_performed": 0}
        
        # Extract key entities/words from the answer for substitution
        key_words = self._extract_key_words(rag_input.query, rag_input.answer)
        
        if not key_words:
            return 1.0, {"reason": "no_substitutable_words", "checks_performed": 0}
        
        checks_performed = 0
        consistent_responses = 0
        check_details = []
        
        for check_num in range(min(self.max_checks, len(key_words) * 2)):
            try:
                # Generate multiple choice question with distractors
                mc_question = self._generate_multiple_choice_question(
                    rag_input.query, rag_input.answer, key_words
                )
                
                if not mc_question:
                    continue
                
                # Get LLM response to the multiple choice question
                response = await self._get_llm_choice(mc_question)
                
                checks_performed += 1
                is_consistent = response == "B"  # Original answer is always option B
                
                if is_consistent:
                    consistent_responses += 1
                else:
                    # Stop early on first inconsistent response
                    check_details.append({
                        "check": check_num + 1,
                        "question": mc_question,
                        "llm_choice": response,
                        "consistent": is_consistent
                    })
                    break
                
                check_details.append({
                    "check": check_num + 1,
                    "question": mc_question,
                    "llm_choice": response,
                    "consistent": is_consistent
                })
                
            except Exception as e:
                check_details.append({
                    "check": check_num + 1,
                    "error": str(e)
                })
                continue
        
        # Calculate consistency score: 1.0 if all checks passed, 0.0 if any failed
        consistency_score = 1.0 if consistent_responses == checks_performed and checks_performed > 0 else 0.0
        
        details = {
            "checks_performed": checks_performed,
            "consistent_responses": consistent_responses,
            "consistency_rate": consistent_responses / checks_performed if checks_performed > 0 else 0.0,
            "check_details": check_details
        }
        
        return consistency_score, details

    async def _evaluate_factual_consistency(
        self, 
        rag_input: RAGInput
    ) -> Tuple[FactualConsistency, Dict[str, Any]]:
        """Evaluate factual consistency against retrieved contexts.
        
        Args:
            rag_input: The RAG input to evaluate
            
        Returns:
            Tuple of (factual_consistency_result, details_dict)
        """
        # Fix: Use explicit None and length check instead of boolean evaluation
        if rag_input.retrieved_contexts is None or len(rag_input.retrieved_contexts) == 0:
            return FactualConsistency.NEUTRAL, {"reason": "no_contexts"}
        
        if not rag_input.answer.strip():
            return FactualConsistency.NEUTRAL, {"reason": "empty_answer"}
        
        # Combine contexts into evidence
        evidence = "\n\n".join(rag_input.retrieved_contexts)
        
        # Create entailment prompt
        prompt = self._create_entailment_prompt(
            evidence, rag_input.query, rag_input.answer
        )
        
        try:
            response = await self.generate_llm_response(prompt)
            
            # Parse the response to extract the relation
            relation = self._parse_entailment_response(response.content)
            
            details = {
                "evidence_contexts": len(rag_input.retrieved_contexts),
                "entailment_prompt": prompt,
                "llm_response": response.content,
                "parsed_relation": relation.value
            }
            
            return relation, details
            
        except Exception as e:
            return FactualConsistency.NEUTRAL, {"error": str(e)}

    def _extract_key_words(self, query: str, answer: str) -> List[str]:
        """Extract key words from answer for substitution, prioritizing entities and nouns.
        
        Args:
            query: User query
            answer: Generated answer
            
        Returns:
            List of key words suitable for substitution
        """
        # Simple keyword extraction - prioritize words in answer but not in query
        query_words = set(query.lower().split())
        answer_words = answer.split()
        
        # Find words exclusive to answer (higher priority)
        exclusive_words = [
            word for word in answer_words 
            if word.lower() not in query_words and len(word) > 2
        ]
        
        # Find proper nouns (capitalized words)
        proper_nouns = [
            word for word in answer_words 
            if word[0].isupper() and len(word) > 2
        ]
        
        # Combine and deduplicate, prioritizing proper nouns and exclusive words
        key_words = list(dict.fromkeys(proper_nouns + exclusive_words))
        
        return key_words[:10]  # Limit to top 10 candidates

    def _generate_multiple_choice_question(
        self, 
        query: str, 
        answer: str, 
        key_words: List[str]
    ) -> Optional[str]:
        """Generate a multiple-choice question with distractors.
        
        Args:
            query: Original user query
            answer: Original answer (will be option B)
            key_words: Key words that can be substituted
            
        Returns:
            Formatted multiple-choice question string
        """
        if not key_words:
            return None
        
        # Select a word to substitute
        target_word = random.choice(key_words)
        
        # Generate simple distractors (in a real implementation, you'd use
        # more sophisticated methods like DBpedia, word embeddings, etc.)
        distractors = self._generate_distractors(target_word)
        
        if len(distractors) < self.num_distractors:
            return None
        
        # Create distractor answers
        distractor_answers = []
        for distractor in distractors[:self.num_distractors]:
            distractor_answer = answer.replace(target_word, distractor)
            distractor_answers.append(distractor_answer)
        
        # Shuffle options (but keep original as B)
        options = [
            f"A) {distractor_answers[0]}",
            f"B) {answer}",  # Original answer
            f"C) {distractor_answers[1] if len(distractor_answers) > 1 else distractor_answers[0]}",
            f"D) {distractor_answers[2] if len(distractor_answers) > 2 else distractor_answers[0]}",
            "E) None of the above."
        ]
        
        question = f"Question: {query}\n\n" + "\n".join(options)
        return question

    def _generate_distractors(self, target_word: str) -> List[str]:
        """Generate distractor words for the target word.
        
        In a full implementation, this would use DBpedia, word embeddings,
        or other sophisticated methods. Here we use simple heuristics.
        
        Args:
            target_word: Word to generate distractors for
            
        Returns:
            List of distractor words
        """
        # Simple distractor generation (placeholder)
        # In practice, use word embeddings, DBpedia, or other methods
        common_names = [
            "Alexander", "Napoleon", "Einstein", "Shakespeare", "Gandhi",
            "Churchill", "Lincoln", "Washington", "Newton", "Darwin"
        ]
        
        common_places = [
            "London", "Paris", "Tokyo", "Sydney", "Berlin",
            "Rome", "Madrid", "Vienna", "Moscow", "Beijing"
        ]
        
        # Simple heuristic: if target looks like a proper noun, use names/places
        if target_word[0].isupper():
            return random.sample(common_names + common_places, 
                               min(self.num_distractors + 2, len(common_names + common_places)))
        
        # For other words, generate simple variations
        return [f"{target_word}_{i}" for i in range(self.num_distractors + 2)]

    async def _get_llm_choice(self, mc_question: str) -> str:
        """Get LLM's choice for multiple-choice question.
        
        Args:
            mc_question: Formatted multiple-choice question
            
        Returns:
            LLM's choice (A, B, C, D, or E)
        """
        prompt = f"""Please answer the following multiple-choice question by selecting the most appropriate option.
Respond with only the letter of your choice (A, B, C, D, or E).

{mc_question}

Your answer:"""
        
        response = await self.generate_llm_response(prompt)
        
        # Extract the choice letter
        choice_match = re.search(r'[ABCDE]', response.content.strip().upper())
        if choice_match:
            return choice_match.group()
        
        return "E"  # Default to "None of the above" if parsing fails

    def _create_entailment_prompt(
        self, 
        evidence: str, 
        question: str, 
        answer: str
    ) -> str:
        """Create prompt for factual consistency evaluation.
        
        Args:
            evidence: Retrieved contexts as evidence
            question: Original user question
            answer: Generated answer
            
        Returns:
            Formatted entailment prompt
        """
        prompt = f"""INSTRUCTION: Assess whether the provided evidence aligns with or contradicts the given question-answer pair, and categorize the relationship as either 'support', 'contradict', or 'neutral'.

Evidence: {evidence}

Question: {question}

Answer: {answer}

Relation:"""
        
        return prompt

    def _parse_entailment_response(self, response: str) -> FactualConsistency:
        """Parse LLM response to extract factual consistency relation.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed FactualConsistency enum value
        """
        response_lower = response.lower().strip()
        
        if "support" in response_lower:
            return FactualConsistency.SUPPORT
        elif "contradict" in response_lower:
            return FactualConsistency.CONTRADICT
        else:
            return FactualConsistency.NEUTRAL

    def _calculate_trust_score(
        self, 
        bc_score: float, 
        fc_result: FactualConsistency
    ) -> float:
        """Calculate overall trust score using the TrustScore rubric.
        
        Args:
            bc_score: Behavioral consistency score (0.0 or 1.0)
            fc_result: Factual consistency result
            
        Returns:
            Overall trust score according to rubric
        """
        is_consistent = bc_score == 1.0
        
        # TrustScore rubric from the paper
        if is_consistent and fc_result == FactualConsistency.SUPPORT:
            return 1.0
        elif not is_consistent and fc_result == FactualConsistency.SUPPORT:
            return 0.8
        elif is_consistent and fc_result == FactualConsistency.NEUTRAL:
            return 0.6
        elif not is_consistent and fc_result == FactualConsistency.NEUTRAL:
            return 0.4
        elif is_consistent and fc_result == FactualConsistency.CONTRADICT:
            return 0.2
        else:  # not is_consistent and fc_result == FactualConsistency.CONTRADICT
            return 0.0

    def _create_explanation(
        self, 
        bc_score: float, 
        fc_result: FactualConsistency, 
        overall_score: float
    ) -> str:
        """Create human-readable explanation of the trust score.
        
        Args:
            bc_score: Behavioral consistency score
            fc_result: Factual consistency result
            overall_score: Overall trust score
            
        Returns:
            Formatted explanation string
        """
        bc_status = "consistent" if bc_score == 1.0 else "inconsistent"
        fc_status = fc_result.value
        
        explanation = f"TrustScore: {overall_score:.1f} - "
        explanation += f"The answer is behaviorally {bc_status} "
        explanation += f"and factually {fc_status} with the evidence. "
        
        if overall_score >= 0.8:
            explanation += "High trustworthiness."
        elif overall_score >= 0.6:
            explanation += "Moderate trustworthiness."
        elif overall_score >= 0.4:
            explanation += "Low trustworthiness."
        else:
            explanation += "Very low trustworthiness."
        
        return explanation

    def _get_trust_rubric_explanation(
        self, 
        bc_score: float, 
        fc_result: FactualConsistency
    ) -> str:
        """Get explanation of which rubric category applies.
        
        Args:
            bc_score: Behavioral consistency score
            fc_result: Factual consistency result
            
        Returns:
            Rubric category explanation
        """
        is_consistent = bc_score == 1.0
        
        if is_consistent and fc_result == FactualConsistency.SUPPORT:
            return "Consistent + Support: 1.0"
        elif not is_consistent and fc_result == FactualConsistency.SUPPORT:
            return "Inconsistent + Support: 0.8"
        elif is_consistent and fc_result == FactualConsistency.NEUTRAL:
            return "Consistent + Neutral: 0.6"
        elif not is_consistent and fc_result == FactualConsistency.NEUTRAL:
            return "Inconsistent + Neutral: 0.4"
        elif is_consistent and fc_result == FactualConsistency.CONTRADICT:
            return "Consistent + Contradict: 0.2"
        else:
            return "Inconsistent + Contradict: 0.0" 