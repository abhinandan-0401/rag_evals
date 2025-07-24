"""Answer Relevance metric for evaluating answer-query alignment."""

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


class AnswerRelevance(BaseMetric):
    """Evaluates how relevant the generated answer is to the user's query.
    
    This metric assesses whether the answer actually addresses what the user asked,
    regardless of factual accuracy or completeness.
    
    Score interpretation:
    - 1.0: Answer directly and fully addresses the query
    - 0.5: Answer partially addresses the query
    - 0.0: Answer does not address the query
    """

    def name(self) -> str:
        """Return the name of this metric."""
        return "answer_relevance"

    @property
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        return MetricType.GENERATION

    def _validate_metric_specific(self, rag_input: RAGInput) -> None:
        """Validate that required inputs are present for answer relevance evaluation."""
        if not rag_input.answer or not rag_input.answer.strip():
            raise ValueError("Answer relevance metric requires a non-empty answer")

    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate answer relevance using modular YAML-based prompts.
        
        Args:
            rag_input: RAG input containing query and answer to evaluate
            
        Returns:
            MetricResult with relevance score and reasoning
        """
        try:
            # Validate inputs
            self._validate_metric_specific(rag_input)
            
            # Quick validation
            if not rag_input.answer.strip():
                return MetricResult(
                    score=0.0,
                    reason="Empty answer provided",
                    details={"answer_length": 0}
                )

            self.logger.info(f"Starting {self.name()} evaluation")

            # Use simple relevance evaluation for reliability
            return await self._simple_relevance_evaluation(rag_input)
            
        except Exception as e:
            self.logger.error(f"Failed {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate answer relevance: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _simple_relevance_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Simple answer relevance evaluation using the main prompt template.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with relevance assessment
        """
        try:
            # Get answer relevance evaluation prompt from YAML (note: fixed filename)
            relevance_prompt = get_prompt("answer_relevancy", "template")
            
            # Format the prompt with variables
            prompt = relevance_prompt.format(
                query=rag_input.query,
                answer=rag_input.answer
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("answer_relevancy", "system_prompt")
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
                    "query_length": len(rag_input.query),
                    "answer_length": len(rag_input.answer),
                    "contexts_provided": len(rag_input.retrieved_contexts) if rag_input.retrieved_contexts is not None else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed simple {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate answer relevance: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _contextual_relevance_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Contextual answer relevance evaluation with context information.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with contextual relevance assessment
        """
        try:
            # Get contextual evaluation prompt from YAML
            evaluation_prompt = get_prompt("answer_relevancy", "evaluation_prompt")
            
            # Format contexts if available
            contexts_str = ""
            if rag_input.retrieved_contexts is not None and len(rag_input.retrieved_contexts) > 0:
                contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
            
            # Format the prompt with variables
            prompt = evaluation_prompt.format(
                query=rag_input.query,
                answer=rag_input.answer,
                context=contexts_str
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("answer_relevancy", "system_prompt")
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
                    "evaluation_type": "contextual_relevance",
                    "evaluation_prompt": "evaluation_prompt",
                    "query_length": len(rag_input.query),
                    "answer_length": len(rag_input.answer),
                    "contexts_count": len(rag_input.retrieved_contexts) if rag_input.retrieved_contexts is not None else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed contextual {self.name()} evaluation: {str(e)}")
            # Fallback to simple evaluation
            return await self._simple_relevance_evaluation(rag_input)

    async def _evaluate_with_multiple_questions(self, rag_input: RAGInput) -> MetricResult:
        """Alternative evaluation method using generated questions (advanced).
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with question-based relevance assessment
        """
        try:
            # This is a more complex method that could be used for advanced evaluation
            # For now, fallback to simple evaluation to ensure reliability
            return await self._simple_relevance_evaluation(rag_input)
            
        except Exception as e:
            self.logger.error(f"Failed question-based {self.name()} evaluation: {str(e)}")
            # Fallback to simple evaluation
            return await self._simple_relevance_evaluation(rag_input)

    def _extract_questions(self, response_text: str) -> List[str]:
        """Extract questions from LLM response.
        
        Args:
            response_text: LLM response containing questions
            
        Returns:
            List of extracted questions
        """
        try:
            questions = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line and ('?' in line or line.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who'))):
                    # Clean up the question
                    question = line.split('.', 1)[-1].strip()
                    if question and len(question) > 10:
                        questions.append(question)
            return questions[:5]  # Limit to 5 questions
            
        except Exception as e:
            self.logger.error(f"Failed to extract questions: {str(e)}")
            return []

    async def _calculate_question_similarity(self, original_query: str, generated_question: str) -> float:
        """Calculate semantic similarity between queries.
        
        Args:
            original_query: Original user query
            generated_question: Generated question from answer
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Simple approach - could be enhanced with embedding similarity
            similarity_prompt = """
            Rate the semantic similarity between these two questions on a scale of 0.0 to 1.0:
            
            Question 1: {query1}
            Question 2: {query2}
            
            Score: [Your score from 0.0 to 1.0]
            """
            
            formatted_prompt = similarity_prompt.format(
                query1=original_query,
                query2=generated_question
            )
            
            response = await self.generate_llm_response(
                prompt=formatted_prompt,
                temperature=0.0,
                max_tokens=50
            )
            
            score, _ = parse_llm_score(response.content)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate question similarity: {str(e)}")
            return 0.0

    @property
    def requires_contexts(self) -> bool:
        """Answer relevance doesn't require contexts."""
        return False 