"""Completeness metric for evaluating comprehensive coverage of question aspects."""

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


class Completeness(BaseMetric):
    """
    Evaluates how completely the generated answer addresses all aspects of the query.
    
    This metric is particularly useful for multi-part questions and complex queries by:
    1. Identifying different aspects/components of the query
    2. Checking coverage of each identified aspect in the answer
    3. Assessing the depth and thoroughness of the response
    4. Detecting missing information that should be addressed
    
    Score interpretation:
    - 1.0: Answer comprehensively addresses all aspects of the query
    - 0.5: Answer covers some aspects but misses important components
    - 0.0: Answer fails to address the main aspects of the query
    """
    
    def __init__(
        self, 
        llm: Any = None,
        check_depth: bool = True,
        identify_missing_aspects: bool = True,
        **kwargs: Any
    ):
        """Initialize the Completeness metric.
        
        Args:
            llm: The LLM instance to use for evaluation
            check_depth: Whether to assess depth of coverage, not just presence
            identify_missing_aspects: Whether to identify specific missing aspects
            **kwargs: Additional configuration parameters
        """
        super().__init__(llm=llm, **kwargs)
        self.check_depth = check_depth
        self.identify_missing_aspects = identify_missing_aspects
    
    def name(self) -> str:
        """Return the name of this metric."""
        return "completeness"

    @property
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        return MetricType.GENERATION

    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        """Evaluate the completeness of the answer.
        
        Args:
            rag_input: RAG input containing query, answer, and contexts
            
        Returns:
            MetricResult with completeness score and analysis
        """
        if not rag_input.answer or not rag_input.answer.strip():
            return MetricResult(
                score=0.0,
                reason="Empty answer provided - cannot assess completeness",
                details={"error": "empty_answer"}
            )
        
        if not rag_input.query or not rag_input.query.strip():
            return MetricResult(
                score=0.0,
                reason="Empty query provided - cannot assess completeness",
                details={"error": "empty_query"}
            )
        
        # For now, use simple evaluation to ensure it works
        return await self._simple_completeness_evaluation(rag_input)

    async def _simple_completeness_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform simple completeness evaluation using a single prompt.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with completeness assessment
        """
        try:
            # Get the simple prompt from YAML
            simple_prompt = get_prompt("completeness", "simple_prompt")
            
            # Format contexts for the prompt
            contexts_str = ""
            if rag_input.retrieved_contexts is not None and len(rag_input.retrieved_contexts) > 0:
                contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
            
            # Format the prompt with variables
            prompt = simple_prompt.format(
                query=rag_input.query,
                answer=rag_input.answer,
                context=contexts_str
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("completeness", "system_prompt")
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
                    "evaluation_type": "simple",
                    "evaluation_prompt": "simple_completeness_assessment",
                    "contexts_count": len(rag_input.retrieved_contexts) if rag_input.retrieved_contexts is not None else 0,
                    "answer_length": len(rag_input.answer),
                    "query_length": len(rag_input.query)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed {self.name()} evaluation: {str(e)}")
            return MetricResult(
                score=0.0,
                reason=f"Failed to evaluate completeness: {str(e)}",
                details={
                    "error": "evaluation_failed", 
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

    async def _detailed_completeness_evaluation(self, rag_input: RAGInput) -> MetricResult:
        """Perform detailed completeness evaluation with aspect identification.
        
        Args:
            rag_input: RAG input data
            
        Returns:
            MetricResult with detailed completeness analysis
        """
        try:
            # Step 1: Identify aspects of the query
            aspects = await self._identify_query_aspects(rag_input)
            
            if not aspects:
                # Fallback to simple evaluation if no aspects identified
                return await self._simple_completeness_evaluation(rag_input)
            
            # Step 2: Evaluate coverage of each aspect
            aspect_coverage = []
            for i, aspect in enumerate(aspects):
                coverage = await self._evaluate_aspect_coverage(
                    rag_input.query, aspect, rag_input.answer, rag_input.retrieved_contexts
                )
                aspect_coverage.append({
                    "aspect": aspect,
                    "coverage": coverage,
                    "index": i
                })
            
            # Step 3: Calculate overall completeness score
            coverage_scores = [item["coverage"]["score"] for item in aspect_coverage]
            overall_score = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
            
            # Step 4: Identify missing or poorly covered aspects
            poorly_covered = [
                item for item in aspect_coverage 
                if item["coverage"]["score"] < 0.5
            ]
            
            well_covered = [
                item for item in aspect_coverage 
                if item["coverage"]["score"] >= 0.7
            ]
            
            # Generate explanation
            if overall_score >= 0.8:
                reason = f"Answer comprehensively covers {len(well_covered)}/{len(aspects)} aspects of the query"
            elif overall_score >= 0.5:
                reason = f"Answer partially covers the query with {len(well_covered)}/{len(aspects)} aspects well-addressed"
            else:
                reason = f"Answer has limited coverage with only {len(well_covered)}/{len(aspects)} aspects adequately addressed"
            
            return MetricResult(
                score=overall_score,
                reason=reason,
                details={
                    "evaluation_type": "detailed",
                    "aspects_identified": len(aspects),
                    "aspects_well_covered": len(well_covered),
                    "aspects_poorly_covered": len(poorly_covered),
                    "aspect_analysis": aspect_coverage,
                    "overall_score": overall_score,
                    "missing_aspects": [item["aspect"] for item in poorly_covered]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed detailed {self.name()} evaluation: {str(e)}")
            # Fallback to simple evaluation
            return await self._simple_completeness_evaluation(rag_input)

    async def _identify_query_aspects(self, rag_input: RAGInput) -> List[str]:
        """Identify different aspects or components of the query.
        
        Args:
            rag_input: The RAG input data
            
        Returns:
            List of identified aspects
        """
        try:
            # Get the aspect prompt from YAML
            aspect_prompt = get_prompt("completeness", "aspect_prompt")
            
            # Format contexts
            contexts_str = ""
            if rag_input.retrieved_contexts is not None and len(rag_input.retrieved_contexts) > 0:
                contexts_str = format_contexts(rag_input.retrieved_contexts, numbered=True)
            
            # Format the prompt with variables
            prompt = aspect_prompt.format(
                query=rag_input.query,
                context=contexts_str
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("completeness", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None

            response = await self.generate_llm_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=300
            )
            
            # Parse aspects from response
            aspects = []
            for line in response.content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets and extract aspect
                    aspect = line.split('.', 1)[-1].strip()
                    if aspect and len(aspect) > 5:  # Filter out very short responses
                        aspects.append(aspect)
            
            return aspects[:5]  # Limit to 5 aspects for practical evaluation
            
        except Exception as e:
            self.logger.error(f"Failed to identify query aspects: {str(e)}")
            return []

    async def _evaluate_aspect_coverage(
        self, 
        query: str, 
        aspect: str, 
        answer: str,
        contexts: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate how well a specific aspect is covered in the answer.
        
        Args:
            query: The original query
            aspect: The specific aspect to evaluate
            answer: The generated answer
            contexts: The retrieved contexts
            
        Returns:
            Dictionary with coverage assessment
        """
        try:
            # Get the coverage prompt from YAML
            coverage_prompt = get_prompt("completeness", "coverage_prompt")
            
            # Format the prompt with variables
            prompt = coverage_prompt.format(
                query=query,
                answer=answer,
                aspects=aspect  # Single aspect for this evaluation
            )

            # Get system prompt
            try:
                system_prompt_template = get_prompt("completeness", "system_prompt")
                system_prompt = system_prompt_template.format()
            except:
                system_prompt = None

            response = await self.generate_llm_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=300
            )
            
            # Parse the response for coverage assessment
            response_lines = response.content.split('\n')
            
            addressed = False
            thoroughness = "Basic"
            quality = "Fair"
            explanation = "No detailed explanation provided"
            
            for line in response_lines:
                line = line.strip().lower()
                if "addressed:" in line:
                    addressed = "yes" in line
                elif "thoroughness:" in line:
                    if "comprehensive" in line:
                        thoroughness = "Comprehensive"
                    elif "adequate" in line:
                        thoroughness = "Adequate"
                    else:
                        thoroughness = "Basic"
                elif "quality:" in line:
                    if "excellent" in line:
                        quality = "Excellent"
                    elif "good" in line:
                        quality = "Good"
                    elif "poor" in line:
                        quality = "Poor"
                    else:
                        quality = "Fair"
                elif "explanation:" in line:
                    explanation = line.split("explanation:", 1)[-1].strip()
            
            # Calculate score based on coverage assessment
            score = 0.0
            if addressed:
                score += 0.4  # Base score for being addressed
                
                if thoroughness == "Comprehensive":
                    score += 0.4
                elif thoroughness == "Adequate":
                    score += 0.2
                
                if quality == "Excellent":
                    score += 0.2
                elif quality == "Good":
                    score += 0.1
            
            return {
                "score": min(score, 1.0),
                "addressed": addressed,
                "thoroughness": thoroughness,
                "quality": quality,
                "explanation": explanation
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate aspect coverage: {str(e)}")
            return {
                "score": 0.0,
                "addressed": False,
                "thoroughness": "Unknown",
                "quality": "Unknown",
                "explanation": f"Error evaluating aspect: {str(e)}"
            } 