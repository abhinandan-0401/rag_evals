"""Comprehensive tests for composite RAG evaluation metrics."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from rag_evals.core.types import RAGInput, MetricResult, MetricType
from rag_evals.metrics.composite import TrustScore, LLMJudge
from rag_evals.metrics.composite.llm_judge import EvaluationCriterion
from rag_evals.metrics.composite.trust_score import FactualConsistency
from rag_evals.llm.providers import LLMProvider


@pytest.fixture
def mock_llm_provider():
    """Fixture for a mock LLM provider."""
    provider = MagicMock(spec=LLMProvider)
    return provider


@pytest.fixture
def sample_rag_input():
    """Fixture for sample RAG input."""
    return RAGInput(
        query="What is the capital of France?",
        retrieved_contexts=[
            "Paris is the capital of France.",
            "The Eiffel Tower is located in Paris.",
            "France is a country in Europe."
        ],
        answer="The capital of France is Paris."
    )


# ===== TRUSTSCORE TESTS =====

@pytest.mark.asyncio
async def test_trust_score_high_trust(mock_llm_provider, sample_rag_input):
    """Test TrustScore with consistent behavior and supporting evidence."""
    # Mock responses for behavioral consistency (always choose B - original answer)
    mc_responses = [
        MagicMock(content="B"),  # Consistent choice
        MagicMock(content="B"),  # Consistent choice
        MagicMock(content="B"),  # Consistent choice
    ]
    
    # Mock response for factual consistency (support)
    fc_response = MagicMock(content="support")
    
    # Set up side effects for different types of calls
    mock_llm_provider.generate = AsyncMock(side_effect=[
        *mc_responses,  # Behavioral consistency checks
        fc_response     # Factual consistency check
    ])
    
    metric = TrustScore(llm_provider=mock_llm_provider, max_checks=3)
    result = await metric.evaluate(sample_rag_input)
    
    assert isinstance(result, MetricResult)
    assert result.score == 1.0  # Consistent + Support = 1.0
    assert metric.metric_type == MetricType.COMPOSITE
    assert "behavioral_consistency" in result.details
    assert "factual_consistency" in result.details
    assert result.details["behavioral_consistency"]["score"] == 1.0
    assert result.details["factual_consistency"]["result"] == "support"


@pytest.mark.asyncio
async def test_trust_score_low_trust(mock_llm_provider, sample_rag_input):
    """Test TrustScore with inconsistent behavior and contradicting evidence."""
    # Mock inconsistent behavioral response (chooses A instead of B)
    mc_response = MagicMock(content="A")
    
    # Mock contradicting factual consistency
    fc_response = MagicMock(content="contradict")
    
    mock_llm_provider.generate = AsyncMock(side_effect=[
        mc_response,    # First behavioral check fails
        fc_response     # Factual consistency check
    ])
    
    metric = TrustScore(llm_provider=mock_llm_provider, max_checks=3)
    result = await metric.evaluate(sample_rag_input)
    
    assert result.score == 0.0  # Inconsistent + Contradict = 0.0
    assert result.details["behavioral_consistency"]["score"] == 0.0
    assert result.details["factual_consistency"]["result"] == "contradict"


@pytest.mark.asyncio
async def test_trust_score_moderate_trust(mock_llm_provider, sample_rag_input):
    """Test TrustScore with consistent behavior and neutral evidence."""
    # Mock consistent behavioral responses
    mc_responses = [MagicMock(content="B") for _ in range(3)]
    
    # Mock neutral factual consistency
    fc_response = MagicMock(content="neutral")
    
    mock_llm_provider.generate = AsyncMock(side_effect=[
        *mc_responses,
        fc_response
    ])
    
    metric = TrustScore(llm_provider=mock_llm_provider, max_checks=3)
    result = await metric.evaluate(sample_rag_input)
    
    assert result.score == 0.6  # Consistent + Neutral = 0.6
    assert "trust_rubric" in result.details
    assert "Consistent + Neutral: 0.6" in result.details["trust_rubric"]


@pytest.mark.asyncio
async def test_trust_score_empty_answer(mock_llm_provider):
    """Test TrustScore with effectively empty answer."""
    rag_input = RAGInput(
        query="What is the capital of France?",
        retrieved_contexts=["Paris is the capital of France."],
        answer="The capital of France is Paris."
    )
    # Manually modify after validation to simulate empty answer processing
    rag_input.answer = "   "  # Empty answer after creation
    
    # Mock factual consistency for empty answer
    fc_response = MagicMock(content="neutral")
    mock_llm_provider.generate = AsyncMock(return_value=fc_response)
    
    metric = TrustScore(llm_provider=mock_llm_provider)
    result = await metric.evaluate(rag_input)
    
    # Empty answer should result in 0.0 behavioral consistency
    assert result.details["behavioral_consistency"]["score"] == 0.0
    assert result.details["behavioral_consistency"]["details"]["reason"] == "empty_answer"


@pytest.mark.asyncio
async def test_trust_score_no_contexts(mock_llm_provider):
    """Test TrustScore with no retrieved contexts."""
    rag_input = RAGInput(
        query="What is the capital of France?",
        retrieved_contexts=["dummy"],
        answer="The capital of France is Paris."
    )
    rag_input.retrieved_contexts = []  # Manually set empty contexts
    
    # Mock behavioral consistency responses
    mc_responses = [MagicMock(content="B") for _ in range(3)]
    mock_llm_provider.generate = AsyncMock(side_effect=mc_responses)
    
    metric = TrustScore(llm_provider=mock_llm_provider)
    result = await metric.evaluate(rag_input)
    
    # No contexts should result in neutral factual consistency
    assert result.details["factual_consistency"]["result"] == "neutral"
    assert result.details["factual_consistency"]["details"]["reason"] == "no_contexts"


# ===== LLM JUDGE TESTS =====

@pytest.mark.asyncio
async def test_llm_judge_high_scores(mock_llm_provider, sample_rag_input):
    """Test LLM Judge with high scores across all criteria."""
    # Mock high scores for all default criteria (5 criteria)
    criteria_responses = [
        MagicMock(content="FINAL SCORE: 5\nEXPLANATION: Excellent relevance."),
        MagicMock(content="FINAL SCORE: 5\nEXPLANATION: Completely accurate."),
        MagicMock(content="FINAL SCORE: 4\nEXPLANATION: Very complete response."),
        MagicMock(content="FINAL SCORE: 5\nEXPLANATION: Very clear and readable."),
        MagicMock(content="FINAL SCORE: 5\nEXPLANATION: Well grounded in contexts.")
    ]
    
    mock_llm_provider.generate = AsyncMock(side_effect=criteria_responses)
    
    metric = LLMJudge(llm_provider=mock_llm_provider)
    result = await metric.evaluate(sample_rag_input)
    
    assert isinstance(result, MetricResult)
    assert result.score > 0.9  # Should be very high
    assert metric.metric_type == MetricType.COMPOSITE
    assert "criteria_scores" in result.details
    assert len(result.details["criteria_scores"]) == 5  # Default criteria count
    assert "overall_calculation" in result.details


@pytest.mark.asyncio
async def test_llm_judge_low_scores(mock_llm_provider, sample_rag_input):
    """Test LLM Judge with low scores across all criteria."""
    # Mock low scores for all criteria
    criteria_responses = [
        MagicMock(content="SCORE: 1\nEXPLANATION: Poor relevance."),
        MagicMock(content="SCORE: 2\nEXPLANATION: Some inaccuracies."),
        MagicMock(content="SCORE: 1\nEXPLANATION: Incomplete response."),
        MagicMock(content="SCORE: 2\nEXPLANATION: Unclear writing."),
        MagicMock(content="SCORE: 1\nEXPLANATION: Not well grounded.")
    ]
    
    mock_llm_provider.generate = AsyncMock(side_effect=criteria_responses)
    
    metric = LLMJudge(llm_provider=mock_llm_provider)
    result = await metric.evaluate(sample_rag_input)
    
    assert result.score < 0.4  # Should be low
    assert "Very poor quality" in result.explanation


@pytest.mark.asyncio
async def test_llm_judge_custom_criteria(mock_llm_provider, sample_rag_input):
    """Test LLM Judge with custom evaluation criteria."""
    custom_criteria = [
        EvaluationCriterion(
            name="creativity",
            description="How creative and original is the response?",
            scale_min=1,
            scale_max=10,
            weight=2.0
        ),
        EvaluationCriterion(
            name="brevity",
            description="How concise is the response?",
            scale_min=1,
            scale_max=5,
            weight=1.0
        )
    ]
    
    # Mock responses for custom criteria
    responses = [
        MagicMock(content="FINAL SCORE: 8\nEXPLANATION: Very creative."),
        MagicMock(content="FINAL SCORE: 4\nEXPLANATION: Good brevity.")
    ]
    
    mock_llm_provider.generate = AsyncMock(side_effect=responses)
    
    metric = LLMJudge(
        llm_provider=mock_llm_provider,
        criteria=custom_criteria
    )
    result = await metric.evaluate(sample_rag_input)
    
    assert len(result.details["criteria_scores"]) == 2
    assert "creativity" in result.details["criteria_scores"]
    assert "brevity" in result.details["criteria_scores"]
    
    # Check weighted calculation
    creativity_weight = result.details["criteria_scores"]["creativity"]["weight"]
    brevity_weight = result.details["criteria_scores"]["brevity"]["weight"]
    assert creativity_weight == 2.0
    assert brevity_weight == 1.0


@pytest.mark.asyncio
async def test_llm_judge_chain_of_thought_disabled(mock_llm_provider, sample_rag_input):
    """Test LLM Judge with chain-of-thought prompting disabled."""
    response = MagicMock(content="SCORE: 3\nEXPLANATION: Average quality.")
    mock_llm_provider.generate = AsyncMock(return_value=response)
    
    metric = LLMJudge(
        llm_provider=mock_llm_provider,
        criteria=[EvaluationCriterion(name="test", description="Test criterion")],
        use_chain_of_thought=False
    )
    result = await metric.evaluate(sample_rag_input)
    
    assert result.details["chain_of_thought"] == False
    # Verify the prompt doesn't include chain-of-thought instructions
    call_args = mock_llm_provider.generate.call_args[0][0]
    assert "CHAIN OF THOUGHT" not in call_args


@pytest.mark.asyncio
async def test_llm_judge_score_parsing_edge_cases(mock_llm_provider, sample_rag_input):
    """Test LLM Judge score parsing with various response formats."""
    # Test different score formats
    responses = [
        MagicMock(content="Score: 3.5 out of 5"),  # Decimal score
        MagicMock(content="The score is 4"),       # Embedded score
        MagicMock(content="Invalid response"),     # No score found
        MagicMock(content="Score: 10"),            # Out of range score
        MagicMock(content="Score: 0")              # Below range score
    ]
    
    mock_llm_provider.generate = AsyncMock(side_effect=responses)
    
    # Create 5 different criteria to avoid naming conflicts
    criteria = [
        EvaluationCriterion(name=f"test_{i}", description="Test", scale_min=1, scale_max=5)
        for i in range(5)
    ]
    metric = LLMJudge(
        llm_provider=mock_llm_provider,
        criteria=criteria
    )
    result = await metric.evaluate(sample_rag_input)
    
    criteria_scores = result.details["criteria_scores"]
    
    # Check that scores are properly parsed and clamped
    scores = [criteria_scores[key]["raw_score"] for key in criteria_scores.keys()]
    
    assert 3.5 in scores    # Decimal parsed correctly
    assert 4 in scores      # Embedded score parsed
    assert 1 in scores      # Default to minimum when no score found
    assert 5 in scores      # Clamped to maximum
    assert 1 in scores      # Clamped to minimum


# ===== INTEGRATION TESTS =====

@pytest.mark.asyncio
async def test_composite_metrics_together(mock_llm_provider, sample_rag_input):
    """Test that both composite metrics can be used together."""
    # Setup responses for TrustScore
    trust_responses = [
        MagicMock(content="B"),         # Behavioral consistency
        MagicMock(content="support")    # Factual consistency
    ]
    
    # Setup responses for LLMJudge (5 default criteria)
    judge_responses = [
        MagicMock(content="SCORE: 4\nEXPLANATION: Good relevance."),
        MagicMock(content="SCORE: 5\nEXPLANATION: Accurate."),
        MagicMock(content="SCORE: 4\nEXPLANATION: Complete."),
        MagicMock(content="SCORE: 4\nEXPLANATION: Clear."),
        MagicMock(content="SCORE: 5\nEXPLANATION: Well grounded.")
    ]
    
    # First calls are for TrustScore, then LLMJudge
    all_responses = trust_responses + judge_responses
    mock_llm_provider.generate = AsyncMock(side_effect=all_responses)
    
    # Test both metrics
    trust_metric = TrustScore(llm_provider=mock_llm_provider, max_checks=1)
    judge_metric = LLMJudge(llm_provider=mock_llm_provider)
    
    trust_result = await trust_metric.evaluate(sample_rag_input)
    judge_result = await judge_metric.evaluate(sample_rag_input)
    
    # Both should succeed
    assert isinstance(trust_result, MetricResult)
    assert isinstance(judge_result, MetricResult)
    assert trust_result.score > 0
    assert judge_result.score > 0
    assert trust_metric.metric_type == MetricType.COMPOSITE
    assert judge_metric.metric_type == MetricType.COMPOSITE


def test_composite_metric_names():
    """Test that composite metrics have correct names."""
    mock_provider = MagicMock()
    
    trust_metric = TrustScore(llm_provider=mock_provider)
    judge_metric = LLMJudge(llm_provider=mock_provider)
    
    assert trust_metric.name() == "trust_score"
    assert judge_metric.name() == "llm_judge"


def test_evaluation_criterion_validation():
    """Test validation of evaluation criteria."""
    mock_provider = MagicMock()
    
    # Test invalid scale
    with pytest.raises(ValueError, match="Invalid scale"):
        invalid_criterion = EvaluationCriterion(
            name="test",
            description="Test",
            scale_min=5,
            scale_max=3  # Invalid: min > max
        )
        LLMJudge(llm_provider=mock_provider, criteria=[invalid_criterion])
    
    # Test invalid weight
    with pytest.raises(ValueError, match="Weight must be positive"):
        invalid_criterion = EvaluationCriterion(
            name="test",
            description="Test",
            weight=-1.0  # Invalid: negative weight
        )
        LLMJudge(llm_provider=mock_provider, criteria=[invalid_criterion])


@pytest.mark.asyncio
async def test_composite_metrics_error_handling(mock_llm_provider, sample_rag_input):
    """Test error handling in composite metrics."""
    # Mock LLM provider that raises an exception
    mock_llm_provider.generate = AsyncMock(side_effect=Exception("LLM API Error"))
    
    trust_metric = TrustScore(llm_provider=mock_llm_provider)
    judge_metric = LLMJudge(llm_provider=mock_llm_provider)
    
    trust_result = await trust_metric.evaluate(sample_rag_input)
    judge_result = await judge_metric.evaluate(sample_rag_input)
    
    # Both should handle errors gracefully
    # TrustScore may still produce a partial score based on what succeeded
    assert isinstance(trust_result.score, float)
    assert 0.0 <= trust_result.score <= 1.0
    assert judge_result.score == 0.0  # LLMJudge should give 0 score when all criteria fail
    assert "trust" in trust_result.explanation.lower() or "failed" in trust_result.explanation.lower()
    
    # Check that error information is captured in the details
    trust_details = trust_result.details
    judge_details = judge_result.details
    
    # TrustScore: At least one component should have error info
    has_error_info = (
        "error" in trust_details or 
        any("error" in str(v) for v in trust_details.values() if isinstance(v, dict))
    )
    assert has_error_info
    
    # LLMJudge: Check that individual criteria show error messages
    criteria_scores = judge_details["criteria_scores"]
    error_explanations = [
        result["explanation"] 
        for result in criteria_scores.values()
    ]
    assert any("failed" in exp.lower() for exp in error_explanations) 