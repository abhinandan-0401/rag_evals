"""Comprehensive tests for all RAG evaluation metrics."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from rag_evals.core.types import RAGInput, MetricResult, MetricType
from rag_evals.metrics import (
    Faithfulness, AnswerRelevance, Coherence, 
    ContextRelevance, ContextPrecision,
    TrustScore, LLMJudge
)
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


# ===== GENERATION METRICS TESTS =====

@pytest.mark.asyncio
async def test_faithfulness_metric(mock_llm_provider, sample_rag_input):
    """Test the Faithfulness metric."""
    mock_llm_provider.generate = AsyncMock(return_value=MagicMock(
        content="Score: 0.95\nExplanation: The answer is well-supported by the contexts."
    ))
    
    metric = Faithfulness(llm_provider=mock_llm_provider)
    result = await metric.evaluate(sample_rag_input)
    
    assert isinstance(result, MetricResult)
    assert result.score == 0.95
    assert "well-supported" in result.explanation.lower()
    assert metric.metric_type == MetricType.GENERATION


@pytest.mark.asyncio
async def test_faithfulness_no_contexts(mock_llm_provider):
    """Test Faithfulness with no contexts."""
    metric = Faithfulness(llm_provider=mock_llm_provider)
    
    rag_input = RAGInput(
        query="What is the capital of France?",
        retrieved_contexts=["dummy"],
        answer="The capital of France is Paris."
    )
    rag_input.retrieved_contexts = []  # Manually set empty contexts
    
    result = await metric.evaluate(rag_input)
    
    assert result.score == 0.0
    assert "no contexts" in result.explanation.lower()
    mock_llm_provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_faithfulness_empty_answer(mock_llm_provider, sample_rag_input):
    """Test Faithfulness with empty answer."""
    metric = Faithfulness(llm_provider=mock_llm_provider)
    sample_rag_input.answer = "   "  # Empty answer
    
    result = await metric.evaluate(sample_rag_input)
    
    assert result.score == 0.0
    assert "empty answer" in result.explanation.lower()
    mock_llm_provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_answer_relevance_metric(mock_llm_provider, sample_rag_input):
    """Test the AnswerRelevance metric."""
    mock_llm_provider.generate = AsyncMock(return_value=MagicMock(
        content="Score: 0.90\nExplanation: The answer directly addresses the question."
    ))
    
    metric = AnswerRelevance(llm_provider=mock_llm_provider)
    result = await metric.evaluate(sample_rag_input)
    
    assert isinstance(result, MetricResult)
    assert result.score == 0.90
    assert metric.metric_type == MetricType.GENERATION


@pytest.mark.asyncio
async def test_answer_relevance_irrelevant_answer(mock_llm_provider):
    """Test AnswerRelevance with irrelevant answer."""
    mock_llm_provider.generate = AsyncMock(return_value=MagicMock(
        content="Score: 0.1\nExplanation: The answer does not address the question."
    ))
    
    metric = AnswerRelevance(llm_provider=mock_llm_provider)
    rag_input = RAGInput(
        query="What is the capital of France?",
        retrieved_contexts=["Paris is the capital of France."],
        answer="Cats are mammals that like to sleep."
    )
    
    result = await metric.evaluate(rag_input)
    
    assert result.score == 0.1
    assert "not address" in result.explanation.lower()


@pytest.mark.asyncio
async def test_coherence_metric(mock_llm_provider, sample_rag_input):
    """Test the Coherence metric."""
    mock_llm_provider.generate = AsyncMock(return_value=MagicMock(
        content="Score: 0.85\nExplanation: The answer is well-structured and coherent."
    ))
    
    metric = Coherence(llm_provider=mock_llm_provider)
    result = await metric.evaluate(sample_rag_input)
    
    assert isinstance(result, MetricResult)
    assert result.score == 0.85
    assert metric.metric_type == MetricType.GENERATION


@pytest.mark.asyncio
async def test_coherence_incoherent_answer(mock_llm_provider):
    """Test Coherence with incoherent answer."""
    mock_llm_provider.generate = AsyncMock(return_value=MagicMock(
        content="Score: 0.2\nExplanation: The answer is confusing and poorly structured."
    ))
    
    metric = Coherence(llm_provider=mock_llm_provider)
    rag_input = RAGInput(
        query="What is the capital of France?",
        retrieved_contexts=["Paris is the capital of France."],
        answer="France... capital... Paris maybe? Weather sunny today."
    )
    
    result = await metric.evaluate(rag_input)
    
    assert result.score == 0.2
    assert "confusing" in result.explanation.lower()


# ===== RETRIEVAL METRICS TESTS =====

@pytest.mark.asyncio
async def test_context_relevance_metric(mock_llm_provider, sample_rag_input):
    """Test the ContextRelevance metric."""
    # Mock scores for 3 contexts: 1.0, 0.5, 0.8
    mock_llm_provider.generate = AsyncMock(side_effect=[
        MagicMock(content="Score: 1.0\nExplanation: Perfectly relevant."),
        MagicMock(content="Score: 0.5\nExplanation: Somewhat relevant."),
        MagicMock(content="Score: 0.8\nExplanation: Mostly relevant.")
    ])
    
    metric = ContextRelevance(llm_provider=mock_llm_provider)
    result = await metric.evaluate(sample_rag_input)
    
    assert isinstance(result, MetricResult)
    # Expected average: (1.0 + 0.5 + 0.8) / 3 = 0.7667
    assert abs(result.score - 0.7666666666666667) < 0.001
    assert metric.metric_type == MetricType.RETRIEVAL
    assert "context_scores" in result.details
    assert len(result.details["context_scores"]) == 3


@pytest.mark.asyncio
async def test_context_relevance_empty_contexts(mock_llm_provider):
    """Test ContextRelevance with no contexts."""
    metric = ContextRelevance(llm_provider=mock_llm_provider)
    
    rag_input = RAGInput(
        query="What is the capital of France?",
        retrieved_contexts=["dummy"],
        answer="The capital of France is Paris."
    )
    rag_input.retrieved_contexts = []  # Manually set empty contexts
    
    result = await metric.evaluate(rag_input)
    
    assert result.score == 0.0
    assert "no contexts" in result.explanation.lower()
    mock_llm_provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_context_precision_perfect_ranking(mock_llm_provider):
    """Test ContextPrecision with perfect ranking."""
    mock_llm_provider.generate = AsyncMock(return_value=MagicMock(
        content="""Context 1: Yes
Context 2: Yes
Context 3: No

Overall Precision Score: 1.0
Explanation: Perfect ranking with relevant contexts appearing first."""
    ))
    
    metric = ContextPrecision(llm_provider=mock_llm_provider)
    rag_input = RAGInput(
        query="What is the capital of France?",
        retrieved_contexts=[
            "Paris is the capital of France.",
            "The population of Paris is about 2.2 million.",
            "Photosynthesis is a process used by plants."
        ],
        answer="The capital of France is Paris."
    )
    
    result = await metric.evaluate(rag_input)
    
    assert result.score == 1.0
    assert metric.metric_type == MetricType.RETRIEVAL
    assert "relevance_assessments" in result.details
    assert result.details["relevant_contexts"] == 2


@pytest.mark.asyncio
async def test_context_precision_poor_ranking(mock_llm_provider):
    """Test ContextPrecision with moderate ranking (3 contexts, relevant at end)."""
    mock_llm_provider.generate = AsyncMock(return_value=MagicMock(
        content="""Context 1: No
Context 2: No
Context 3: Yes

Overall Precision Score: 0.33
Explanation: Moderate ranking with relevant context appearing last."""
    ))
    
    metric = ContextPrecision(llm_provider=mock_llm_provider)
    rag_input = RAGInput(
        query="What is the capital of France?",
        retrieved_contexts=[
            "Photosynthesis is a process used by plants.",
            "The weather today is sunny.",
            "Paris is the capital of France."
        ],
        answer="The capital of France is Paris."
    )
    
    result = await metric.evaluate(rag_input)
    
    assert abs(result.score - 0.3333333333333333) < 0.001
    assert result.details["relevant_contexts"] == 1
    assert result.details["ranking_quality"] == "moderate_ranking"


@pytest.mark.asyncio
async def test_context_precision_single_context(mock_llm_provider):
    """Test ContextPrecision with single context."""
    metric = ContextPrecision(llm_provider=mock_llm_provider)
    rag_input = RAGInput(
        query="What is the capital of France?",
        retrieved_contexts=["Paris is the capital of France."],
        answer="The capital of France is Paris."
    )
    
    result = await metric.evaluate(rag_input)
    
    assert result.score == 1.0
    assert result.details["ranking_applicable"] == False
    mock_llm_provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_context_precision_truly_poor_ranking(mock_llm_provider):
    """Test ContextPrecision with truly poor ranking (5 contexts, relevant at very end)."""
    mock_llm_provider.generate = AsyncMock(return_value=MagicMock(
        content="""Context 1: No
Context 2: No
Context 3: No
Context 4: No
Context 5: Yes

Overall Precision Score: 0.2
Explanation: Poor ranking with relevant context appearing very last."""
    ))
    
    metric = ContextPrecision(llm_provider=mock_llm_provider)
    rag_input = RAGInput(
        query="What is the capital of France?",
        retrieved_contexts=[
            "Photosynthesis is a process used by plants.",
            "The weather today is sunny.",
            "Cats are mammals that like to sleep.",
            "The ocean is deep and blue.",
            "Paris is the capital of France."
        ],
        answer="The capital of France is Paris."
    )
    
    result = await metric.evaluate(rag_input)
    
    assert result.score == 0.2
    assert result.details["relevant_contexts"] == 1
    assert result.details["ranking_quality"] == "poor_ranking"


# ===== INTEGRATION TESTS =====

@pytest.mark.asyncio
async def test_all_metrics_together(mock_llm_provider, sample_rag_input):
    """Test that all metrics can be used together."""
    # Setup different responses for different metrics
    responses = [
        MagicMock(content="Score: 0.9\nExplanation: Faithful answer."),  # Faithfulness
        MagicMock(content="Score: 0.85\nExplanation: Relevant answer."),  # AnswerRelevance
        MagicMock(content="Score: 0.8\nExplanation: Coherent answer."),  # Coherence
        MagicMock(content="Score: 0.9\nExplanation: Relevant context."),  # ContextRelevance (context 1)
        MagicMock(content="Score: 0.7\nExplanation: Relevant context."),  # ContextRelevance (context 2)
        MagicMock(content="Score: 0.6\nExplanation: Relevant context."),  # ContextRelevance (context 3)
        MagicMock(content="""Context 1: Yes
Context 2: Yes
Context 3: Yes

Overall Precision Score: 1.0
Explanation: Perfect ranking."""),  # ContextPrecision
        # TrustScore responses
        MagicMock(content="B"),  # Behavioral consistency
        MagicMock(content="support"),  # Factual consistency
        # LLMJudge responses (5 criteria)
        MagicMock(content="SCORE: 4\nEXPLANATION: Good relevance."),
        MagicMock(content="SCORE: 5\nEXPLANATION: Accurate."),
        MagicMock(content="SCORE: 4\nEXPLANATION: Complete."),
        MagicMock(content="SCORE: 4\nEXPLANATION: Clear."),
        MagicMock(content="SCORE: 5\nEXPLANATION: Well grounded.")
    ]
    
    mock_llm_provider.generate = AsyncMock(side_effect=responses)
    
    # Test all metrics including composite ones
    metrics = [
        Faithfulness(llm_provider=mock_llm_provider),
        AnswerRelevance(llm_provider=mock_llm_provider),
        Coherence(llm_provider=mock_llm_provider),
        ContextRelevance(llm_provider=mock_llm_provider),
        ContextPrecision(llm_provider=mock_llm_provider),
        TrustScore(llm_provider=mock_llm_provider, max_checks=1),
        LLMJudge(llm_provider=mock_llm_provider)
    ]
    
    results = []
    for metric in metrics:
        result = await metric.evaluate(sample_rag_input)
        results.append(result)
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0
    
    # Verify we got results from all metrics
    assert len(results) == 7
    assert all(isinstance(r, MetricResult) for r in results)


def test_metric_names():
    """Test that all metrics have correct names."""
    mock_provider = MagicMock()
    
    metrics = [
        Faithfulness(llm_provider=mock_provider),
        AnswerRelevance(llm_provider=mock_provider),
        Coherence(llm_provider=mock_provider),
        ContextRelevance(llm_provider=mock_provider),
        ContextPrecision(llm_provider=mock_provider),
        TrustScore(llm_provider=mock_provider),
        LLMJudge(llm_provider=mock_provider)
    ]
    
    expected_names = [
        "faithfulness",
        "answer_relevance",
        "coherence",
        "context_relevance",
        "context_precision",
        "trust_score",
        "llm_judge"
    ]
    
    for metric, expected_name in zip(metrics, expected_names):
        assert metric.name() == expected_name


def test_metric_types():
    """Test that metrics have correct types."""
    mock_provider = MagicMock()
    
    generation_metrics = [
        Faithfulness(llm_provider=mock_provider),
        AnswerRelevance(llm_provider=mock_provider),
        Coherence(llm_provider=mock_provider)
    ]
    
    retrieval_metrics = [
        ContextRelevance(llm_provider=mock_provider),
        ContextPrecision(llm_provider=mock_provider)
    ]
    
    composite_metrics = [
        TrustScore(llm_provider=mock_provider),
        LLMJudge(llm_provider=mock_provider)
    ]
    
    for metric in generation_metrics:
        assert metric.metric_type == MetricType.GENERATION
    
    for metric in retrieval_metrics:
        assert metric.metric_type == MetricType.RETRIEVAL
    
    for metric in composite_metrics:
        assert metric.metric_type == MetricType.COMPOSITE 