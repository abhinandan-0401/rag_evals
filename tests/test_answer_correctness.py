"""
Comprehensive tests for Answer Correctness metric with Azure OpenAI provider.
"""

import pytest
import os
from unittest.mock import AsyncMock, MagicMock
from rag_evals.metrics.generation.answer_correctness import AnswerCorrectness
from rag_evals.core.types import RAGInput, MetricResult, MetricType
from rag_evals.llm.providers import AzureOpenAIProvider

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") 
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

def skip_if_no_azure_creds():
    """Skip test if Azure OpenAI credentials are not available."""
    return pytest.mark.skipif(
        not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT]),
        reason="Azure OpenAI credentials not available"
    )


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    provider = MagicMock()
    return provider


@pytest.fixture 
def azure_provider():
    """Real Azure OpenAI provider for integration tests."""
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT]):
        pytest.skip("Azure OpenAI credentials not available")
    
    return AzureOpenAIProvider(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION
    )


@pytest.fixture
def sample_correctness_scenarios():
    """Sample scenarios for testing answer correctness."""
    return {
        "highly_correct": RAGInput(
            query="What is the capital of France?",
            answer="The capital of France is Paris. Paris is located in the north-central part of the country and is known for its rich history, culture, and landmarks like the Eiffel Tower.",
            retrieved_contexts=[
                "Paris is the capital and most populous city of France.",
                "Paris is located in northern France on the river Seine.",
                "The city is known for its culture, architecture, and landmarks including the Eiffel Tower."
            ]
        ),
        "partially_correct": RAGInput(
            query="When was the Declaration of Independence signed?",
            answer="The Declaration of Independence was signed on July 4, 1776, in Philadelphia by George Washington and Benjamin Franklin. It established the United States as an independent nation from British rule.",
            retrieved_contexts=[
                "The Declaration of Independence was signed on July 4, 1776.",
                "The document was signed in Philadelphia, Pennsylvania.",
                "Benjamin Franklin was one of the signers, but George Washington was not present at the signing."
            ]
        ),
        "factually_incorrect": RAGInput(
            query="How many continents are there?",
            answer="There are 8 continents in the world: Africa, Asia, Europe, North America, South America, Australia, Antarctica, and Atlantis. Atlantis was discovered in 2010 under the Atlantic Ocean.",
            retrieved_contexts=[
                "There are seven continents: Africa, Antarctica, Asia, Europe, North America, Australia/Oceania, and South America.",
                "The seven-continent model is the most commonly taught in English-speaking countries.",
                "Atlantis is a fictional island mentioned in Plato's dialogues and has never been discovered."
            ]
        ),
        "logically_inconsistent": RAGInput(
            query="What is the speed of light?",
            answer="The speed of light is approximately 300,000 kilometers per second in a vacuum. However, light travels faster in water than in air because water is denser. This means light moves at about 400,000 km/s underwater.",
            retrieved_contexts=[
                "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
                "Light travels slower in denser media. In water, light travels at about 225,000 km/s.",
                "The speed of light is a fundamental constant in physics."
            ]
        )
    }


class TestAnswerCorrectnessMocked:
    """Test Answer Correctness metric with mocked LLM responses."""

    @pytest.mark.asyncio
    async def test_answer_correctness_basic_functionality(self, mock_llm_provider, sample_correctness_scenarios):
        """Test basic Answer Correctness functionality with mocked responses."""
        # Mock responses for all evaluation components
        mock_llm_provider.generate = AsyncMock(side_effect=[
            # Factual accuracy response
            MagicMock(content="Score: 0.95\nExplanation: All facts are accurate and well-supported"),
            # Context consistency response  
            MagicMock(content="Score: 0.90\nExplanation: Answer is consistent with provided contexts"),
            # Logical consistency response
            MagicMock(content="Score: 0.92\nExplanation: Answer is logically coherent and consistent")
        ])
        
        metric = AnswerCorrectness(llm_provider=mock_llm_provider)
        result = await metric.evaluate(sample_correctness_scenarios["highly_correct"])
        
        assert isinstance(result, MetricResult)
        assert 0.9 <= result.score <= 1.0  # Should be high for all good components
        assert metric.metric_type == MetricType.GENERATION
        assert metric.name() == "answer_correctness"
        assert "evaluation_breakdown" in result.details
        assert "components_evaluated" in result.details

    @pytest.mark.asyncio
    async def test_answer_correctness_factual_errors(self, mock_llm_provider, sample_correctness_scenarios):
        """Test Answer Correctness with factual errors."""
        # Mock responses indicating factual errors
        mock_llm_provider.generate = AsyncMock(side_effect=[
            # Factual accuracy response (low score for incorrect facts)
            MagicMock(content="Score: 0.2\nExplanation: Contains significant factual errors - 8 continents and Atlantis discovery are incorrect"),
            # Context consistency response (low due to contradictions)
            MagicMock(content="Score: 0.1\nExplanation: Answer contradicts context information about number of continents"),
            # Logical consistency response (moderate - internally consistent but wrong)
            MagicMock(content="Score: 0.6\nExplanation: Internally logical but based on incorrect premises")
        ])
        
        metric = AnswerCorrectness(llm_provider=mock_llm_provider)
        result = await metric.evaluate(sample_correctness_scenarios["factually_incorrect"])
        
        assert isinstance(result, MetricResult)
        assert result.score < 0.5  # Should be low for factual errors
        assert result.details["components_evaluated"] == 3
        
        # Check that factual accuracy scored low
        factual_score = result.details["evaluation_breakdown"]["factual_accuracy"]["score"]
        assert factual_score < 0.5

    @pytest.mark.asyncio
    async def test_answer_correctness_disabled_components(self, mock_llm_provider, sample_correctness_scenarios):
        """Test Answer Correctness with some components disabled."""
        # Mock response for only factual accuracy (other components disabled)
        mock_llm_provider.generate = AsyncMock(return_value=MagicMock(
            content="Score: 0.85\nExplanation: Good factual accuracy"
        ))
        
        metric = AnswerCorrectness(
            llm_provider=mock_llm_provider,
            use_context_verification=False,
            check_logical_consistency=False
        )
        result = await metric.evaluate(sample_correctness_scenarios["highly_correct"])
        
        assert isinstance(result, MetricResult)
        assert result.details["components_evaluated"] == 1  # Only factual accuracy
        assert "factual_accuracy" in result.details["evaluation_breakdown"]
        assert "context_consistency" not in result.details["evaluation_breakdown"]
        assert "logical_consistency" not in result.details["evaluation_breakdown"]

    @pytest.mark.asyncio
    async def test_answer_correctness_no_contexts(self, mock_llm_provider):
        """Test Answer Correctness with no retrieved contexts."""
        # Create scenario without contexts
        no_contexts_input = RAGInput(
            query="What is AI?",
            answer="Artificial intelligence is the simulation of human intelligence in machines.",
            retrieved_contexts=["dummy"]  # Start with dummy to pass validation
        )
        no_contexts_input.retrieved_contexts = []  # Then set to empty
        
        # Mock responses for factual accuracy and logical consistency only
        mock_llm_provider.generate = AsyncMock(side_effect=[
            MagicMock(content="Score: 0.80\nExplanation: Factually accurate definition"),
            MagicMock(content="Score: 0.85\nExplanation: Logically consistent")
        ])
        
        metric = AnswerCorrectness(llm_provider=mock_llm_provider)
        result = await metric.evaluate(no_contexts_input)
        
        assert isinstance(result, MetricResult)
        assert result.details["components_evaluated"] == 2  # No context consistency
        assert "context_consistency" not in result.details["evaluation_breakdown"]

    @pytest.mark.asyncio
    async def test_answer_correctness_empty_answer(self, mock_llm_provider):
        """Test Answer Correctness with empty answer."""
        rag_input = RAGInput(
            query="What is AI?",
            answer="dummy",  # Start with dummy to pass validation
            retrieved_contexts=["AI is artificial intelligence."]
        )
        rag_input.answer = ""  # Then set to empty
        
        metric = AnswerCorrectness(llm_provider=mock_llm_provider)
        result = await metric.evaluate(rag_input)
        
        assert result.score == 0.0
        assert "Empty answer provided" in result.explanation
        assert result.details["error"] == "empty_answer"

    @pytest.mark.asyncio
    async def test_answer_correctness_explanation_generation(self, mock_llm_provider, sample_correctness_scenarios):
        """Test Answer Correctness explanation generation."""
        # Mock different scores for different components
        mock_llm_provider.generate = AsyncMock(side_effect=[
            MagicMock(content="Score: 0.9\nExplanation: Excellent factual accuracy"),
            MagicMock(content="Score: 0.7\nExplanation: Good context consistency"),
            MagicMock(content="Score: 0.8\nExplanation: Good logical consistency")
        ])
        
        metric = AnswerCorrectness(llm_provider=mock_llm_provider)
        result = await metric.evaluate(sample_correctness_scenarios["highly_correct"])
        
        assert isinstance(result, MetricResult)
        assert "mostly correct" in result.explanation or "highly correct" in result.explanation
        assert "Factual Accuracy: excellent" in result.explanation
        assert "Context Consistency: good" in result.explanation
        assert "Logical Consistency: excellent" in result.explanation  # 0.8 is excellent (>= 0.8)


class TestAnswerCorrectnessIntegration:
    """Integration tests with real Azure OpenAI API."""

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_answer_correctness_highly_correct_scenario(self, azure_provider, sample_correctness_scenarios):
        """Test Answer Correctness with highly correct answer using real API."""
        metric = AnswerCorrectness(llm_provider=azure_provider)
        result = await metric.evaluate(sample_correctness_scenarios["highly_correct"])
        
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0
        assert result.details["components_evaluated"] >= 2
        
        print(f"\n[INFO] Highly correct scenario:")
        print(f"[INFO] Query: {sample_correctness_scenarios['highly_correct'].query}")
        print(f"[INFO] Answer: {sample_correctness_scenarios['highly_correct'].answer[:80]}...")
        print(f"[INFO] Overall score: {result.score:.3f}")
        print(f"[INFO] Components evaluated: {result.details['components_evaluated']}")
        print(f"[INFO] Explanation: {result.explanation}")
        
        # Should have high score for correct answer
        assert result.score > 0.7

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_answer_correctness_partially_correct_scenario(self, azure_provider, sample_correctness_scenarios):
        """Test Answer Correctness with partially correct answer using real API."""
        metric = AnswerCorrectness(llm_provider=azure_provider)
        result = await metric.evaluate(sample_correctness_scenarios["partially_correct"])
        
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0
        
        print(f"\n[INFO] Partially correct scenario:")
        print(f"[INFO] Query: {sample_correctness_scenarios['partially_correct'].query}")
        print(f"[INFO] Answer: {sample_correctness_scenarios['partially_correct'].answer[:80]}...")
        print(f"[INFO] Overall score: {result.score:.3f}")
        print(f"[INFO] Explanation: {result.explanation}")
        
        # Should have moderate score (some errors but mostly correct)
        assert 0.4 <= result.score <= 0.9

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_answer_correctness_factually_incorrect_scenario(self, azure_provider, sample_correctness_scenarios):
        """Test Answer Correctness with factually incorrect answer using real API."""
        metric = AnswerCorrectness(llm_provider=azure_provider)
        result = await metric.evaluate(sample_correctness_scenarios["factually_incorrect"])
        
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0
        
        print(f"\n[INFO] Factually incorrect scenario:")
        print(f"[INFO] Query: {sample_correctness_scenarios['factually_incorrect'].query}")
        print(f"[INFO] Answer: {sample_correctness_scenarios['factually_incorrect'].answer[:80]}...")
        print(f"[INFO] Overall score: {result.score:.3f}")
        print(f"[INFO] Explanation: {result.explanation}")
        
        # Should have low score for factual errors
        assert result.score < 0.7

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_answer_correctness_logically_inconsistent_scenario(self, azure_provider, sample_correctness_scenarios):
        """Test Answer Correctness with logically inconsistent answer using real API."""
        metric = AnswerCorrectness(llm_provider=azure_provider)
        result = await metric.evaluate(sample_correctness_scenarios["logically_inconsistent"])
        
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0
        
        print(f"\n[INFO] Logically inconsistent scenario:")
        print(f"[INFO] Query: {sample_correctness_scenarios['logically_inconsistent'].query}")
        print(f"[INFO] Answer: {sample_correctness_scenarios['logically_inconsistent'].answer[:80]}...")
        print(f"[INFO] Overall score: {result.score:.3f}")
        print(f"[INFO] Explanation: {result.explanation}")
        
        # Should detect logical inconsistencies
        logic_score = result.details["evaluation_breakdown"]["logical_consistency"]["score"]
        assert logic_score < 0.8

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_answer_correctness_detailed_breakdown(self, azure_provider, sample_correctness_scenarios):
        """Test detailed breakdown of Answer Correctness evaluation using real API."""
        metric = AnswerCorrectness(llm_provider=azure_provider)
        result = await metric.evaluate(sample_correctness_scenarios["highly_correct"])
        
        assert isinstance(result, MetricResult)
        assert "evaluation_breakdown" in result.details
        
        breakdown = result.details["evaluation_breakdown"]
        
        print(f"\n[INFO] Detailed evaluation breakdown:")
        for component_name, component_data in breakdown.items():
            score = component_data["score"]
            explanation = component_data["details"]["llm_explanation"]
            print(f"[INFO] {component_name}: {score:.3f}")
            print(f"[INFO]   Explanation: {explanation[:80]}...")
            print()
        
        # Verify all expected components are present
        assert "factual_accuracy" in breakdown
        if sample_correctness_scenarios["highly_correct"].retrieved_contexts:
            assert "context_consistency" in breakdown
        assert "logical_consistency" in breakdown

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_answer_correctness_performance_timing(self, azure_provider, sample_correctness_scenarios):
        """Test Answer Correctness performance and timing."""
        import time
        
        metric = AnswerCorrectness(llm_provider=azure_provider)
        
        start_time = time.time()
        result = await metric.evaluate(sample_correctness_scenarios["highly_correct"])
        end_time = time.time()
        
        evaluation_time = end_time - start_time
        
        assert isinstance(result, MetricResult)
        assert evaluation_time < 45.0  # Should complete within 45 seconds (multiple LLM calls)
        
        print(f"\n[INFO] Performance test:")
        print(f"[INFO] Evaluation time: {evaluation_time:.2f} seconds")
        print(f"[INFO] Components evaluated: {result.details['components_evaluated']}")
        print(f"[INFO] Final score: {result.score:.3f}")

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_answer_correctness_component_configuration(self, azure_provider, sample_correctness_scenarios):
        """Test Answer Correctness with different component configurations."""
        # Test with only factual accuracy
        metric_factual_only = AnswerCorrectness(
            llm_provider=azure_provider,
            use_context_verification=False,
            check_logical_consistency=False
        )
        result_factual = await metric_factual_only.evaluate(sample_correctness_scenarios["highly_correct"])
        
        # Test with all components
        metric_all = AnswerCorrectness(llm_provider=azure_provider)
        result_all = await metric_all.evaluate(sample_correctness_scenarios["highly_correct"])
        
        print(f"\n[INFO] Component configuration comparison:")
        print(f"[INFO] Factual only - Components: {result_factual.details['components_evaluated']}, Score: {result_factual.score:.3f}")
        print(f"[INFO] All components - Components: {result_all.details['components_evaluated']}, Score: {result_all.score:.3f}")
        
        assert result_factual.details["components_evaluated"] < result_all.details["components_evaluated"] 