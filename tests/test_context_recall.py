"""
Comprehensive tests for Context Recall metric with Azure OpenAI provider.
"""

import pytest
import os
from unittest.mock import AsyncMock, MagicMock
from rag_evals.metrics.retrieval.context_recall import ContextRecall
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
def sample_rag_scenarios():
    """Sample RAG scenarios for testing."""
    # Create no_contexts scenario manually to bypass validation
    no_contexts_input = RAGInput(
        query="What is artificial intelligence?",
        answer="Artificial intelligence is the simulation of human intelligence in machines that are programmed to think like humans.",
        retrieved_contexts=["dummy"]  # Start with dummy to pass validation
    )
    no_contexts_input.retrieved_contexts = []  # Then set to empty
    
    return {
        "well_supported": RAGInput(
            query="What is machine learning?",
            answer="Machine learning is a subset of artificial intelligence that enables computers to learn from data. It uses algorithms to identify patterns and make predictions without being explicitly programmed for each task.",
            retrieved_contexts=[
                "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
                "Machine learning algorithms build a model based on training data in order to make predictions or decisions without being explicitly programmed to do so.",
                "The goal of machine learning is to enable computers to learn automatically from data and identify patterns to make decisions with minimal human intervention."
            ]
        ),
        "partially_supported": RAGInput(
            query="What are the benefits of renewable energy?",
            answer="Renewable energy sources like solar and wind are environmentally friendly and reduce carbon emissions. They also create millions of jobs and provide energy independence. Additionally, they eliminate all air pollution completely.",
            retrieved_contexts=[
                "Renewable energy sources such as solar, wind, and hydroelectric power produce significantly lower carbon emissions compared to fossil fuels.",
                "The renewable energy sector has become a major source of employment, with wind and solar industries creating hundreds of thousands of jobs.",
                "Renewable energy helps reduce dependence on imported fossil fuels, improving energy security for nations."
            ]
        ),
        "poorly_supported": RAGInput(
            query="What is quantum computing?",
            answer="Quantum computing uses quantum bits that can be in multiple states simultaneously through superposition. Current quantum computers have achieved quantum supremacy and can solve any problem faster than classical computers. IBM has already built quantum computers with over 10,000 qubits.",
            retrieved_contexts=[
                "Quantum computing is a type of computation that harnesses quantum mechanical phenomena like superposition and entanglement.",
                "Quantum bits or qubits are the basic unit of quantum information, unlike classical bits which can only be 0 or 1.",
                "Quantum computers are still in early stages of development and face significant technical challenges."
            ]
        ),
        "no_contexts": no_contexts_input
    }


class TestContextRecallMocked:
    """Test Context Recall metric with mocked LLM responses."""

    @pytest.mark.asyncio
    async def test_context_recall_basic_functionality(self, mock_llm_provider, sample_rag_scenarios):
        """Test basic Context Recall functionality with mocked responses."""
        # Mock claim extraction
        mock_llm_provider.generate = AsyncMock(side_effect=[
            # Claims extraction response
            MagicMock(content="CLAIM: Machine learning is a subset of artificial intelligence\nCLAIM: Machine learning enables computers to learn from data\nCLAIM: Machine learning uses algorithms to identify patterns"),
            # Verification responses for each claim
            MagicMock(content="Status: SUPPORTED\nExplanation: This claim is directly stated in Context 1"),
            MagicMock(content="Status: SUPPORTED\nExplanation: This claim matches the information in Context 1"),
            MagicMock(content="Status: SUPPORTED\nExplanation: This is supported by Context 2 about algorithms and patterns")
        ])
        
        metric = ContextRecall(llm_provider=mock_llm_provider)
        result = await metric.evaluate(sample_rag_scenarios["well_supported"])
        
        assert isinstance(result, MetricResult)
        assert result.score == 1.0  # All claims supported
        assert metric.metric_type == MetricType.RETRIEVAL
        assert metric.name() == "context_recall"
        assert "claims_extracted" in result.details
        assert "claims_supported" in result.details

    @pytest.mark.asyncio
    async def test_context_recall_partial_support(self, mock_llm_provider, sample_rag_scenarios):
        """Test Context Recall with partially supported claims."""
        # Mock responses for partial support scenario
        mock_llm_provider.generate = AsyncMock(side_effect=[
            # Claims extraction response
            MagicMock(content="CLAIM: Renewable energy reduces carbon emissions\nCLAIM: Renewable energy creates millions of jobs\nCLAIM: Renewable energy eliminates all air pollution"),
            # Verification responses
            MagicMock(content="Status: SUPPORTED\nExplanation: Directly stated in Context 1"),
            MagicMock(content="Status: SUPPORTED\nExplanation: Context 2 mentions job creation"), 
            MagicMock(content="Status: NOT SUPPORTED\nExplanation: This claim is too absolute and not supported by contexts")
        ])
        
        metric = ContextRecall(llm_provider=mock_llm_provider)
        result = await metric.evaluate(sample_rag_scenarios["partially_supported"])
        
        assert isinstance(result, MetricResult)
        assert result.score == 2/3  # 2 out of 3 claims supported
        assert result.details["claims_extracted"] == 3
        assert result.details["claims_supported"] == 2

    @pytest.mark.asyncio
    async def test_context_recall_no_claims(self, mock_llm_provider, sample_rag_scenarios):
        """Test Context Recall when no claims are extracted."""
        mock_llm_provider.generate = AsyncMock(return_value=MagicMock(content="No verifiable claims found."))
        
        metric = ContextRecall(llm_provider=mock_llm_provider)
        result = await metric.evaluate(sample_rag_scenarios["well_supported"])
        
        assert result.score == 1.0  # No claims means perfect recall
        assert result.details["claims_extracted"] == 0
        assert "No verifiable claims" in result.explanation

    @pytest.mark.asyncio
    async def test_context_recall_no_contexts(self, mock_llm_provider, sample_rag_scenarios):
        """Test Context Recall with no retrieved contexts."""
        metric = ContextRecall(llm_provider=mock_llm_provider)
        result = await metric.evaluate(sample_rag_scenarios["no_contexts"])
        
        assert result.score == 0.0
        assert "No contexts provided" in result.explanation
        assert result.details["error"] == "missing_contexts"

    @pytest.mark.asyncio
    async def test_context_recall_empty_answer(self, mock_llm_provider):
        """Test Context Recall with empty answer."""
        rag_input = RAGInput(
            query="What is AI?",
            answer="dummy",  # Start with dummy to pass validation
            retrieved_contexts=["AI is artificial intelligence."]
        )
        rag_input.answer = ""  # Then set to empty
        
        metric = ContextRecall(llm_provider=mock_llm_provider)
        result = await metric.evaluate(rag_input)
        
        assert result.score == 0.0
        assert "Empty answer provided" in result.explanation
        assert result.details["error"] == "empty_answer"

    @pytest.mark.asyncio
    async def test_context_recall_max_claims_limit(self, mock_llm_provider, sample_rag_scenarios):
        """Test Context Recall with max claims limit."""
        # Generate response with more claims than the limit
        many_claims = "\n".join([f"CLAIM: This is claim number {i}" for i in range(1, 15)])
        mock_llm_provider.generate = AsyncMock(side_effect=[
            MagicMock(content=many_claims),
            # Verification responses for limited claims (max 5)
            *[MagicMock(content="Status: SUPPORTED\nExplanation: Supported") for _ in range(5)]
        ])
        
        metric = ContextRecall(llm_provider=mock_llm_provider, max_claims_to_extract=5)
        result = await metric.evaluate(sample_rag_scenarios["well_supported"])
        
        assert result.details["claims_extracted"] == 5  # Limited to max
        assert len(result.details["claim_verifications"]) == 5


class TestContextRecallIntegration:
    """Integration tests with real Azure OpenAI API."""

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_context_recall_well_supported_scenario(self, azure_provider, sample_rag_scenarios):
        """Test Context Recall with well-supported claims using real API."""
        metric = ContextRecall(llm_provider=azure_provider)
        result = await metric.evaluate(sample_rag_scenarios["well_supported"])
        
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0
        assert result.details["claims_extracted"] > 0
        assert "claims_supported" in result.details
        
        print(f"\n[INFO] Well-supported scenario:")
        print(f"[INFO] Claims extracted: {result.details['claims_extracted']}")
        print(f"[INFO] Claims supported: {result.details['claims_supported']}")
        print(f"[INFO] Recall score: {result.score:.3f}")
        print(f"[INFO] Explanation: {result.explanation}")

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_context_recall_partially_supported_scenario(self, azure_provider, sample_rag_scenarios):
        """Test Context Recall with partially supported claims using real API."""
        metric = ContextRecall(llm_provider=azure_provider)
        result = await metric.evaluate(sample_rag_scenarios["partially_supported"])
        
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0
        assert result.details["claims_extracted"] > 0
        
        print(f"\n[INFO] Partially supported scenario:")
        print(f"[INFO] Claims extracted: {result.details['claims_extracted']}")
        print(f"[INFO] Claims supported: {result.details['claims_supported']}")
        print(f"[INFO] Recall score: {result.score:.3f}")
        print(f"[INFO] Explanation: {result.explanation}")
        
        # Should have some unsupported claims
        assert result.details["claims_supported"] < result.details["claims_extracted"]

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_context_recall_poorly_supported_scenario(self, azure_provider, sample_rag_scenarios):
        """Test Context Recall with poorly supported claims using real API."""
        metric = ContextRecall(llm_provider=azure_provider)
        result = await metric.evaluate(sample_rag_scenarios["poorly_supported"])
        
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0
        assert result.details["claims_extracted"] > 0
        
        print(f"\n[INFO] Poorly supported scenario:")
        print(f"[INFO] Claims extracted: {result.details['claims_extracted']}")
        print(f"[INFO] Claims supported: {result.details['claims_supported']}")
        print(f"[INFO] Recall score: {result.score:.3f}")
        print(f"[INFO] Explanation: {result.explanation}")
        
        # Should have low recall score
        assert result.score < 0.8

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_context_recall_detailed_verification(self, azure_provider, sample_rag_scenarios):
        """Test detailed claim verification process using real API."""
        metric = ContextRecall(llm_provider=azure_provider, max_claims_to_extract=3)
        result = await metric.evaluate(sample_rag_scenarios["well_supported"])
        
        assert isinstance(result, MetricResult)
        assert "claim_verifications" in result.details
        
        claim_verifications = result.details["claim_verifications"]
        assert len(claim_verifications) <= 3
        
        print(f"\n[INFO] Detailed claim verification:")
        for verification in claim_verifications:
            print(f"[INFO] Claim {verification['index']}: {verification['claim'][:50]}...")
            print(f"[INFO] Supported: {verification['supported']}")
            print(f"[INFO] Evidence: {verification['evidence'][:80]}...")
            print()

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_context_recall_edge_cases(self, azure_provider):
        """Test Context Recall with edge cases using real API."""
        # Test with very short answer
        short_answer_input = RAGInput(
            query="What is AI?",
            answer="AI is smart.",
            retrieved_contexts=["Artificial intelligence is intelligence demonstrated by machines."]
        )
        
        metric = ContextRecall(llm_provider=azure_provider)
        result = await metric.evaluate(short_answer_input)
        
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0
        
        print(f"\n[INFO] Short answer scenario:")
        print(f"[INFO] Answer: '{short_answer_input.answer}'")
        print(f"[INFO] Claims extracted: {result.details['claims_extracted']}")
        print(f"[INFO] Recall score: {result.score:.3f}")

    @skip_if_no_azure_creds()
    @pytest.mark.asyncio
    async def test_context_recall_performance_timing(self, azure_provider, sample_rag_scenarios):
        """Test Context Recall performance and timing."""
        import time
        
        metric = ContextRecall(llm_provider=azure_provider)
        
        start_time = time.time()
        result = await metric.evaluate(sample_rag_scenarios["well_supported"])
        end_time = time.time()
        
        evaluation_time = end_time - start_time
        
        assert isinstance(result, MetricResult)
        assert evaluation_time < 30.0  # Should complete within 30 seconds
        
        print(f"\n[INFO] Performance test:")
        print(f"[INFO] Evaluation time: {evaluation_time:.2f} seconds")
        print(f"[INFO] Claims processed: {result.details['claims_extracted']}")
        print(f"[INFO] Contexts analyzed: {result.details['contexts_count']}")
        print(f"[INFO] Final score: {result.score:.3f}") 