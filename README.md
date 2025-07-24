# 🎯 Monitoring-RAG

**Comprehensive RAG Evaluation with LangChain Integration**

[![PyPI version](https://badge.fury.io/py/rag_evals.svg)](https://pypi.org/project/monitoring-rag/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A streamlined Python library for evaluating Retrieval-Augmented Generation (RAG) systems using LangChain's powerful framework. Evaluate your RAG pipeline with a single line of code.

## 🚀 Key Features

- **🎯 Unified Interface**: Single `RAGEvaluator` class for all metrics
- **⚡ LangChain Powered**: Built on industry-standard LangChain framework
- **📊 Comprehensive Metrics**: 12+ evaluation metrics covering generation, retrieval, and composite evaluation
- **🔄 Async Support**: Full asynchronous evaluation for high performance
- **🛠️ Flexible Configuration**: Support for OpenAI and Azure OpenAI
- **📦 Simple API**: Evaluate with just query, context, and generated text

## 📋 Quick Reference

```python
# Import the main evaluator
from rag_evals import RAGEvaluator

# Initialize
evaluator = RAGEvaluator(llm_provider="openai", model="gpt-4")

# Evaluate all metrics
results = evaluator.evaluate(
    query="Your question",
    answer="RAG system's answer",
    retrieved_contexts=["Retrieved context"]
)

# Evaluate specific metrics
results = evaluator.evaluate(
    query="...",
    answer="...",
    retrieved_contexts=["..."],
    metrics=["faithfulness", "answer_relevance"]
)

# Available metrics
generation_metrics = ["faithfulness", "answer_relevance", "answer_correctness", 
                     "completeness", "coherence", "helpfulness"]
retrieval_metrics = ["context_recall", "context_relevance", "context_precision"]
composite_metrics = ["llm_judge", "rag_certainty", "trust_score"]
```

## 🚀 Installation

```bash
pip install monitoring-rag
```

> **Note**: Install as `monitoring-rag` but import as `rag_evals`

## 🔧 Quick Start

### Basic Usage

```python
from rag_evals import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator(
    llm_provider="openai",  # or "azure"
    model="gpt-4",
    api_key="your-api-key"  # or set OPENAI_API_KEY environment variable
)

# Evaluate a RAG response
results = evaluator.evaluate(
    query="What is machine learning?",
    answer="Machine learning is a subset of artificial intelligence that uses algorithms to learn patterns from data.",
    retrieved_contexts=["Machine learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data."]
)

print(results)
# Output: {
#     "faithfulness": 0.95,
#     "answer_relevance": 0.92,
#     "context_relevance": 0.88,
#     "completeness": 0.85,
#     "coherence": 0.90,
#     ...
# }
```

### Azure OpenAI Configuration

```python
from rag_evals import RAGEvaluator

# Using Azure OpenAI
evaluator = RAGEvaluator(
    llm_provider="azure",
    model="gpt-4",
    azure_config={
        "api_key": "your-azure-key",
        "azure_endpoint": "https://your-resource.openai.azure.com/",
        "azure_deployment": "gpt-4-deployment",
        "api_version": "2024-02-01"
    }
)

# Or using environment variables
# AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
evaluator = RAGEvaluator(llm_provider="azure", model="gpt-4")
```

### Batch Evaluation

```python
# Evaluate multiple responses at once
inputs = [
    {
        "query": "What is Python?",
        "generated_text": "Python is a programming language.",
        "context": "Python is a high-level programming language known for its simplicity."
    },
    {
        "query": "What is machine learning?",
        "generated_text": "ML uses algorithms to find patterns.",
        "context": "Machine learning involves training algorithms on data."
    }
]

results = evaluator.evaluate_batch(inputs)
# Returns list of result dictionaries
```

### Selective Metric Evaluation

```python
# Evaluate only specific metrics
results = evaluator.evaluate(
    query="What is AI?",
    generated_text="AI is artificial intelligence.",
    context="Artificial intelligence refers to machine intelligence.",
    metrics=["faithfulness", "answer_relevance", "coherence"]
)

# Configure evaluator with specific metrics
evaluator = RAGEvaluator(
    llm_provider="openai",
    model="gpt-4",
    metrics=["faithfulness", "completeness", "trust_score"]
)
```

## 📊 Available Metrics

### Generation Metrics
Evaluate the quality of generated responses:

- **Faithfulness**: How well the answer is grounded in the provided contexts
- **Answer Relevance**: How relevant the answer is to the user's query  
- **Answer Correctness**: Factual accuracy of the generated answer
- **Completeness**: How thoroughly the answer addresses the query
- **Coherence**: Logical flow and readability of the answer
- **Helpfulness**: Practical value and actionability of the answer

### Retrieval Metrics
Evaluate the quality of retrieved contexts:

- **Context Recall**: How well contexts support generating the ground truth
- **Context Relevance**: How relevant retrieved contexts are to the query
- **Context Precision**: Quality and ranking of retrieved contexts

### Composite Metrics
Holistic evaluation combining multiple aspects:

- **LLM Judge**: Comprehensive multi-dimensional evaluation
- **RAG Certainty**: Confidence and reliability assessment
- **Trust Score**: Overall trustworthiness evaluation

## 📖 Step-by-Step Tutorial

### Step 1: Installation and Setup

```bash
# Install the library
pip install rag_evals

# Create a .env file for your API keys (optional)
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Step 2: Basic Single Evaluation

```python
from rag_evals import RAGEvaluator

# Initialize the evaluator
evaluator = RAGEvaluator(
    llm_provider="openai",
    model="gpt-4"
    # API key will be read from OPENAI_API_KEY environment variable
)

# Your RAG system output
query = "What are the benefits of renewable energy?"
context = """
Renewable energy sources like solar and wind power offer numerous benefits:
1. They reduce greenhouse gas emissions
2. They provide energy independence
3. They create jobs in the green economy
4. Operating costs are lower than fossil fuels
"""
generated_text = """
Renewable energy provides several key benefits including reducing carbon emissions,
increasing energy independence, creating green jobs, and offering lower operating costs
compared to traditional fossil fuels.
"""

# Evaluate all metrics
results = evaluator.evaluate(
    query=query,
    generated_text=generated_text,
    context=context
)

# Display results
for metric, score in results.items():
    print(f"{metric}: {score:.2f}")
```

### Step 3: Evaluating Specific Metrics

```python
# Evaluate only generation metrics
generation_results = evaluator.evaluate(
    query=query,
    generated_text=generated_text,
    context=context,
    metrics=["faithfulness", "answer_relevance", "completeness"]
)

print("Generation Metrics:")
for metric, score in generation_results.items():
    print(f"  {metric}: {score:.2f}")

# Evaluate only retrieval metrics (requires ground truth)
retrieval_results = evaluator.evaluate(
    query=query,
    generated_text=generated_text,
    context=context,
    ground_truth="Renewable energy reduces emissions, provides energy independence, creates jobs, and has lower operating costs.",
    metrics=["context_recall", "context_precision"]
)

print("\nRetrieval Metrics:")
for metric, score in retrieval_results.items():
    print(f"  {metric}: {score:.2f}")
```

### Step 4: Batch Evaluation for Multiple Samples

```python
# Prepare multiple samples
samples = [
    {
        "query": "What is machine learning?",
        "generated_text": "Machine learning is a subset of AI that enables systems to learn from data.",
        "context": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
    },
    {
        "query": "Explain quantum computing",
        "generated_text": "Quantum computing uses quantum mechanics principles to process information.",
        "context": "Quantum computing is a type of computation that harnesses quantum mechanical phenomena like superposition and entanglement."
    },
    {
        "query": "What are neural networks?",
        "generated_text": "Neural networks are computing systems inspired by biological neural networks.",
        "context": "Artificial neural networks are computing systems vaguely inspired by the biological neural networks in animal brains."
    }
]

# Evaluate batch
batch_results = evaluator.evaluate_batch(samples)

# Analyze results
for i, result in enumerate(batch_results):
    print(f"\nSample {i+1}:")
    avg_score = sum(result.values()) / len(result)
    print(f"  Average Score: {avg_score:.2f}")
    print(f"  Faithfulness: {result.get('faithfulness', 0):.2f}")
    print(f"  Relevance: {result.get('answer_relevance', 0):.2f}")
```

### Step 5: Working with Azure OpenAI

```python
# Method 1: Using environment variables
import os
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4-deployment"

evaluator = RAGEvaluator(
    llm_provider="azure",
    model="gpt-4"
)

# Method 2: Direct configuration
evaluator = RAGEvaluator(
    llm_provider="azure",
    model="gpt-4",
    azure_config={
        "api_key": "your-azure-key",
        "azure_endpoint": "https://your-resource.openai.azure.com/",
        "azure_deployment": "gpt-4-deployment",
        "api_version": "2024-02-01"
    }
)
```

### Step 6: Interpreting Results

```python
def interpret_score(score: float, metric: str) -> str:
    """Interpret metric scores with recommendations."""
    if score >= 0.9:
        return "Excellent - No improvement needed"
    elif score >= 0.7:
        return "Good - Minor improvements possible"
    elif score >= 0.5:
        return "Moderate - Consider improvements"
    else:
        return "Poor - Significant improvements needed"

# Evaluate and interpret
results = evaluator.evaluate(query, generated_text, context)

print("RAG System Evaluation Report")
print("=" * 40)
for metric, score in results.items():
    interpretation = interpret_score(score, metric)
    print(f"{metric:20} {score:.2f} - {interpretation}")

# Identify weakest areas
weak_metrics = {m: s for m, s in results.items() if s < 0.7}
if weak_metrics:
    print("\n⚠️ Areas needing improvement:")
    for metric, score in weak_metrics.items():
        print(f"  - {metric}: {score:.2f}")
```

### Step 7: Production Usage Pattern

```python
import logging
from typing import Dict, Any

class RAGSystem:
    def __init__(self):
        self.evaluator = RAGEvaluator(
            llm_provider="openai",
            model="gpt-4",
            metrics=["faithfulness", "answer_relevance", "completeness"]
        )
        self.threshold = 0.7  # Minimum acceptable score
        
    def generate_and_evaluate(self, query: str) -> Dict[str, Any]:
        # Your RAG pipeline here
        context = self.retrieve_context(query)
        generated_text = self.generate_answer(query, context)
        
        # Evaluate the response
        scores = self.evaluator.evaluate(
            query=query,
            generated_text=generated_text,
            context=context
        )
        
        # Check if response meets quality threshold
        avg_score = sum(scores.values()) / len(scores)
        
        return {
            "answer": generated_text,
            "context": context,
            "evaluation": scores,
            "average_score": avg_score,
            "meets_threshold": avg_score >= self.threshold
        }
    
    def retrieve_context(self, query: str) -> str:
        # Your retrieval logic here
        return "Retrieved context..."
    
    def generate_answer(self, query: str, context: str) -> str:
        # Your generation logic here
        return "Generated answer..."

# Usage
rag_system = RAGSystem()
result = rag_system.generate_and_evaluate("What is climate change?")

if result["meets_threshold"]:
    print(f"✅ Response quality approved: {result['average_score']:.2f}")
else:
    print(f"❌ Response below threshold: {result['average_score']:.2f}")
    print("Consider regenerating or improving retrieval")
```

## 🔧 Advanced Usage

### Custom LLM Configuration

```python
# Configure LLM parameters
evaluator = RAGEvaluator(
    llm_provider="openai",
    model="gpt-4",
    temperature=0.1,
    max_tokens=1000,
    top_p=0.9
)
```

### Dynamic Metric Management

```python
# List available metrics
print(evaluator.list_metrics())

# Add/remove metrics dynamically
evaluator.add_metric("trust_score")
evaluator.remove_metric("coherence")

# Check configured metrics
print(evaluator.get_configured_metrics())
```

### Async Evaluation

```python
import asyncio

async def evaluate_async():
    results = await evaluator.aevaluate(
        query="What is RAG?",
        generated_text="RAG combines retrieval and generation...",
        context="Retrieval-Augmented Generation enhances LLMs..."
    )
    return results

# Run async evaluation
results = asyncio.run(evaluate_async())
```

## 💡 Common Use Cases

### 1. RAG System Development
```python
# During development, evaluate different retrieval strategies
strategies = ["dense", "sparse", "hybrid"]
results = {}

for strategy in strategies:
    context = retrieve_with_strategy(query, strategy)
    answer = generate_answer(query, context)
    
    scores = evaluator.evaluate(query, answer, context)
    results[strategy] = scores
    
# Compare strategies
best_strategy = max(results.items(), key=lambda x: sum(x[1].values()))
print(f"Best strategy: {best_strategy[0]}")
```

### 2. A/B Testing Different Models
```python
# Compare different LLM models
models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
model_performance = {}

for model in models:
    eval_temp = RAGEvaluator(llm_provider="openai", model=model)
    scores = eval_temp.evaluate(query, generated_text, context)
    model_performance[model] = sum(scores.values()) / len(scores)

# Find best performing model
best_model = max(model_performance.items(), key=lambda x: x[1])
print(f"Best model: {best_model[0]} (avg score: {best_model[1]:.2f})")
```

### 3. Quality Assurance Pipeline
```python
def qa_pipeline(query: str, answer: str, context: str) -> bool:
    """Quality assurance check before serving responses."""
    # Define minimum thresholds
    thresholds = {
        "faithfulness": 0.8,
        "answer_relevance": 0.7,
        "coherence": 0.75
    }
    
    # Evaluate critical metrics
    scores = evaluator.evaluate(
        query, answer, context, 
        metrics=list(thresholds.keys())
    )
    
    # Check all thresholds
    passed = all(scores[m] >= thresholds[m] for m in thresholds)
    
    if not passed:
        failed_metrics = [m for m in thresholds if scores[m] < thresholds[m]]
        print(f"QA Failed: {failed_metrics}")
    
    return passed
```

### 4. Continuous Monitoring
```python
import json
from datetime import datetime

def log_evaluation(query: str, answer: str, context: str, log_file: str):
    """Log evaluations for monitoring RAG system performance over time."""
    scores = evaluator.evaluate(query, answer, context)
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query[:100],  # Truncate for logging
        "scores": scores,
        "average_score": sum(scores.values()) / len(scores)
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    # Alert if performance drops
    if log_entry["average_score"] < 0.6:
        print(f"⚠️ Low performance alert: {log_entry['average_score']:.2f}")
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. API Key Errors
```python
# Issue: "API key not found" error
# Solution 1: Set environment variable
import os
os.environ["OPENAI_API_KEY"] = "your-key"

# Solution 2: Pass directly
evaluator = RAGEvaluator(
    llm_provider="openai",
    model="gpt-4",
    api_key="your-key"
)

# Solution 3: Use .env file
# Create .env file with: OPENAI_API_KEY=your-key
from dotenv import load_dotenv
load_dotenv()
```

#### 2. Azure Configuration Issues
```python
# Issue: Azure endpoint errors
# Solution: Ensure all required fields are provided
evaluator = RAGEvaluator(
    llm_provider="azure",
    model="gpt-4",
    azure_config={
        "api_key": "required",
        "azure_endpoint": "required - must end with /",
        "azure_deployment": "required - your deployment name",
        "api_version": "2024-02-01"  # Use latest version
    }
)
```

#### 3. Memory Issues with Batch Processing
```python
# Issue: Out of memory with large batches
# Solution: Process in smaller chunks
def evaluate_large_dataset(data, chunk_size=10):
    all_results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        results = evaluator.evaluate_batch(chunk)
        all_results.extend(results)
        
        # Optional: Add delay to avoid rate limits
        time.sleep(1)
    
    return all_results
```

#### 4. Handling Rate Limits
```python
import time
from typing import Dict, Any

def evaluate_with_retry(
    query: str, 
    generated_text: str, 
    context: str,
    max_retries: int = 3
) -> Dict[str, float]:
    """Evaluate with automatic retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return evaluator.evaluate(query, generated_text, context)
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
```

#### 5. Debugging Low Scores
```python
def debug_low_scores(query: str, answer: str, context: str):
    """Analyze why scores are low."""
    results = evaluator.evaluate(query, answer, context)
    
    print("Debugging Low Scores:")
    print("-" * 40)
    
    # Check each component
    if results.get("faithfulness", 0) < 0.5:
        print("❌ Faithfulness Issue: Answer may contain hallucinations")
        print("   Check if all claims are supported by context")
    
    if results.get("answer_relevance", 0) < 0.5:
        print("❌ Relevance Issue: Answer doesn't address the query")
        print("   Ensure answer directly responds to the question")
    
    if results.get("context_relevance", 0) < 0.5:
        print("❌ Context Issue: Retrieved context is not relevant")
        print("   Improve retrieval strategy or query processing")
    
    if results.get("completeness", 0) < 0.5:
        print("❌ Completeness Issue: Answer is incomplete")
        print("   Add more relevant information to the answer")
```

## 🏗️ Architecture

```
RAGEvaluator
├── Generation Metrics
│   ├── Faithfulness
│   ├── Answer Relevance  
│   ├── Answer Correctness
│   ├── Completeness
│   ├── Coherence
│   └── Helpfulness
├── Retrieval Metrics
│   ├── Context Recall
│   ├── Context Relevance
│   └── Context Precision
└── Composite Metrics
    ├── LLM Judge
    ├── RAG Certainty
    └── Trust Score
```

## 🔄 Migration from v1.x

RAG Evals 2.0 introduces breaking changes for a cleaner, more powerful API:

### Before (v1.x)
```python
from rag_evals.metrics import Faithfulness
from rag_evals.llm import OpenAIProvider

llm = OpenAIProvider(api_key="key", model="gpt-4")
metric = Faithfulness(llm_provider=llm)
result = await metric.evaluate(rag_input)
```

### After (v2.0)
```python
from rag_evals import RAGEvaluator

evaluator = RAGEvaluator(llm_provider="openai", model="gpt-4", api_key="key")
results = evaluator.evaluate(query="...", generated_text="...", context="...")
```

## 🛠️ Configuration

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Azure OpenAI  
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

### Configuration File Support

```python
# Load from config file
import json

with open("rag_config.json") as f:
    config = json.load(f)

evaluator = RAGEvaluator(**config)
```

Example `rag_config.json`:
```json
{
    "llm_provider": "azure",
    "model": "gpt-4",
    "azure_config": {
        "azure_endpoint": "https://your-resource.openai.azure.com/",
        "azure_deployment": "gpt-4-deployment"
    },
    "metrics": ["faithfulness", "answer_relevance", "completeness"]
}
```

## 📊 Example Results

```python
{
    "faithfulness": 0.95,        # 95% of claims supported by context
    "answer_relevance": 0.92,    # Highly relevant to the query
    "answer_correctness": 0.88,  # Factually accurate
    "completeness": 0.85,        # Covers most aspects of the query
    "coherence": 0.90,          # Well-structured and logical
    "helpfulness": 0.87,        # Practically useful
    "context_recall": 0.93,     # Context supports ground truth well
    "context_relevance": 0.89,  # Retrieved contexts are relevant
    "context_precision": 0.86,  # High-quality context ranking
    "llm_judge": 0.91,          # Overall high quality
    "rag_certainty": 0.88,      # High confidence in response
    "trust_score": 0.90         # Highly trustworthy
}
```

## 🔍 Understanding Scores

All metrics return scores between **0.0** (poor) and **1.0** (excellent):

- **0.9-1.0**: Excellent quality
- **0.7-0.9**: Good quality  
- **0.5-0.7**: Moderate quality
- **0.3-0.5**: Poor quality
- **0.0-0.3**: Very poor quality

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [LangChain](https://github.com/langchain-ai/langchain) for robust LLM integration
- Inspired by research in RAG evaluation and LLM-as-a-judge methodologies
- Community feedback and contributions drive continuous improvement

## 📚 Citation

If you use RAG Evals in your research, please cite:

```bibtex
@software{rag_evals,
  title = {RAG Evals: Comprehensive RAG Evaluation with LangChain},
  author = {RAG Evals Team},
  year = {2025},
  url = {https://github.com/ragevals/rag_evals}
}
``` 