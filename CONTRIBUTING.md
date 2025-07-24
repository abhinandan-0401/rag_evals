# Contributing to RAG Evals

We welcome contributions to RAG Evals! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Azure OpenAI access (for testing)

### Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/ragevals/rag_evals.git
cd rag_evals
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install in development mode**
```bash
pip install -e ".[dev,test]"
```

4. **Set up environment variables**
```bash
cp env.template .env
# Edit .env with your Azure OpenAI credentials
```

5. **Run tests**
```bash
pytest
```

## 🛠️ Development Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **flake8**: Linting

Run all checks:
```bash
black src/ tests/
isort src/ tests/
mypy src/
flake8 src/ tests/
```

### Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting PR
- Aim for >90% test coverage
- Use pytest fixtures for common test data

### Documentation

- Update docstrings for all new functions/classes
- Update README.md if adding new features
- Add examples for new metrics or functionality

## 🔄 Contribution Process

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests**
5. **Run the test suite**
6. **Commit your changes**
   ```bash
   git commit -m "feat: add your feature description"
   ```
7. **Push to your fork**
8. **Submit a Pull Request**

### Commit Message Convention

We follow conventional commits:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `style:` Code style changes
- `chore:` Maintenance tasks

## 📊 Adding New Metrics

To add a new evaluation metric:

1. **Choose the appropriate category**:
   - `src/metrics/generation/` - For answer quality metrics
   - `src/metrics/retrieval/` - For context quality metrics
   - `src/metrics/composite/` - For multi-dimensional metrics

2. **Create the metric class**:
```python
from ..core.base_metric import BaseMetric
from ..core.types import RAGInput, MetricResult, MetricType

class YourMetric(BaseMetric):
    def name(self) -> str:
        return "your_metric"
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.GENERATION  # or RETRIEVAL/COMPOSITE
    
    async def evaluate(self, rag_input: RAGInput) -> MetricResult:
        # Your evaluation logic here
        pass
```

3. **Add prompts** (if needed):
   - Create `src/prompts/your_metric.yaml`
   - Define templates for LLM-based evaluation

4. **Add tests**:
   - Create `tests/test_your_metric.py`
   - Test edge cases and error handling

5. **Update exports**:
   - Add to `src/metrics/__init__.py`
   - Add to `src/__init__.py`
   - Add to `RAGEvaluator.AVAILABLE_METRICS`

## 🐛 Bug Reports

When reporting bugs, please include:

- Python version
- RAG Evals version
- Complete error traceback
- Minimal code to reproduce
- Expected vs actual behavior

## 💡 Feature Requests

For feature requests, please:

- Describe the use case
- Explain the expected behavior
- Consider if it fits the project scope
- Provide examples if possible

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You

Thank you for contributing to RAG Evals! Your contributions help make RAG evaluation better for everyone. 