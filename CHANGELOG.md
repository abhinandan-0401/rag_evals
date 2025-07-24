# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2025-01-24

### Added
- Complete RAG evaluation library with LangChain integration
- Unified `RAGEvaluator` interface for all metrics
- 12 comprehensive evaluation metrics:
  - **Generation**: faithfulness, answer_relevance, answer_correctness, completeness, coherence, helpfulness
  - **Retrieval**: context_recall, context_relevance, context_precision  
  - **Composite**: llm_judge, rag_certainty, trust_score
- Azure OpenAI as primary LLM provider
- Comprehensive async support
- Advanced error handling and recovery
- Performance monitoring and analytics
- Dependency management system
- Modular YAML-based prompt system
- Professional documentation and packaging
- Modern `src/` layout with comprehensive test suite

### Changed
- **Breaking**: Complete API redesign for simplicity
- **Breaking**: Moved from individual metric classes to unified evaluator
- **Breaking**: Changed parameter names (`generated_text` → `answer`, `context` → `retrieved_contexts`)
- Modern `src/` layout for better package structure
- Switched to `pyproject.toml` from `setup.py`

### Fixed
- LLM Judge evaluation failures (missing enum values)
- Duplicate metric columns in CSV output (rag_certainty duplication)
- Import and validation issues across all modules
- Prompt template loading and caching
- Memory management in batch processing

### Infrastructure
- Comprehensive test suite with >90% coverage
- Professional documentation with detailed examples
- Development tools setup (black, isort, mypy, pytest)
- GitHub-ready project structure

## [1.x.x] - Legacy Versions
- Initial implementations with individual metric classes
- Basic LangChain integration
- Limited error handling
- Simple prompt management 