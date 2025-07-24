"""Modular prompt management system for RAG evaluation metrics.

This module provides a unified interface for loading and managing prompt templates
from YAML files with proper modularity, caching, and LangChain integration.
"""

# New modular prompt management system
from .manager import (
    PromptManager,
    get_prompt_manager,
    get_prompt,
    format_contexts
)

import warnings

# Initialize the prompt manager
try:
    # Use new prompt manager for loading
    manager = get_prompt_manager()
    
    # Test basic functionality
    available_metrics = manager.list_available_metrics()
    
    # Legacy prompt variables - maintained for backward compatibility
    # These will work if the prompts are available in YAML
    try:
        answer_relevancy = manager.get_prompt("answer_relevancy", "template")
    except:
        answer_relevancy = None
        
    try:
        faithfulness = manager.get_prompt("faithfulness", "template")
    except:
        faithfulness = None
    
    try:
        context_recall = manager.get_prompt("context_recall", "template")
    except:
        context_recall = None
        
    try:
        context_precision = manager.get_prompt("context_precision", "template")
    except:
        context_precision = None
        
    try:
        trust_score = manager.get_prompt("trust_score", "template")
    except:
        trust_score = None

except Exception as e:
    # If prompt manager fails, issue warning but don't crash
    warnings.warn(f"Prompt manager initialization failed: {e}. Some functionality may be limited.", UserWarning)
    manager = None
    answer_relevancy = None
    faithfulness = None
    context_recall = None
    context_precision = None
    trust_score = None


# Legacy compatibility functions - with deprecation warnings
def get_prompt_legacy(metric_name: str, prompt_type: str = "evaluation", category: str = "generation"):
    """Legacy get_prompt function - deprecated.
    
    Args:
        metric_name: Name of the metric
        prompt_type: Type of prompt 
        category: Category (generation, retrieval, composite)
        
    Returns:
        Prompt template
        
    This function is deprecated. Use get_prompt() from the new prompt management system instead.
    """
    warnings.warn(
        "get_prompt_legacy() is deprecated. Use get_prompt() from the new prompt management system.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if manager:
        return manager.get_prompt(metric_name, prompt_type, category)
    else:
        raise RuntimeError("Prompt manager not available")


def format_contexts_legacy(contexts, numbered: bool = True):
    """Legacy context formatting function - deprecated.
    
    This function is deprecated. Use format_contexts() from the new prompt management system.
    """
    warnings.warn(
        "format_contexts_legacy() is deprecated. Use format_contexts() from the new prompt management system.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if manager:
        return manager.format_contexts(contexts, numbered)
    else:
        # Simple fallback implementation
        if isinstance(contexts, str):
            return contexts
        
        if numbered:
            return "\n".join([f"{i+1}. {ctx}" for i, ctx in enumerate(contexts)])
        else:
            return "\n".join(contexts)


# Public API
__all__ = [
    # New prompt management system
    "PromptManager",
    "get_prompt_manager",
    "get_prompt",
    "format_contexts",
    
    # Legacy compatibility (deprecated)
    "get_prompt_legacy",
    "format_contexts_legacy",
    
    # Legacy prompt variables (may be None if loading fails)
    "answer_relevancy",
    "faithfulness", 
    "context_recall",
    "context_precision",
    "trust_score",
] 