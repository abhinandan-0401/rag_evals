"""Template utilities for prompt formatting."""

from typing import List, Union


def create_context_info(contexts: Union[str, List[str]], include_numbering: bool = True) -> str:
    """Create formatted context information for prompts.
    
    Args:
        contexts: Single context string or list of context strings
        include_numbering: Whether to include numbering for multiple contexts
        
    Returns:
        Formatted context string
    """
    if isinstance(contexts, str):
        return contexts
    elif isinstance(contexts, list):
        if len(contexts) == 1:
            return contexts[0]
        else:
            if include_numbering:
                return "\n\n".join(f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts))
            else:
                return "\n\n".join(contexts)
    else:
        return str(contexts)


def create_contexts_string(contexts: Union[str, List[str]]) -> str:
    """Create a formatted string from contexts.
    
    Args:
        contexts: Single context string or list of context strings
        
    Returns:
        Formatted context string
    """
    return create_context_info(contexts, include_numbering=True)


def format_prompt_with_context(template: str, contexts: Union[str, List[str]], **kwargs) -> str:
    """Format a prompt template with context and other variables.
    
    Args:
        template: The prompt template string
        contexts: Context information
        **kwargs: Other variables to substitute in the template
        
    Returns:
        Formatted prompt string
    """
    formatted_contexts = create_context_info(contexts)
    return template.format(contexts=formatted_contexts, **kwargs)


def create_numbered_list(items: List[str]) -> str:
    """Create a numbered list from items.
    
    Args:
        items: List of items to number
        
    Returns:
        Numbered list as string
    """
    return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))


def create_bulleted_list(items: List[str]) -> str:
    """Create a bulleted list from items.
    
    Args:
        items: List of items to bullet
        
    Returns:
        Bulleted list as string
    """
    return "\n".join(f"• {item}" for item in items) 