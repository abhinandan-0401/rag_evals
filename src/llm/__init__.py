"""LangChain LLM interface with Azure OpenAI support."""

import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


def create_llm(
    provider: str = "azure",
    model: str = "gpt-4", 
    api_key: Optional[str] = None,
    azure_config: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Union[ChatOpenAI, AzureChatOpenAI]:
    """Create a LangChain LLM instance.
    
    Args:
        provider: LLM provider ("azure" or "openai")
        model: Model name (e.g., "gpt-4", "gpt-35-turbo")
        api_key: API key for the provider
        azure_config: Azure-specific configuration
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        **kwargs: Additional LLM parameters
        
    Returns:
        Configured LangChain LLM instance
        
    Raises:
        ValueError: If required configuration is missing
        Exception: If LLM creation fails
    """
    # Prepare common LLM parameters
    llm_params = {
        "temperature": temperature,
        "max_retries": kwargs.get("max_retries", 3),
        **kwargs
    }
    
    if max_tokens:
        llm_params["max_tokens"] = max_tokens
    
    try:
        if provider.lower() == "azure":
            # Azure OpenAI configuration
            if azure_config is None:
                # Use environment variables with proper defaults
                azure_config = {
                    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"), 
                    "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                    "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
                }
            
            # Validate required Azure parameters
            required_params = ["api_key", "azure_endpoint", "azure_deployment"]
            missing_params = [p for p in required_params if not azure_config.get(p)]
            
            if missing_params:
                raise ValueError(
                    f"Missing required Azure OpenAI parameters: {missing_params}. "
                    f"Please provide them in azure_config or set environment variables: "
                    f"AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT"
                )
            
            # Create Azure OpenAI LLM
            return AzureChatOpenAI(
                api_key=azure_config.get("api_key") or api_key,
                azure_endpoint=azure_config["azure_endpoint"],
                azure_deployment=azure_config["azure_deployment"],
                api_version=azure_config.get("api_version", "2024-02-01"),
                model=model,
                **llm_params
            )
            
        elif provider.lower() == "openai":
            # OpenAI configuration
            final_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not final_api_key:
                raise ValueError(
                    "Missing OpenAI API key. Provide api_key parameter or set OPENAI_API_KEY environment variable."
                )
            
            # Create OpenAI LLM
            return ChatOpenAI(
                api_key=final_api_key,
                model=model,
                **llm_params
            )
            
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'azure' or 'openai'.")
            
    except Exception as e:
        raise Exception(f"Failed to create {provider} LLM: {str(e)}") from e


async def generate_with_langchain(
    llm: Union[ChatOpenAI, AzureChatOpenAI],
    prompt: str,
    system_prompt: Optional[str] = None,
    **kwargs
) -> LLMResponse:
    """Generate response using LangChain LLM.
    
    Args:
        llm: LangChain LLM instance
        prompt: User prompt
        system_prompt: Optional system prompt
        **kwargs: Additional generation parameters
        
    Returns:
        LLMResponse with generated content
        
    Raises:
        Exception: If generation fails
    """
    try:
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        # Generate response
        response = await llm.ainvoke(messages, **kwargs)
        
        # Extract usage information if available
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
                "completion_tokens": response.usage_metadata.get("output_tokens", 0),
                "total_tokens": response.usage_metadata.get("total_tokens", 0)
            }
        
        return LLMResponse(
            content=response.content,
            usage=usage,
            model=getattr(llm, 'model_name', None) or getattr(llm, 'model', None)
        )
        
    except Exception as e:
        raise Exception(f"LLM generation failed: {str(e)}") from e


# Import manager components
from .manager import LLMManager, get_llm_manager, get_managed_llm

# Export main components
__all__ = [
    "create_llm",
    "generate_with_langchain", 
    "LLMResponse",
    "ChatOpenAI",
    "AzureChatOpenAI",
    # Manager components
    "LLMManager",
    "get_llm_manager", 
    "get_managed_llm"
] 