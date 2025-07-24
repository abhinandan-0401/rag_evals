"""Advanced prompt management system for RAG evaluation metrics.

This module provides a unified, scalable interface for loading and managing
prompt templates from YAML files with caching, validation, and LangChain integration.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from functools import lru_cache
import warnings

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts.base import BasePromptTemplate

from ..core.logging import get_logger
from ..core.exceptions import PromptError


class PromptManager:
    """Centralized prompt management system with YAML-based configuration.
    
    Provides modular, scalable prompt management with:
    - YAML-based prompt storage
    - LangChain integration
    - Caching for performance
    - Validation and error handling
    - Support for multiple prompt types per metric
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory containing YAML prompt files (defaults to package prompts dir)
        """
        self.logger = get_logger("prompts")
        self.prompts_dir = prompts_dir or Path(__file__).parent
        self._cache: Dict[str, BasePromptTemplate] = {}
        self._yaml_cache: Dict[str, Dict[str, Any]] = {}
        
        # Load all YAML files on initialization
        self._load_all_yaml_files()
        
        self.logger.info(
            f"PromptManager initialized with {len(self._yaml_cache)} YAML files",
            extra={
                "prompts_dir": str(self.prompts_dir),
                "yaml_files": list(self._yaml_cache.keys())
            }
        )
    
    def _load_all_yaml_files(self) -> None:
        """Load all YAML files in the prompts directory."""
        for yaml_file in self.prompts_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f)
                    if content:
                        self._yaml_cache[yaml_file.stem] = content
                        
                self.logger.debug(
                    f"Loaded YAML file: {yaml_file.name}",
                    extra={"file": yaml_file.name, "keys": list(content.keys()) if content else []}
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load YAML file {yaml_file.name}: {str(e)}",
                    extra={"file": yaml_file.name, "error": str(e)}
                )
                raise PromptError(
                    f"Failed to load prompt file {yaml_file.name}",
                    prompt_name=yaml_file.stem,
                    original_error=e
                )
    
    @lru_cache(maxsize=128)
    def get_prompt(
        self, 
        metric_name: str, 
        prompt_type: str = "template", 
        category: Optional[str] = None,
        as_chat_prompt: bool = False
    ) -> BasePromptTemplate:
        """Get a prompt template for a specific metric and type.
        
        Args:
            metric_name: Name of the metric (e.g., 'faithfulness')
            prompt_type: Type of prompt (e.g., 'template', 'system_prompt', 'evaluation_prompt')  
            category: Category file to search (e.g., 'generation', 'retrieval', 'composite')
            as_chat_prompt: Whether to return a ChatPromptTemplate instead of PromptTemplate
            
        Returns:
            LangChain prompt template
            
        Raises:
            PromptError: If prompt is not found or invalid
        """
        cache_key = f"{metric_name}:{prompt_type}:{category}:{as_chat_prompt}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Search for prompt in multiple locations
        prompt_data = self._find_prompt_data(metric_name, prompt_type, category)
        
        if not prompt_data:
            raise PromptError(
                f"Prompt not found: {metric_name}.{prompt_type}",
                prompt_name=f"{metric_name}.{prompt_type}",
                context={
                    "metric_name": metric_name,
                    "prompt_type": prompt_type,
                    "category": category,
                    "available_metrics": self.list_available_metrics()
                }
            )
        
        # Create LangChain prompt template
        template = self._create_prompt_template(prompt_data, as_chat_prompt)
        
        # Cache the template
        self._cache[cache_key] = template
        
        self.logger.debug(
            f"Created prompt template: {cache_key}",
            extra={
                "metric_name": metric_name,
                "prompt_type": prompt_type,
                "template_type": type(template).__name__,
                "input_variables": template.input_variables
            }
        )
        
        return template
    
    def _find_prompt_data(
        self, 
        metric_name: str, 
        prompt_type: str, 
        category: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Find prompt data from individual metric YAML files.
        
        Simplified search for individual-only structure:
        1. Look for individual metric file (e.g., faithfulness.yaml)
        2. Extract the specific prompt type from that metric
        """
        # Search for individual metric file first
        if metric_name in self._yaml_cache:
            metric_data = self._yaml_cache[metric_name]
            if isinstance(metric_data, dict) and metric_name in metric_data:
                metric_section = metric_data[metric_name]
                if isinstance(metric_section, dict):
                    # Look for specific prompt type
                    if prompt_type in metric_section:
                        return metric_section[prompt_type]
                    # Fallback to template if prompt_type is "template" and _type exists
                    elif prompt_type == "template" and "_type" in metric_section:
                        return metric_section
        
        # If not found in individual file, return None
        return None
    
    def _create_prompt_template(
        self, 
        prompt_data: Dict[str, Any], 
        as_chat_prompt: bool = False
    ) -> BasePromptTemplate:
        """Create a LangChain prompt template from prompt data.
        
        Args:
            prompt_data: Prompt configuration dictionary
            as_chat_prompt: Whether to create ChatPromptTemplate
            
        Returns:
            LangChain prompt template
        """
        if not isinstance(prompt_data, dict):
            raise PromptError(
                "Invalid prompt data format - must be a dictionary",
                context={"prompt_data_type": type(prompt_data).__name__}
            )
        
        # Extract template content
        template_content = prompt_data.get("template", "")
        input_variables = prompt_data.get("input_variables", [])
        
        if not template_content:
            raise PromptError(
                "Prompt template content is empty",
                context={"prompt_data": prompt_data}
            )
        
        # Validate input variables match template
        self._validate_template_variables(template_content, input_variables)
        
        # Create appropriate prompt type
        if as_chat_prompt:
            return ChatPromptTemplate.from_template(
                template_content,
                input_variables=input_variables
            )
        else:
            return PromptTemplate(
                template=template_content,
                input_variables=input_variables
            )
    
    def _validate_template_variables(self, template: str, declared_variables: List[str]) -> None:
        """Validate that template variables match declared input variables."""
        import re
        
        # Find variables in template using regex
        template_variables = set(re.findall(r'\{(\w+)\}', template))
        declared_set = set(declared_variables)
        
        # Check for mismatches
        missing_declarations = template_variables - declared_set
        unused_declarations = declared_set - template_variables
        
        if missing_declarations:
            warnings.warn(
                f"Template uses undeclared variables: {missing_declarations}",
                UserWarning
            )
        
        if unused_declarations:
            warnings.warn(
                f"Declared variables not used in template: {unused_declarations}",
                UserWarning
            )
    
    def get_metric_prompts(self, metric_name: str) -> Dict[str, BasePromptTemplate]:
        """Get all available prompts for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary mapping prompt types to templates
        """
        prompts = {}
        
        # Search all YAML files for prompts related to this metric
        for file_name, content in self._yaml_cache.items():
            if isinstance(content, dict):
                # Check if metric exists in this file
                if metric_name in content:
                    metric_data = content[metric_name]
                    if isinstance(metric_data, dict):
                        # Get all prompt types for this metric
                        for prompt_type, prompt_data in metric_data.items():
                            if isinstance(prompt_data, dict) and "template" in prompt_data:
                                try:
                                    prompts[prompt_type] = self.get_prompt(
                                        metric_name, prompt_type, file_name
                                    )
                                except PromptError:
                                    continue
                
                # Check if this is a direct metric file
                elif file_name == metric_name and "_type" in content:
                    try:
                        prompts["template"] = self.get_prompt(metric_name, "template")
                    except PromptError:
                        continue
        
        return prompts
    
    def list_available_metrics(self) -> List[str]:
        """Get list of all available metrics with prompts.
        
        Returns:
            List of metric names
        """
        metrics = set()
        
        # Add individual metric files
        for file_name, content in self._yaml_cache.items():
            if isinstance(content, dict) and "_type" in content:
                metrics.add(file_name)
        
        # Add metrics from category files
        for file_name, content in self._yaml_cache.items():
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, dict) and ("template" in value or "_type" in value):
                        metrics.add(key)
        
        return sorted(list(metrics))
    
    def list_prompt_types(self, metric_name: str) -> List[str]:
        """Get list of available prompt types for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of prompt type names
        """
        prompt_types = set()
        
        for file_name, content in self._yaml_cache.items():
            if isinstance(content, dict) and metric_name in content:
                metric_data = content[metric_name]
                if isinstance(metric_data, dict):
                    for key, value in metric_data.items():
                        if isinstance(value, dict) and "template" in value:
                            prompt_types.add(key)
        
        return sorted(list(prompt_types))
    
    def reload_prompts(self) -> None:
        """Reload all prompts from YAML files and clear cache."""
        self._cache.clear()
        self._yaml_cache.clear()
        self._load_all_yaml_files()
        
        self.logger.info(
            "Reloaded all prompts",
            extra={"yaml_files_count": len(self._yaml_cache)}
        )
    
    def validate_all_prompts(self) -> Dict[str, List[str]]:
        """Validate all prompts and return any issues found.
        
        Returns:
            Dictionary mapping categories to lists of validation issues
        """
        issues = {"errors": [], "warnings": []}
        
        for metric_name in self.list_available_metrics():
            try:
                metric_prompts = self.get_metric_prompts(metric_name)
                if not metric_prompts:
                    issues["warnings"].append(f"No prompts found for metric: {metric_name}")
            except Exception as e:
                issues["errors"].append(f"Error validating {metric_name}: {str(e)}")
        
        return issues
    
    def format_contexts(self, contexts: Union[str, List[str]], numbered: bool = True) -> str:
        """Format contexts for use in prompts.
        
        Args:
            contexts: Single context string or list of contexts
            numbered: Whether to number multiple contexts
            
        Returns:
            Formatted context string
        """
        if isinstance(contexts, str):
            return contexts
        elif isinstance(contexts, list):
            if len(contexts) == 0:
                return ""
            elif len(contexts) == 1:
                return contexts[0]
            else:
                if numbered:
                    return "\n\n".join(f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts))
                else:
                    return "\n\n".join(contexts)
        else:
            return str(contexts)
    
    def __repr__(self) -> str:
        """String representation of the prompt manager."""
        return (f"PromptManager(prompts_dir='{self.prompts_dir}', "
                f"metrics={len(self.list_available_metrics())}, "
                f"cached_prompts={len(self._cache)})")


# Global prompt manager instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance.
    
    Returns:
        Global PromptManager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(
    metric_name: str, 
    prompt_type: str = "template", 
    category: Optional[str] = None,
    as_chat_prompt: bool = False
) -> BasePromptTemplate:
    """Convenience function to get a prompt template.
    
    Args:
        metric_name: Name of the metric
        prompt_type: Type of prompt
        category: Category file to search
        as_chat_prompt: Whether to return ChatPromptTemplate
        
    Returns:
        LangChain prompt template
    """
    return get_prompt_manager().get_prompt(metric_name, prompt_type, category, as_chat_prompt)


def format_contexts(contexts: Union[str, List[str]], numbered: bool = True) -> str:
    """Convenience function to format contexts.
    
    Args:
        contexts: Single context string or list of contexts
        numbered: Whether to number multiple contexts
        
    Returns:
        Formatted context string
    """
    return get_prompt_manager().format_contexts(contexts, numbered)


__all__ = [
    "PromptManager",
    "get_prompt_manager", 
    "get_prompt",
    "format_contexts"
] 