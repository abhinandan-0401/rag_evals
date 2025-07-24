"""LLM Manager for centralized LLM instance management.

This module provides sophisticated LLM management including:
- Connection pooling and caching
- Configuration validation and normalization
- Performance monitoring
- Resource cleanup and management
"""

import hashlib
import time
import weakref
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from threading import Lock
import logging

from . import create_llm, ChatOpenAI, AzureChatOpenAI
from ..core.logging import get_logger
from ..core.exceptions import LLMConfigurationError, LLMError
from ..core.error_handler import with_error_recovery, get_recovery_manager


@dataclass(frozen=True)
class LLMConfig:
    """Immutable LLM configuration for caching and comparison."""
    provider: str
    model: str
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.provider not in ["azure", "openai"]:
            raise LLMConfigurationError(f"Unsupported provider: {self.provider}")
        
        if self.provider == "azure" and not all([self.azure_endpoint, self.azure_deployment]):
            raise LLMConfigurationError(
                "Azure provider requires azure_endpoint and azure_deployment"
            )
    
    def to_hash(self) -> str:
        """Generate a hash for this configuration for caching."""
        # Don't include api_key in hash for security
        config_without_key = asdict(self)
        config_without_key.pop('api_key', None)
        config_str = str(sorted(config_without_key.items()))
        return hashlib.md5(config_str.encode()).hexdigest()


class LLMPool:
    """Thread-safe LLM instance pool with automatic cleanup."""
    
    def __init__(self, max_pool_size: int = 10):
        self.max_pool_size = max_pool_size
        self._pool: Dict[str, Any] = {}
        self._usage_count: Dict[str, int] = {}
        self._last_used: Dict[str, float] = {}
        self._lock = Lock()
        self.logger = get_logger("llm_pool")
    
    def get_or_create(self, config: LLMConfig) -> Any:
        """Get existing LLM or create new one."""
        config_hash = config.to_hash()
        
        with self._lock:
            # Check if we have this LLM in pool
            if config_hash in self._pool:
                llm_ref = self._pool[config_hash]
                llm = llm_ref() if isinstance(llm_ref, weakref.ref) else llm_ref
                
                if llm is not None:
                    self._usage_count[config_hash] = self._usage_count.get(config_hash, 0) + 1
                    self._last_used[config_hash] = time.time()
                    
                    self.logger.debug(
                        f"Retrieved LLM from pool: {config.provider}:{config.model}",
                        extra={"config_hash": config_hash, "usage_count": self._usage_count[config_hash]}
                    )
                    return llm
                else:
                    # Clean up dead reference
                    self._cleanup_entry(config_hash)
            
            # Create new LLM with error recovery
            recovery_manager = get_recovery_manager()
            
            async def create_llm_operation():
                return create_llm(
                    provider=config.provider,
                    model=config.model,
                    api_key=config.api_key,
                    azure_config={
                        "azure_endpoint": config.azure_endpoint,
                        "azure_deployment": config.azure_deployment,
                        "api_version": config.api_version,
                        "api_key": config.api_key
                    } if config.provider == "azure" else None,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    max_retries=config.max_retries
                )
            
            try:
                # Use error recovery for LLM creation
                llm = recovery_manager.handle_sync(
                    lambda: create_llm(
                        provider=config.provider,
                        model=config.model,
                        api_key=config.api_key,
                        azure_config={
                            "azure_endpoint": config.azure_endpoint,
                            "azure_deployment": config.azure_deployment,
                            "api_version": config.api_version,
                            "api_key": config.api_key
                        } if config.provider == "azure" else None,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        max_retries=config.max_retries
                    ),
                    component="llm_pool",
                    operation_name="create_llm"
                )
                
                if llm is None:
                    # Degraded mode - return None to indicate failure
                    self.logger.warning(
                        f"LLM creation entered degraded mode: {config.provider}:{config.model}",
                        extra={"config_hash": config_hash}
                    )
                    return None
                
                # Add to pool
                self._add_to_pool(config_hash, llm)
                
                self.logger.info(
                    f"Created new LLM: {config.provider}:{config.model}",
                    extra={"config_hash": config_hash, "pool_size": len(self._pool)}
                )
                
                return llm
                
            except Exception as e:
                self.logger.error(
                    f"Failed to create LLM after recovery attempts: {config.provider}:{config.model}",
                    extra={"error": str(e), "config_hash": config_hash}
                )
                raise LLMError(f"LLM creation failed: {str(e)}", original_error=e)
    
    def _add_to_pool(self, config_hash: str, llm: Any) -> None:
        """Add LLM to pool with size management."""
        # Clean up old entries if pool is full
        if len(self._pool) >= self.max_pool_size:
            self._cleanup_oldest()
        
        # Use weak reference to allow garbage collection
        self._pool[config_hash] = weakref.ref(llm)
        self._usage_count[config_hash] = 1
        self._last_used[config_hash] = time.time()
    
    def _cleanup_entry(self, config_hash: str) -> None:
        """Clean up a specific pool entry."""
        self._pool.pop(config_hash, None)
        self._usage_count.pop(config_hash, None)
        self._last_used.pop(config_hash, None)
    
    def _cleanup_oldest(self) -> None:
        """Remove the least recently used LLM from pool."""
        if not self._last_used:
            return
        
        oldest_hash = min(self._last_used, key=self._last_used.get)
        self._cleanup_entry(oldest_hash)
        
        self.logger.debug(
            f"Cleaned up oldest LLM from pool",
            extra={"config_hash": oldest_hash}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "max_pool_size": self.max_pool_size,
                "total_usage": sum(self._usage_count.values()),
                "configurations": list(self._pool.keys())
            }
    
    def clear(self) -> None:
        """Clear the entire pool."""
        with self._lock:
            cleared_count = len(self._pool)
            self._pool.clear()
            self._usage_count.clear()
            self._last_used.clear()
            
            self.logger.info(
                f"Cleared LLM pool",
                extra={"cleared_instances": cleared_count}
            )


class LLMManager:
    """Centralized LLM management with pooling, caching, and monitoring."""
    
    _instance: Optional['LLMManager'] = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern for global LLM management."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize LLM manager."""
        if hasattr(self, '_initialized'):
            return
        
        self.pool = LLMPool()
        self.logger = get_logger("llm_manager")
        self._performance_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "creation_time": 0.0,
            "errors": 0
        }
        self._initialized = True
        
        self.logger.info("LLM Manager initialized")
    
    def get_llm(
        self,
        provider: str = "azure",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        azure_config: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[ChatOpenAI, AzureChatOpenAI]:
        """Get or create an LLM instance with caching.
        
        Args:
            provider: LLM provider ("azure" or "openai")
            model: Model name
            api_key: API key
            azure_config: Azure-specific configuration
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            LLM instance (cached or newly created)
            
        Raises:
            LLMConfigurationError: If configuration is invalid
            LLMError: If LLM creation fails
        """
        start_time = time.time()
        self._performance_stats["total_requests"] += 1
        
        try:
            # Normalize configuration
            config = self._normalize_config(
                provider, model, api_key, azure_config, 
                temperature, max_tokens, **kwargs
            )
            
            # Get or create LLM
            llm = self.pool.get_or_create(config)
            
            # Update performance stats
            creation_time = time.time() - start_time
            self._performance_stats["creation_time"] += creation_time
            
            self.logger.debug(
                f"LLM request completed: {provider}:{model}",
                extra={
                    "provider": provider,
                    "model": model,
                    "creation_time": creation_time,
                    "from_cache": creation_time < 0.001  # Very fast = from cache
                }
            )
            
            return llm
            
        except Exception as e:
            self._performance_stats["errors"] += 1
            self.logger.error(
                f"LLM request failed: {provider}:{model}",
                extra={"error": str(e), "provider": provider, "model": model}
            )
            raise
    
    def _normalize_config(
        self,
        provider: str,
        model: str,
        api_key: Optional[str],
        azure_config: Optional[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> LLMConfig:
        """Normalize and validate LLM configuration."""
        # Handle Azure config
        azure_endpoint = None
        azure_deployment = None
        api_version = None
        
        if provider == "azure" and azure_config:
            azure_endpoint = azure_config.get("azure_endpoint")
            azure_deployment = azure_config.get("azure_deployment")
            api_version = azure_config.get("api_version", "2024-02-01")
            # Use API key from azure_config if not provided directly
            if not api_key:
                api_key = azure_config.get("api_key")
        
        # Create configuration
        return LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=kwargs.get("max_retries", 3)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive LLM manager statistics."""
        pool_stats = self.pool.get_stats()
        
        return {
            "manager": {
                "total_requests": self._performance_stats["total_requests"],
                "cache_hits": self._performance_stats["cache_hits"],
                "total_errors": self._performance_stats["errors"],
                "avg_creation_time": (
                    self._performance_stats["creation_time"] / 
                    max(self._performance_stats["total_requests"], 1)
                ),
                "error_rate": (
                    self._performance_stats["errors"] / 
                    max(self._performance_stats["total_requests"], 1)
                )
            },
            "pool": pool_stats
        }
    
    def clear_cache(self) -> None:
        """Clear all cached LLM instances."""
        self.pool.clear()
        self.logger.info("LLM cache cleared")
    
    def validate_config(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        azure_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate LLM configuration without creating instance.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self._normalize_config(
                provider, model, api_key, azure_config, 0.0, None
            )
            return True, None
        except Exception as e:
            return False, str(e)


# Global LLM manager instance
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


# Convenience function that uses the manager
def get_managed_llm(
    provider: str = "azure",
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    azure_config: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Union[ChatOpenAI, AzureChatOpenAI]:
    """Get LLM instance through the manager (convenience function)."""
    return get_llm_manager().get_llm(
        provider=provider,
        model=model,
        api_key=api_key,
        azure_config=azure_config,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    ) 