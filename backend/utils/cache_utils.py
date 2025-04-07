"""
Caching utilities for the XAIR system.
Provides efficient caching for expensive operations.
"""

import os
import json
import pickle
import hashlib
import logging
import time
from typing import Any, Dict, Optional, Callable, TypeVar, cast, Union
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)

# Type variables for function signatures
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

class Cache:
    """Generic cache implementation with disk persistence."""
    
    def __init__(
        self,
        cache_dir: str,
        max_size: int = 1000,
        expiry_time: int = 86400,  # 24 hours in seconds
        save_interval: int = 100,  # Save to disk every N operations
        serializer: str = "json"  # "json" or "pickle"
    ):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of items in the cache
            expiry_time: Cache expiry time in seconds
            save_interval: Save to disk every N operations
            serializer: Serialization format ("json" or "pickle")
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.expiry_time = expiry_time
        self.save_interval = save_interval
        self.serializer = serializer
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache data structure
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_count: Dict[str, int] = {}
        self.operation_count = 0
        
        # Load existing cache from disk
        self.load_cache()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key in self.cache:
            entry = self.cache[key]
            # Check if expired
            if time.time() - entry.get("timestamp", 0) > self.expiry_time:
                # Remove expired entry
                self._remove(key)
                return None
            
            # Update access count
            self.access_count[key] = self.access_count.get(key, 0) + 1
            
            # Return the value
            return entry.get("value")
        
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Create cache entry
        entry = {
            "value": value,
            "timestamp": time.time()
        }
        
        # Add to cache
        self.cache[key] = entry
        self.access_count[key] = 1
        
        # Check cache size and evict if necessary
        self._enforce_size_limit()
        
        # Increment operation count and save if needed
        self.operation_count += 1
        if self.operation_count % self.save_interval == 0:
            self.save_cache()
    
    def _remove(self, key: str) -> None:
        """
        Remove an item from the cache.
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
        
        if key in self.access_count:
            del self.access_count[key]
    
    def _enforce_size_limit(self) -> None:
        """Enforce the cache size limit by removing least accessed items."""
        if len(self.cache) > self.max_size:
            # Sort keys by access count (least accessed first)
            sorted_keys = sorted(self.access_count.keys(), key=lambda k: self.access_count.get(k, 0))
            
            # Remove items until we're under the limit
            items_to_remove = len(self.cache) - self.max_size
            for key in sorted_keys[:items_to_remove]:
                self._remove(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_count.clear()
        self.operation_count = 0
        
        # Also remove cache files
        self.save_cache()
    
    def save_cache(self) -> None:
        """Save the cache to disk."""
        if self.serializer == "json":
            cache_file = os.path.join(self.cache_dir, "cache.json")
            try:
                # Convert cache to serializable format
                serializable_cache = {}
                for key, entry in self.cache.items():
                    # Skip entries that can't be JSON serialized
                    try:
                        json.dumps(entry["value"])
                        serializable_cache[key] = entry
                    except (TypeError, OverflowError):
                        pass
                
                with open(cache_file, "w") as f:
                    json.dump({
                        "cache": serializable_cache,
                        "access_count": self.access_count,
                        "timestamp": time.time()
                    }, f)
                
                logger.debug(f"Saved cache to {cache_file}")
            except Exception as e:
                logger.error(f"Error saving cache to disk: {e}")
        else:
            # Use pickle for non-JSON serializable objects
            cache_file = os.path.join(self.cache_dir, "cache.pkl")
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump({
                        "cache": self.cache,
                        "access_count": self.access_count,
                        "timestamp": time.time()
                    }, f)
                
                logger.debug(f"Saved cache to {cache_file}")
            except Exception as e:
                logger.error(f"Error saving cache to disk: {e}")
    
    def load_cache(self) -> None:
        """Load the cache from disk."""
        if self.serializer == "json":
            cache_file = os.path.join(self.cache_dir, "cache.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                        
                        self.cache = data.get("cache", {})
                        self.access_count = data.get("access_count", {})
                        
                        # Cleanup expired entries
                        self._cleanup_expired()
                        
                        logger.info(f"Loaded {len(self.cache)} entries from cache")
                except Exception as e:
                    logger.error(f"Error loading cache from disk: {e}")
                    self.cache = {}
                    self.access_count = {}
        else:
            # Use pickle for non-JSON serializable objects
            cache_file = os.path.join(self.cache_dir, "cache.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        data = pickle.load(f)
                        
                        self.cache = data.get("cache", {})
                        self.access_count = data.get("access_count", {})
                        
                        # Cleanup expired entries
                        self._cleanup_expired()
                        
                        logger.info(f"Loaded {len(self.cache)} entries from cache")
                except Exception as e:
                    logger.error(f"Error loading cache from disk: {e}")
                    self.cache = {}
                    self.access_count = {}
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry.get("timestamp", 0) > self.expiry_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove(key)
        
        if expired_keys:
            logger.info(f"Removed {len(expired_keys)} expired entries from cache")


def memoize(
    expire_after: int = 3600,  # 1 hour in seconds
    key_fn: Optional[Callable[..., str]] = None,
    cache_none: bool = False
) -> Callable[[F], F]:
    """
    Memoization decorator that caches function results.
    
    Args:
        expire_after: Cache expiry time in seconds
        key_fn: Function to generate cache key from arguments
        cache_none: Whether to cache None results
        
    Returns:
        Decorated function
    """
    cache: Dict[str, Dict[str, Any]] = {}
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [str(func.__module__), str(func.__name__)]
                
                # Add args to key
                for arg in args:
                    try:
                        key_parts.append(str(arg))
                    except Exception:
                        # If argument can't be converted to string, use its id
                        key_parts.append(f"obj:{id(arg)}")
                
                # Add kwargs to key (sorted by key for consistency)
                for k in sorted(kwargs.keys()):
                    try:
                        key_parts.append(f"{k}:{kwargs[k]}")
                    except Exception:
                        # If argument can't be converted to string, use its id
                        key_parts.append(f"{k}:obj:{id(kwargs[k])}")
                
                # Create hash of the key parts
                key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Check cache
            if key in cache:
                entry = cache[key]
                # Check if expired
                if time.time() - entry.get("timestamp", 0) <= expire_after:
                    return entry.get("result")
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result if not None or if cache_none is True
            if result is not None or cache_none:
                cache[key] = {
                    "result": result,
                    "timestamp": time.time()
                }
            
            return result
        
        # Add cache clear method to the function
        wrapper.clear_cache = lambda: cache.clear()  # type: ignore
        
        return cast(F, wrapper)
    
    return decorator


def cached_property(expire_after: int = 3600) -> Callable[[Callable[[Any], T]], property]:
    """
    Decorator for creating a property that is cached with expiration.
    
    Args:
        expire_after: Cache expiry time in seconds
        
    Returns:
        Property decorator
    """
    def decorator(func: Callable[[Any], T]) -> property:
        cache_key = f"__cached_{func.__name__}"
        timestamp_key = f"__cached_{func.__name__}_timestamp"
        
        @wraps(func)
        def getter(self: Any) -> T:
            # Check if result is cached and not expired
            if hasattr(self, cache_key):
                timestamp = getattr(self, timestamp_key, 0)
                if time.time() - timestamp <= expire_after:
                    return getattr(self, cache_key)
            
            # Compute the result
            result = func(self)
            
            # Cache the result
            setattr(self, cache_key, result)
            setattr(self, timestamp_key, time.time())
            
            return result
        
        return property(getter)
    
    return decorator