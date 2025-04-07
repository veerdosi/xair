"""
Error handling utilities for the XAIR system.
Provides consistent error handling and logging across modules.
"""

import logging
import functools
import traceback
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger(__name__)

# Type variables for function signatures
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')


class XAIRError(Exception):
    """Base exception class for XAIR system."""
    pass


class ModelError(XAIRError):
    """Exception for model-related errors."""
    pass


class DataProcessingError(XAIRError):
    """Exception for data processing errors."""
    pass


class ConfigurationError(XAIRError):
    """Exception for configuration-related errors."""
    pass


class KnowledgeGraphError(XAIRError):
    """Exception for knowledge graph related errors."""
    pass


def handle_exceptions(
    fallback_return: Any = None,
    reraise: bool = False,
    error_log_level: int = logging.ERROR
) -> Callable[[F], F]:
    """
    Decorator for consistent exception handling.
    
    Args:
        fallback_return: Value to return in case of exception
        reraise: Whether to reraise the exception after handling
        error_log_level: Logging level for the error
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error with traceback
                logger.log(
                    error_log_level,
                    f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                )
                
                # Reraise if specified
                if reraise:
                    raise
                
                # Otherwise return the fallback value
                return fallback_return
        
        return cast(F, wrapper)
    
    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    reraise: bool = True
) -> Callable[[F], F]:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for successive retries
        exceptions: Exception types to catch and retry
        reraise: Whether to reraise the exception after all retries fail
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import time
            
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(
                            f"Failed after {max_attempts} attempts in {func.__name__}: {str(e)}"
                        )
                        if reraise:
                            raise
                        return None
                    
                    logger.warning(
                        f"Attempt {attempts}/{max_attempts} failed in {func.__name__}: {str(e)}. "
                        f"Retrying in {current_delay:.2f}s"
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None  # Should never reach here
        
        return cast(F, wrapper)
    
    return decorator


def safe_file_operation(operation: Callable[[], T]) -> T:
    """
    Safely execute a file operation with proper exception handling.
    
    Args:
        operation: Function to execute
        
    Returns:
        Result of the operation
        
    Raises:
        XAIRError: For file operation errors
    """
    try:
        return operation()
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise XAIRError(f"File not found: {str(e)}") from e
    except PermissionError as e:
        logger.error(f"Permission denied: {str(e)}")
        raise XAIRError(f"Permission denied: {str(e)}") from e
    except IsADirectoryError as e:
        logger.error(f"Expected a file but found a directory: {str(e)}")
        raise XAIRError(f"Expected a file but found a directory: {str(e)}") from e
    except Exception as e:
        logger.error(f"File operation error: {str(e)}")
        raise XAIRError(f"File operation error: {str(e)}") from e