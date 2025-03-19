"""
Error handling module for XAIR.
Defines custom exceptions and error handling utilities.
"""

import logging
import traceback
import json
from typing import Optional, Dict, Any, List, Union

# Configure logger
logger = logging.getLogger(__name__)

class XAIRError(Exception):
    """Base exception class for XAIR errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.status_code = status_code
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - {json.dumps(self.details)}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "status_code": self.status_code
        }

class LLMAPIError(XAIRError):
    """Exception raised for errors in the LLM API"""
    def __init__(self, message: str, status_code: int, response_text: str, request_data: Optional[Dict] = None):
        details = {
            "response_text": response_text,
            "request_data": request_data
        }
        super().__init__(message, details, status_code)

class AuthenticationError(XAIRError):
    """Exception raised for authentication errors"""
    def __init__(self, message: str, api_info: Optional[Dict] = None):
        super().__init__(message, api_info, status_code=401)

class RateLimitError(XAIRError):
    """Exception raised when rate limits are exceeded"""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(message, details, status_code=429)

class TokenLimitError(XAIRError):
    """Exception raised when token limits are exceeded"""
    def __init__(self, message: str, max_tokens: int, requested_tokens: int):
        details = {
            "max_tokens": max_tokens,
            "requested_tokens": requested_tokens
        }
        super().__init__(message, details, status_code=400)

class InvalidInputError(XAIRError):
    """Exception raised for invalid input parameters"""
    def __init__(self, message: str, invalid_fields: Optional[List[str]] = None):
        details = {"invalid_fields": invalid_fields} if invalid_fields else {}
        super().__init__(message, details, status_code=400)

class ServiceUnavailableError(XAIRError):
    """Exception raised when a required service is unavailable"""
    def __init__(self, message: str, service_name: str):
        super().__init__(message, {"service": service_name}, status_code=503)

def handle_api_error(status_code: int, response_text: str, request_data: Optional[Dict] = None) -> XAIRError:
    """
    Convert API errors into appropriate exception types
    
    Args:
        status_code: HTTP status code from the API
        response_text: Response text from the API
        request_data: Optional data that was sent in the request
        
    Returns:
        An appropriate XAIRError subclass
    """
    try:
        response_data = json.loads(response_text)
        error_message = response_data.get("error", {}).get("message", "Unknown API error")
    except (json.JSONDecodeError, AttributeError):
        error_message = response_text[:200] + ("..." if len(response_text) > 200 else "")
    
    if status_code == 401:
        return AuthenticationError(f"Authentication failed: {error_message}")
    elif status_code == 429:
        retry_after = None
        try:
            response_data = json.loads(response_text)
            retry_after = response_data.get("error", {}).get("retry_after")
        except (json.JSONDecodeError, AttributeError):
            pass
        return RateLimitError(f"Rate limit exceeded: {error_message}", retry_after)
    elif status_code == 400:
        # Check if it's a token limit issue
        if "maximum context length" in error_message.lower() or "token limit" in error_message.lower():
            # Extract token numbers if available
            max_tokens = 0
            requested_tokens = 0
            # This would need more precise parsing based on the actual error format
            return TokenLimitError(error_message, max_tokens, requested_tokens)
        else:
            return InvalidInputError(f"Invalid request: {error_message}")
    elif status_code == 503 or status_code == 502:
        return ServiceUnavailableError(f"Service unavailable: {error_message}", "LLM API")
    else:
        return LLMAPIError(f"API error (status {status_code}): {error_message}", 
                         status_code, response_text, request_data)

def log_exception(e: Exception, log_level: int = logging.ERROR) -> None:
    """
    Log an exception with details
    
    Args:
        e: The exception to log
        log_level: The logging level to use
    """
    logger.log(log_level, f"Exception: {str(e)}")
    logger.log(log_level, f"Type: {type(e).__name__}")
    logger.log(log_level, f"Traceback: {traceback.format_exc()}")
    
    if isinstance(e, XAIRError) and e.details:
        logger.log(log_level, f"Details: {json.dumps(e.details, indent=2)}")

def format_error_for_user(e: Exception) -> Dict[str, Any]:
    """
    Format an error for user display, hiding sensitive details
    
    Args:
        e: The exception to format
        
    Returns:
        Dictionary with formatted error information
    """
    if isinstance(e, XAIRError):
        result = e.to_dict()
        # Redact sensitive information
        if "request_data" in result.get("details", {}):
            if "api_key" in result["details"]["request_data"]:
                result["details"]["request_data"]["api_key"] = "REDACTED"
        return result
    else:
        return {
            "error": type(e).__name__,
            "message": str(e),
            "details": {},
            "status_code": 500
        }