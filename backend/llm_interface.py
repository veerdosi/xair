"""
LLM Interface module for XAIR.
Provides a robust interface to interact with Large Language Models.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any
import aiohttp
import asyncio
import json
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import numpy as np
from abc import ABC, abstractmethod

# Import error handling
from backend.error_handling import (
    XAIRError, LLMAPIError, AuthenticationError, RateLimitError,
    TokenLimitError, InvalidInputError, ServiceUnavailableError,
    handle_api_error, log_exception, format_error_for_user
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from an LLM query"""
    text: str
    logits: Optional[np.ndarray] = None
    attention: Optional[np.ndarray] = None
    tokens: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseLLMInterface(ABC):
    """Abstract base class for LLM interfaces"""
    
    @abstractmethod
    async def query(self, prompt: str, **kwargs) -> LLMResponse:
        """Send a query to the LLM and get a response"""
        pass

    @abstractmethod
    async def get_token_probabilities(self, text: str) -> Dict[str, float]:
        """Get probabilities for next tokens given a text"""
        pass

    @abstractmethod
    async def get_attention_flow(self, text: str) -> np.ndarray:
        """Get attention flow matrix for a text"""
        pass

class LLMInterface(BaseLLMInterface):
    """Interface for interacting with OpenAI LLMs like GPT-4o"""
    
    def __init__(
        self,
        api_key: str,
        api_endpoint: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        return_logits: bool = True,
        return_attention: bool = True,
        timeout: float = 30.0,
        max_retries: int = 3,
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize the LLM interface
        
        Args:
            api_key: OpenAI API key
            api_endpoint: API endpoint URL
            model_name: Model to use (e.g., "gpt-4o")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_logits: Whether to return logits
            return_attention: Whether to return attention data
            timeout: Timeout for API requests in seconds
            max_retries: Maximum number of retries for failed requests
            session: Optional existing aiohttp session to use
        """
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.return_logits = return_logits
        self.return_attention = return_attention
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session
        self._created_session = False
        self._embeddings_cache = {}

    async def __aenter__(self):
        """Set up the session for the async context manager"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers=self._get_default_headers(),
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            self._created_session = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the session when exiting the async context"""
        if self._created_session and self.session:
            await self.session.close()
            self.session = None
            self._created_session = False

    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @retry(
        retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True
    )
    async def query(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a query to the LLM and get a response
        
        Args:
            prompt: The text prompt to send
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            stop: Optional list of stop sequences
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            LLMResponse object with the response data
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limits are exceeded
            TokenLimitError: If token limits are exceeded
            InvalidInputError: If the input is invalid
            ServiceUnavailableError: If the service is unavailable
            LLMAPIError: For other API errors
        """
        if self.session is None:
            raise RuntimeError("Session not initialized. Use async with context.")

        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "logprobs": self.return_logits,
            "stream": False,
            **kwargs
        }
        
        if stop:
            payload["stop"] = stop

        start_time = time.time()
        try:
            async with self.session.post(
                f"{self.api_endpoint}/chat/completions",
                json=payload,
                raise_for_status=False
            ) as response:
                response_text = await response.text()
                
                # Log request duration
                duration = time.time() - start_time
                logger.debug(f"LLM request completed in {duration:.2f}s")
                
                if response.status != 200:
                    # Handle API errors with custom exceptions
                    error = handle_api_error(
                        response.status, 
                        response_text, 
                        {"model": self.model_name, "prompt_length": len(prompt)}
                    )
                    log_exception(error)
                    raise error
                
                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    raise LLMAPIError(
                        "Failed to parse API response", 
                        500, 
                        response_text[:200],
                        {"error": str(e)}
                    )
                
                # Extract response data
                try:
                    # Extract text from the response
                    text = data["choices"][0]["message"]["content"]
                    
                    # Extract logits if available
                    logits = None
                    if self.return_logits and "logprobs" in data["choices"][0]:
                        logits = np.array(data["choices"][0]["logprobs"]["token_logprobs"])
                    
                    # Extract tokens if available
                    tokens = None
                    if self.return_logits and "logprobs" in data["choices"][0]:
                        tokens = data["choices"][0]["logprobs"]["tokens"]
                    
                    # OpenAI doesn't directly provide attention matrices
                    attention = None
                    
                    # Extract metadata
                    metadata = {
                        "model": data.get("model", self.model_name),
                        "usage": data.get("usage", {}),
                        "finish_reason": data["choices"][0].get("finish_reason"),
                        "response_time": duration
                    }
                    
                    return LLMResponse(
                        text=text,
                        logits=logits,
                        attention=attention,
                        tokens=tokens,
                        metadata=metadata
                    )
                except (KeyError, IndexError) as e:
                    raise LLMAPIError(
                        "Unexpected API response format", 
                        500, 
                        response_text[:200],
                        {"error": str(e)}
                    )

        except aiohttp.ClientError as e:
            # Handle network errors
            if "Timeout" in str(e):
                raise ServiceUnavailableError(
                    f"Request timed out after {self.timeout}s", 
                    "OpenAI API"
                ) from e
            else:
                raise LLMAPIError(
                    f"Network error: {str(e)}", 
                    500, 
                    str(e),
                    {"request_timeout": self.timeout}
                ) from e

    async def get_token_probabilities(self, text: str) -> Dict[str, float]:
        """
        Get probabilities for next tokens given a text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping tokens to probabilities
            
        Raises:
            ValueError: If logits or tokens are not returned from LLM
        """
        try:
            response = await self.query(text)
            if response.logits is None or response.tokens is None:
                raise ValueError("Logits or tokens not returned from LLM. Check API settings.")

            # Convert logprobs to probabilities
            # logprob = log(p), so p = exp(logprob)
            probs = np.exp(response.logits)

            # Create dictionary mapping tokens to probabilities
            return dict(zip(response.tokens, probs.tolist()))
        except XAIRError:
            # Re-raise XAIR errors directly
            raise
        except Exception as e:
            # Convert other exceptions to LLMAPIError
            error = LLMAPIError(
                f"Failed to get token probabilities: {str(e)}",
                500,
                str(e),
                {"text_length": len(text)}
            )
            log_exception(error)
            raise error

    async def get_attention_flow(self, text: str) -> np.ndarray:
        """
        Get attention flow matrices for a given text.
        Since OpenAI API doesn't provide attention matrices directly,
        this implementation creates a more sophisticated approximation based on:
        1. Token embedding similarities
        2. Sequential positioning
        3. Syntactic structure analysis
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of shape (n_tokens, n_tokens) representing attention flow
            
        Raises:
            ValueError: If tokens are not returned from LLM
            ImportError: If required libraries are not available
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            import spacy
            import numpy as np
            
            # Get tokens from the LLM
            response = await self.query(text)
            if response.tokens is None:
                raise ValueError("Tokens not returned from LLM")
            
            tokens = response.tokens
            n_tokens = len(tokens)
            
            # Use cached models or load them
            if not hasattr(self, '_tokenizer') or not hasattr(self, '_model'):
                try:
                    logger.info("Loading NLP models for attention flow calculation")
                    self._tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                    self._model = AutoModel.from_pretrained('bert-base-uncased')
                    try:
                        self._nlp = spacy.load('en_core_web_sm')
                    except OSError:
                        # Try to download if not found
                        logger.info("Downloading spaCy model...")
                        import subprocess
                        subprocess.run([
                            "python", "-m", "spacy", "download", "en_core_web_sm"
                        ], check=True)
                        self._nlp = spacy.load('en_core_web_sm')
                except Exception as e:
                    logger.error(f"Failed to load NLP models: {e}")
                    raise ImportError(f"Required NLP models could not be loaded: {e}")
                
            # Initialize attention matrix
            attention = np.zeros((n_tokens, n_tokens))
            
            try:
                # Process text with transformer model
                inputs = self._tokenizer(text, return_tensors='pt')
                with torch.no_grad():
                    outputs = self._model(**inputs)
                
                # Get the hidden states from the last layer
                hidden_states = outputs.last_hidden_state.squeeze(0)
                
                # Calculate embedding-based attention
                for i in range(n_tokens):
                    for j in range(n_tokens):
                        # Skip if the same token
                        if i == j:
                            attention[i, j] = 1.0  # Self-attention
                            continue
                        
                        # Calculate cosine similarity between token embeddings
                        # Use the min to avoid index errors
                        i_idx = min(i, hidden_states.shape[0]-1)
                        j_idx = min(j, hidden_states.shape[0]-1)
                        
                        token_i_embedding = hidden_states[i_idx]
                        token_j_embedding = hidden_states[j_idx]
                        
                        similarity = torch.cosine_similarity(
                            token_i_embedding.unsqueeze(0),
                            token_j_embedding.unsqueeze(0)
                        ).item()
                        
                        # Scale similarity to a positive value
                        attention[i, j] = max(0, similarity)
                
                # Apply positional weighting (closer tokens get more attention)
                positional_weight = np.zeros((n_tokens, n_tokens))
                for i in range(n_tokens):
                    for j in range(n_tokens):
                        if i != j:
                            # Exponential decay with distance
                            distance = abs(i - j)
                            positional_weight[i, j] = np.exp(-distance / 5)  # 5 is a scaling factor
                
                # Add syntactic structure awareness using spaCy
                doc = self._nlp(text)
                syntactic_weight = np.zeros((n_tokens, n_tokens))
                
                # Map tokens to spaCy tokens (approximate)
                for i, token_i in enumerate(tokens):
                    for j, token_j in enumerate(tokens):
                        # Find if tokens are syntactically related
                        for spacy_token in doc:
                            # Very simplified check - in production you would use more sophisticated matching
                            if token_i.lower() in spacy_token.text.lower() and token_j.lower() in spacy_token.head.text.lower():
                                syntactic_weight[i, j] += 0.5
                            elif token_j.lower() in spacy_token.text.lower() and token_i.lower() in spacy_token.head.text.lower():
                                syntactic_weight[i, j] += 0.5
                
                # Combine the different attention components
                combined_attention = (
                    0.4 * attention +      # Semantic similarity
                    0.4 * positional_weight +  # Positional proximity
                    0.2 * syntactic_weight     # Syntactic relationships
                )
                
                # Normalize the attention weights
                row_sums = combined_attention.sum(axis=1, keepdims=True)
                normalized_attention = combined_attention / np.maximum(row_sums, 1e-10)
                
                return normalized_attention
                
            except Exception as e:
                logger.warning(f"Advanced attention flow calculation failed: {e}", exc_info=True)
                raise
                
        except (ImportError, Exception) as e:
            # Fallback to simple method if the advanced method fails
            logger.warning(f"Using fallback attention flow method due to error: {str(e)}")
            
            # Get tokens from the LLM (retry this call)
            response = await self.query(text)
            if response.tokens is None:
                raise ValueError("Tokens not returned from LLM")
            
            tokens = response.tokens
            n_tokens = len(tokens)
            
            # Create a simpler fallback attention matrix
            attention = np.zeros((n_tokens, n_tokens))
            for i in range(n_tokens):
                for j in range(n_tokens):
                    # Simple positional decay with self-attention
                    if i == j:
                        attention[i, j] = 1.0  # Self-attention
                    else:
                        attention[i, j] = 1.0 / (1.0 + abs(i - j))
                        
            return attention
        
    async def batch_query(self, prompts: List[str], **kwargs) -> List[LLMResponse]:
        """
        Process multiple prompts in parallel
        
        Args:
            prompts: List of prompts to process
            **kwargs: Additional parameters to pass to query method
            
        Returns:
            List of LLMResponse objects
        """
        try:
            # Process prompts concurrently
            return await asyncio.gather(*[
                self.query(prompt, **kwargs) for prompt in prompts
            ])
        except Exception as e:
            logger.error(f"Batch query failed: {e}")
            # If one query fails, the whole gather will fail
            # Re-try sequentially to get results for prompts that succeed
            results = []
            for prompt in prompts:
                try:
                    results.append(await self.query(prompt, **kwargs))
                except Exception as prompt_error:
                    logger.warning(f"Individual prompt failed: {prompt_error}")
                    # Add None for failed prompts
                    results.append(None)
            return results
    
    async def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get embeddings for text(s)
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        if not isinstance(texts, list):
            texts = [texts]
            
        # Check cache for each text
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            # Use hash as cache key
            cache_key = hash(text)
            if cache_key in self._embeddings_cache:
                results.append(self._embeddings_cache[cache_key])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # If we have uncached texts, get their embeddings
        if uncached_texts:
            try:
                async with self.session.post(
                    f"{self.api_endpoint}/embeddings",
                    headers=self._get_default_headers(),
                    json={
                        "model": "text-embedding-3-small",  # Default embedding model
                        "input": uncached_texts
                    }
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        error = handle_api_error(
                            response.status,
                            response_text,
                            {"model": "text-embedding-3-small"}
                        )
                        log_exception(error)
                        raise error
                    
                    data = await response.json()
                    
                    # Process and cache embeddings
                    for i, embedding_data in enumerate(data.get('data', [])):
                        embedding = np.array(embedding_data.get('embedding', []))
                        text_idx = uncached_indices[i]
                        
                        # Cache the embedding
                        cache_key = hash(texts[text_idx])
                        self._embeddings_cache[cache_key] = embedding
                        
                        # Insert at the correct position
                        if text_idx < len(results):
                            results.insert(text_idx, embedding)
                        else:
                            results.append(embedding)
            
            except Exception as e:
                error = LLMAPIError(
                    f"Failed to get embeddings: {str(e)}",
                    500,
                    str(e),
                    {"num_texts": len(uncached_texts)}
                )
                log_exception(error)
                raise error
        
        # Stack embeddings into a single array
        return np.stack(results) if len(results) > 1 else results[0]
    
    def format_error(self, e: Exception) -> Dict[str, Any]:
        """
        Format an error for user display
        
        Args:
            e: The exception to format
            
        Returns:
            Dictionary with formatted error information
        """
        return format_error_for_user(e)


# For testing the module independently
async def test_llm_interface():
    print("Testing LLM Interface...")
    api_key = "your-api-key-here"  # Replace with your actual API key
    
    try:
        async with LLMInterface(api_key=api_key) as llm:
            # Test basic query
            response = await llm.query("Explain how neural networks work in one paragraph.")
            print(f"Response: {response.text}\n")
            
            # Test token probabilities
            try:
                probs = await llm.get_token_probabilities("What is machine learning?")
                print(f"Token probabilities sample: {list(probs.items())[:5]}\n")
            except Exception as e:
                print(f"Token probabilities error: {e}\n")
            
            # Test attention flow
            try:
                attention = await llm.get_attention_flow("What is deep learning?")
                print(f"Attention matrix shape: {attention.shape}\n")
            except Exception as e:
                print(f"Attention flow error: {e}\n")
                
            print("LLM Interface tests completed successfully!")
            
    except Exception as e:
        print(f"Error testing LLM Interface: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm_interface())