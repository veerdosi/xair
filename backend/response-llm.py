from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    text: str
    logits: Optional[np.ndarray] = None
    attention: Optional[np.ndarray] = None
    tokens: Optional[List[str]] = None

class BaseLLMInterface(ABC):
    @abstractmethod
    async def query(self, prompt: str) -> LLMResponse:
        pass

    @abstractmethod
    async def get_token_probabilities(self, text: str) -> Dict[str, float]:
        pass

    @abstractmethod
    async def get_attention_flow(self, text: str) -> np.ndarray:
        pass

class LLMInterface(BaseLLMInterface):
    def __init__(
        self,
        api_key: str,
        api_endpoint: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        return_logits: bool = True,
        return_attention: bool = True,
    ):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.return_logits = return_logits
        self.return_attention = return_attention
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def query(self, prompt: str) -> LLMResponse:
        """
        Send a query to the ChatGPT-4o API and return the response.
        Includes retry logic for resilience.
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async with context.")

        try:
            async with self.session.post(
                f"{self.api_endpoint}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "logprobs": self.return_logits,
                    "stream": False,
                    "logit_bias": {}  # Optional parameter
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")
                
                data = await response.json()
                
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
                
                # OpenAI doesn't directly provide attention matrices, so this is null
                attention = None
                
                return LLMResponse(
                    text=text,
                    logits=logits,
                    attention=attention,
                    tokens=tokens,
                )

        except aiohttp.ClientError as e:
            logger.error(f"Network error during API call: {e}")
            raise

    async def get_token_probabilities(self, text: str) -> Dict[str, float]:
        """
        Get token probabilities for a given text.
        Returns a dictionary mapping tokens to their probabilities.
        """
        response = await self.query(text)
        if response.logits is None or response.tokens is None:
            raise ValueError("Logits or tokens not returned from LLM")

        # Convert logprobs to probabilities
        # logprob = log(p), so p = exp(logprob)
        probs = np.exp(response.logits)

        return dict(zip(response.tokens, probs.tolist()))

    async def get_attention_flow(self, text: str) -> np.ndarray:
        """
        Get attention flow matrices for a given text.
        Note: OpenAI API doesn't provide attention matrices directly.
        This method returns a placeholder or raises an exception.
        """
        raise NotImplementedError(
            "ChatGPT-4o API does not provide attention matrices. "
            "Consider using embeddings or other alternatives for analyzing token relationships."
        )

    async def batch_query(self, prompts: List[str]) -> List[LLMResponse]:
        """
        Process multiple prompts in parallel.
        """
        return await asyncio.gather(*(self.query(prompt) for prompt in prompts))

async def main():
    # Example usage
    api_key = "your-api-key-here"
    async with ChatGPT4oInterface(api_key=api_key) as client:
        response = await client.query("Tell me about artificial intelligence.")
        print(f"Response: {response.text}")
        
        # Get token probabilities
        try:
            probs = await client.get_token_probabilities("What is machine learning?")
            print(f"Token probabilities: {probs}")
        except NotImplementedError as e:
            print(f"Token probabilities error: {e}")
        
        # Try to get attention flow (will raise NotImplementedError)
        try:
            attention = await client.get_attention_flow("What is deep learning?")
        except NotImplementedError as e:
            print(f"Attention flow error: {e}")

if __name__ == "__main__":
    asyncio.run(main())