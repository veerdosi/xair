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

class DeepSeekInterface(BaseLLMInterface):
    def __init__(
        self,
        api_key: str,
        api_endpoint: str = "https://api.deepseek.ai/v1",
        model_name: str = "deepseek-r1",
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
        Send a query to the DeepSeek API and return the response.
        Includes retry logic for resilience.
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async with context.")

        try:
            async with self.session.post(
                f"{self.api_endpoint}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "return_logits": self.return_logits,
                    "return_attention": self.return_attention,
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")
                
                data = await response.json()
                return LLMResponse(
                    text=data["text"],
                    logits=np.array(data.get("logits")) if data.get("logits") else None,
                    attention=np.array(data.get("attention")) if data.get("attention") else None,
                    tokens=data.get("tokens"),
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

        # Apply softmax to get probabilities
        exp_logits = np.exp(response.logits - np.max(response.logits))
        probs = exp_logits / exp_logits.sum()

        return dict(zip(response.tokens, probs.tolist()))

    async def get_attention_flow(self, text: str) -> np.ndarray:
        """
        Get attention flow matrices for a given text.
        Returns a numpy array of attention weights.
        """
        response = await self.query(text)
        if response.attention is None:
            raise ValueError("Attention matrices not returned from LLM")
        return response.attention

    async def batch_query(self, prompts: List[str]) -> List[LLMResponse]:
        """
        Process multiple prompts in parallel.
        """
        return await asyncio.gather(*(self.query(prompt) for prompt in prompts))

if __name__ == "__main__":
    asyncio.run(main())
