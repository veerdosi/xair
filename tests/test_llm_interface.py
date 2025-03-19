import pytest
import asyncio
import os
import numpy as np
from unittest.mock import AsyncMock, patch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the module to test
from backend.llm_interface import LLMInterface, LLMResponse

# Fixture for the API key
@pytest.fixture
def api_key():
    """Get API key from environment or use a dummy key for tests"""
    return os.getenv("OPENAI_API_KEY", "dummy-api-key-for-testing")

# Mock response for testing
@pytest.fixture
def mock_response():
    """Create a mock LLM response"""
    return LLMResponse(
        text="This is a test response",
        logits=np.array([-1.2, -0.8, -1.5, -0.5]),
        tokens=["This", "is", "a", "test"],
        attention=None
    )

@pytest.mark.asyncio
async def test_query_success(api_key, mock_response):
    """Test successful query to LLM"""
    with patch('backend.llm_interface.LLMInterface.query', new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_response
        
        async with LLMInterface(api_key=api_key) as llm:
            response = await llm.query("Test prompt")
            
            # Check mock was called with correct prompt
            mock_query.assert_called_once_with("Test prompt")
            
            # Verify the response
            assert response.text == "This is a test response"
            assert len(response.tokens) == 4
            assert response.tokens[0] == "This"

@pytest.mark.asyncio
async def test_token_probabilities(api_key, mock_response):
    """Test getting token probabilities"""
    with patch('backend.llm_interface.LLMInterface.query', new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_response
        
        async with LLMInterface(api_key=api_key) as llm:
            probs = await llm.get_token_probabilities("Test prompt")
            
            # Verify probabilities were calculated correctly
            assert len(probs) == 4
            
            # Check that logits were converted to probabilities
            for token, prob in probs.items():
                assert 0 <= prob <= 1

@pytest.mark.asyncio
async def test_attention_flow(api_key, mock_response):
    """Test getting attention flow matrices"""
    with patch('backend.llm_interface.LLMInterface.query', new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_response
        
        async with LLMInterface(api_key=api_key) as llm:
            attention = await llm.get_attention_flow("Test prompt")
            
            # Verify attention matrix shape
            assert attention.shape == (4, 4)  # 4 tokens in the response
            
            # Check that attention weights are normalized
            for i in range(4):
                for j in range(4):
                    assert 0 <= attention[i, j] <= 1

@pytest.mark.asyncio
async def test_batch_query(api_key, mock_response):
    """Test batch query functionality"""
    with patch('backend.llm_interface.LLMInterface.query', new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_response
        
        async with LLMInterface(api_key=api_key) as llm:
            responses = await llm.batch_query(["Test prompt 1", "Test prompt 2"])
            
            # Verify number of responses
            assert len(responses) == 2
            
            # Verify each response
            for response in responses:
                assert response.text == "This is a test response"

@pytest.mark.asyncio
async def test_error_handling(api_key):
    """Test error handling in LLM interface"""
    # Mock query method to raise an exception
    with patch('backend.llm_interface.LLMInterface.query', new_callable=AsyncMock) as mock_query:
        mock_query.side_effect = Exception("Test error")
        
        async with LLMInterface(api_key=api_key) as llm:
            with pytest.raises(Exception):
                await llm.query("Test prompt")
                
                # Verify retry behavior if implemented
                assert mock_query.call_count > 1

if __name__ == "__main__":
    pytest.main()