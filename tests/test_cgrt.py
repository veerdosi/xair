import pytest
import asyncio
import networkx as nx
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock

from backend.cgrt import CGRTGenerator, CrossGeneration, GenerationPath
from backend.llm_interface import LLMInterface, LLMResponse

# Mock data for testing
@pytest.fixture
def mock_generation_paths():
    """Create mock generation paths for testing"""
    path1 = GenerationPath(
        tokens=["The", "quick", "brown", "fox"],
        probabilities=[0.9, 0.8, 0.7, 0.6],
        attention_maps=[np.ones((1, 1)) for _ in range(4)],
        divergence_points=[2],
        importance_score=0.75
    )
    
    path2 = GenerationPath(
        tokens=["The", "quick", "red", "fox"],
        probabilities=[0.9, 0.8, 0.5, 0.6],
        attention_maps=[np.ones((1, 1)) for _ in range(4)],
        divergence_points=[2],
        importance_score=0.65
    )
    
    return [path1, path2]

@pytest.fixture
def mock_cross_generation(mock_generation_paths):
    """Create a mock CrossGeneration object"""
    return CrossGeneration(
        paths=mock_generation_paths,
        shared_prefix=["The", "quick"],
        divergence_map={2: {"brown", "red"}},
        attention_flow=np.ones((4, 4))
    )

@pytest.fixture
def mock_llm():
    """Create a mock LLM interface"""
    mock = AsyncMock(spec=LLMInterface)
    
    # Mock response for query
    mock_response = LLMResponse(
        text="The quick brown fox",
        logits=np.array([-1.2, -0.8, -1.5, -0.5]),
        tokens=["The", "quick", "brown", "fox"],
        attention=None
    )
    mock.query.return_value = mock_response
    
    # Mock token probabilities
    mock.get_token_probabilities.return_value = {
        "brown": 0.7,
        "red": 0.5,
        "fast": 0.3
    }
    
    # Mock attention flow
    mock.get_attention_flow.return_value = np.ones((4, 4))
    
    return mock

@pytest.mark.asyncio
async def test_generate_cross_paths(mock_llm, mock_generation_paths):
    """Test generating cross paths"""
    # Create a mock generate_single_path method
    with patch.object(CGRTGenerator, '_generate_single_path', new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = mock_generation_paths
        
        generator = CGRTGenerator(mock_llm)
        result = await generator.generate_cross_paths("Test prompt")
        
        # Verify that the method was called with different temperatures
        assert mock_generate.call_count == generator.num_generations
        
        # Check the result
        assert len(result.paths) == len(mock_generation_paths)
        assert result.shared_prefix == ["The", "quick"]
        assert 2 in result.divergence_map
        assert result.divergence_map[2] == {"brown", "red"}

@pytest.mark.asyncio
async def test_generate_single_path(mock_llm):
    """Test generating a single path"""
    # Mock LLM responses
    mock_llm.query.return_value = LLMResponse(
        text="The quick brown fox",
        logits=np.array([-1.2, -0.8, -1.5, -0.5]),
        tokens=["The", "quick", "brown", "fox"],
        attention=None
    )
    
    mock_llm.get_token_probabilities.return_value = {
        "brown": 0.7,
        "red": 0.5,
        "fast": 0.3
    }
    
    mock_llm.get_attention_flow.return_value = np.ones((4, 4))
    
    generator = CGRTGenerator(mock_llm)
    path = await generator._generate_single_path("Test prompt", 0.8, 4)
    
    # Verify the path
    assert isinstance(path, GenerationPath)
    assert len(path.tokens) == 4
    assert len(path.probabilities) == 4
    assert len(path.attention_maps) == 4
    assert path.importance_score > 0

def test_find_shared_prefix(mock_generation_paths):
    """Test finding shared prefix across generations"""
    generator = CGRTGenerator(mock_llm)
    prefix = generator._find_shared_prefix(mock_generation_paths)
    
    # Verify shared prefix
    assert prefix == ["The", "quick"]

def test_map_divergences(mock_generation_paths):
    """Test mapping divergence points"""
    generator = CGRTGenerator(mock_llm)
    shared_prefix = ["The", "quick"]
    divergence_map = generator._map_divergences(mock_generation_paths, shared_prefix)
    
    # Verify divergence map
    assert 2 in divergence_map
    assert divergence_map[2] == {"brown", "red"}

def test_aggregate_attention(mock_generation_paths):
    """Test attention aggregation"""
    generator = CGRTGenerator(mock_llm)
    attention = generator._aggregate_attention(mock_generation_paths)
    
    # Verify attention shape
    assert attention.shape == (1, 1)
    
    # Verify values
    assert np.all(attention == 1.0)  # All values are 1.0 due to mock data

def test_calculate_path_importance():
    """Test path importance calculation"""
    generator = CGRTGenerator(mock_llm)
    probs = [0.9, 0.8, 0.7]
    attn_maps = [np.ones((2, 2)) for _ in range(3)]
    
    importance = generator._calculate_path_importance(probs, attn_maps)
    
    # Verify importance score is in expected range
    assert 0 <= importance <= 1
    # Specific value based on the formula in _calculate_path_importance
    expected = 0.7 * np.mean(probs) + 0.3 * 1.0  # second term is mean of attention maps (all 1s)
    assert abs(importance - expected) < 1e-10

if __name__ == "__main__":
    pytest.main()