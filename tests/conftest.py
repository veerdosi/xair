"""
Pytest configuration for XAIR system tests.
"""

import os
import sys
import pytest
import logging
import tempfile
import shutil

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fixtures
@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.eos_token = "[EOS]"
            self.pad_token = None
        
        def __call__(self, text, return_tensors="pt"):
            import torch
            # Simplified tokenization
            tokens = text.split()
            token_ids = [i + 1 for i in range(len(tokens))]
            
            class TokenizerOutput:
                def __init__(self, input_ids, attention_mask):
                    self.input_ids = input_ids
                    self.attention_mask = attention_mask
                
                def to(self, device):
                    return self
            
            input_ids = torch.tensor([token_ids])
            attention_mask = torch.ones_like(input_ids)
            
            return TokenizerOutput(input_ids, attention_mask)
        
        def decode(self, token_ids, skip_special_tokens=True):
            # Simple decoding
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            
            if isinstance(token_ids, list) and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            
            # Convert token IDs to text
            return " ".join([f"token_{i}" for i in token_ids])
        
        def batch_decode(self, token_ids, skip_special_tokens=True):
            if isinstance(token_ids, list) and len(token_ids) == 0:
                return []
                
            # Handle single item or list of items
            if not isinstance(token_ids[0], list) and not hasattr(token_ids[0], "tolist"):
                return [self.decode(token_ids, skip_special_tokens)]
            
            return [self.decode(ids, skip_special_tokens) for ids in token_ids]
    
    return MockTokenizer()

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    class MockOutput:
        def __init__(self, sequences, scores=None, hidden_states=None, attentions=None):
            import torch
            self.sequences = sequences
            self.scores = scores
            self.hidden_states = hidden_states
            self.attentions = attentions
    
    class MockModel:
        def __init__(self):
            import torch
            self.device = "cpu"
        
        def generate(
            self,
            input_ids,
            output_hidden_states=False,
            output_attentions=False,
            return_dict_in_generate=True,
            max_new_tokens=10,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            output_scores=False,
            attention_mask=None,
        ):
            import torch
            import numpy as np
            
            # Generate some random output IDs
            batch_size = input_ids.shape[0]
            input_len = input_ids.shape[1]
            gen_len = min(max_new_tokens, 5)  # Limit to 5 tokens for testing
            
            # Create sequences: original input + new tokens
            sequences = torch.cat([
                input_ids,
                torch.randint(100, 1000, (batch_size, gen_len))
            ], dim=1)
            
            # Create dummy scores if requested
            scores = None
            if output_scores:
                scores = [torch.randn(batch_size, 1000) for _ in range(gen_len)]
            
            # Create dummy hidden states if requested
            hidden_states = None
            if output_hidden_states:
                # Create 12 layers of hidden states
                hidden_states = tuple(
                    tuple(torch.randn(batch_size, sequences.shape[1], 128) for _ in range(2))
                    for _ in range(12)
                )
            
            # Create dummy attention matrices if requested
            attentions = None
            if output_attentions:
                # Create 12 layers of attention matrices (4 heads)
                seq_len = sequences.shape[1]
                attentions = tuple(
                    tuple(torch.rand(batch_size, 4, seq_len, seq_len) for _ in range(2))
                    for _ in range(12)
                )
            
            return MockOutput(sequences, scores, hidden_states, attentions)
        
        def to(self, device):
            self.device = device
            return self
    
    return MockModel()

@pytest.fixture
def mock_llm_interface(mock_model, mock_tokenizer):
    """Create a mock LLM interface for testing."""
    from backend.models.llm_interface import LlamaInterface
    
    # Create a real interface but with mock components
    interface = LlamaInterface(
        model_name_or_path="mock-llama",
        device="cpu"
    )
    
    # Replace with mocks
    interface.model = mock_model
    interface.tokenizer = mock_tokenizer
    
    return interface