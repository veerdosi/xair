"""
Tests for the configuration module.
"""

import os
import json
import pytest
from backend.utils.config import XAIRConfig, CGRTConfig

def test_config_init():
    """Test configuration initialization."""
    config = XAIRConfig()
    
    # Check default values
    assert config.model_name_or_path == "meta-llama/Llama-3.2-1B"
    assert config.device == "auto"
    assert config.max_tokens == 256
    assert config.output_dir == "output"
    assert config.verbose is False
    
    # Check component configs
    assert config.cgrt.model_name_or_path == config.model_name_or_path
    assert config.cgrt.device == config.device
    assert config.cgrt.max_new_tokens == config.max_tokens
    assert config.cgrt.output_dir == os.path.join(config.output_dir, "cgrt")
    
    assert config.counterfactual.output_dir == os.path.join(config.output_dir, "counterfactual")
    assert config.knowledge_graph.output_dir == os.path.join(config.output_dir, "knowledge_graph")

def test_config_save_load(temp_output_dir):
    """Test saving and loading configuration."""
    # Create a custom config
    custom_config = XAIRConfig(
        model_name_or_path="custom/model",
        device="cuda",
        max_tokens=512,
        output_dir=temp_output_dir,
        verbose=True
    )
    
    # Customize component configs
    custom_config.cgrt.temperatures = [0.5, 1.0]
    custom_config.cgrt.paths_per_temp = 2
    custom_config.counterfactual.top_k_tokens = 10
    custom_config.knowledge_graph.min_similarity_threshold = 0.7
    
    # Save the config
    config_path = os.path.join(temp_output_dir, "config.json")
    custom_config.save(config_path)
    
    # Check that the file exists
    assert os.path.exists(config_path)
    
    # Load the config
    loaded_config = XAIRConfig.load(config_path)
    
    # Check that all values match
    assert loaded_config.model_name_or_path == custom_config.model_name_or_path
    assert loaded_config.device == custom_config.device
    assert loaded_config.max_tokens == custom_config.max_tokens
    assert loaded_config.output_dir == custom_config.output_dir
    assert loaded_config.verbose == custom_config.verbose
    
    # Check component configs
    assert loaded_config.cgrt.temperatures == custom_config.cgrt.temperatures
    assert loaded_config.cgrt.paths_per_temp == custom_config.cgrt.paths_per_temp
    assert loaded_config.counterfactual.top_k_tokens == custom_config.counterfactual.top_k_tokens
    assert loaded_config.knowledge_graph.min_similarity_threshold == custom_config.knowledge_graph.min_similarity_threshold

def test_config_from_args():
    """Test creating config from command line arguments."""
    # Create a mock args object
    class MockArgs:
        def __init__(self):
            self.model = "test/model"
            self.device = "mps"
            self.max_tokens = 128
            self.temperatures = "0.3,0.8"
            self.paths_per_temp = 3
            self.counterfactual_tokens = 8
            self.attention_threshold = 0.4
            self.max_counterfactuals = 15
            self.kg_use_local_model = True
            self.kg_similarity_threshold = 0.8
            self.kg_skip = True
            self.output_dir = "test_output"
            self.verbose = True
    
    # Create config from args
    args = MockArgs()
    config = XAIRConfig.from_args(args)
    
    # Check that values were properly extracted
    assert config.model_name_or_path == "test/model"
    assert config.device == "mps"
    assert config.max_tokens == 128
    assert config.output_dir == "test_output"
    assert config.verbose is True
    assert config.skip_kg is True
    
    # Check component configs
    assert config.cgrt.temperatures == [0.3, 0.8]
    assert config.cgrt.paths_per_temp == 3
    assert config.counterfactual.top_k_tokens == 8
    assert config.counterfactual.min_attention_threshold == 0.4
    assert config.counterfactual.max_total_candidates == 15
    assert config.knowledge_graph.use_local_model is True
    assert config.knowledge_graph.min_similarity_threshold == 0.8