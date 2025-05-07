"""
Configuration module for the XAIR system.
Provides centralized configuration management across modules.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

@dataclass
class CGRTConfig:
    """Configuration for CGRT component."""
    model_name_or_path: str = "meta-llama/Llama-3.2-1B"
    device: str = "auto"
    temperatures: list = field(default_factory=lambda: [0.2, 0.7, 1.0])
    paths_per_temp: int = 1
    max_new_tokens: int = 256
    output_dir: str = "output/cgrt"
    num_layers_to_analyze: int = 5
    attention_aggregation: str = "mean"
    attention_weight_by_layer: bool = True
    verbose: bool = False

@dataclass
class CounterfactualConfig:
    """Configuration for Counterfactual component."""
    top_k_tokens: int = 5
    min_attention_threshold: float = 0.3
    min_semantic_similarity: float = 0.5
    max_candidates_per_node: int = 3
    max_total_candidates: int = 20
    filter_by_pos: bool = True
    output_dir: str = "output/counterfactual"
    verbose: bool = False

@dataclass
class KnowledgeGraphConfig:
    """Configuration for Knowledge Graph component."""
    model_name: str = "all-MiniLM-L6-v2"
    use_local_model: bool = True
    wikidata_endpoint: str = "https://query.wikidata.org/sparql"
    cache_dir: str = "output/knowledge_graph/cache"
    cache_expiry: int = 86400  # 24 hours
    use_gpu: bool = False
    min_similarity_threshold: float = 0.6
    output_dir: str = "output/knowledge_graph"
    verbose: bool = False

@dataclass
class XAIRConfig:
    """Main configuration for XAIR system."""
    model_name_or_path: str = "meta-llama/Llama-3.2-1B"
    device: str = "auto"
    max_tokens: int = 256
    output_dir: str = "output"
    verbose: bool = False
    # Performance optimization parameters
    performance: str = "balanced"  # "max_speed", "balanced", "max_quality"
    fast_mode: bool = False
    fast_init: bool = False
    cgrt: CGRTConfig = field(default_factory=CGRTConfig)
    counterfactual: CounterfactualConfig = field(default_factory=CounterfactualConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    skip_kg: bool = False

    def __post_init__(self):
        # Update component configs to match main config
        self.cgrt.model_name_or_path = self.model_name_or_path
        self.cgrt.device = self.device
        self.cgrt.max_new_tokens = self.max_tokens
        self.cgrt.output_dir = os.path.join(self.output_dir, "cgrt")
        self.cgrt.verbose = self.verbose

        self.counterfactual.output_dir = os.path.join(self.output_dir, "counterfactual")
        self.counterfactual.verbose = self.verbose

        self.knowledge_graph.output_dir = os.path.join(self.output_dir, "knowledge_graph")
        self.knowledge_graph.verbose = self.verbose

    def save(self, path: str) -> None:
        """Save configuration to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Saved configuration to {path}")

    @classmethod
    def load(cls, path: str) -> 'XAIRConfig':
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)

        # Extract component configs
        cgrt_dict = config_dict.pop("cgrt", {})
        counterfactual_dict = config_dict.pop("counterfactual", {})
        kg_dict = config_dict.pop("knowledge_graph", {})

        # Create config object
        config = cls(**config_dict)

        # Update component configs
        if cgrt_dict:
            config.cgrt = CGRTConfig(**cgrt_dict)
        if counterfactual_dict:
            config.counterfactual = CounterfactualConfig(**counterfactual_dict)
        if kg_dict:
            config.knowledge_graph = KnowledgeGraphConfig(**kg_dict)

        return config

    @classmethod
    def from_args(cls, args) -> 'XAIRConfig':
        """Create configuration from argparse Namespace."""
        # Extract temperatures from string
        if hasattr(args, "temperatures") and isinstance(args.temperatures, str):
            temperatures = [float(t) for t in args.temperatures.split(",")]
        else:
            temperatures = [0.2, 0.7, 1.0]

        # Create main config
        config = cls(
            model_name_or_path=args.model,
            device=args.device,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            verbose=args.verbose,
            # Performance settings
            performance=args.performance if hasattr(args, "performance") else "balanced",
            fast_mode=args.fast_mode if hasattr(args, "fast_mode") else False,
            fast_init=args.fast_init if hasattr(args, "fast_init") else False,
            skip_kg=args.kg_skip if hasattr(args, "kg_skip") else False
        )

        # Update CGRT config
        config.cgrt.temperatures = temperatures
        config.cgrt.paths_per_temp = args.paths_per_temp

        # Update Counterfactual config
        config.counterfactual.top_k_tokens = args.counterfactual_tokens
        config.counterfactual.min_attention_threshold = args.attention_threshold
        config.counterfactual.max_total_candidates = args.max_counterfactuals

        # Update Knowledge Graph config
        if hasattr(args, "kg_use_local_model"):
            config.knowledge_graph.use_local_model = args.kg_use_local_model
        if hasattr(args, "kg_similarity_threshold"):
            config.knowledge_graph.min_similarity_threshold = args.kg_similarity_threshold

        return config
