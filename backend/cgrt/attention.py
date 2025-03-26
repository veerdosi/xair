"""
Attention Analysis module for the XAIR system.
Processes attention matrices to calculate importance scores.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import torch
import logging

logger = logging.getLogger(__name__)

class AttentionAnalyzer:
    """Analyzes attention patterns in LLM outputs."""
    
    def __init__(
        self,
        num_layers_to_analyze: int = 5,  # Analyze only the last N layers by default
        attention_aggregation: str = "mean",  # "mean", "max", or "weighted"
        attention_weight_by_layer: bool = True,  # Give higher weight to later layers
        verbose: bool = False
    ):
        """
        Initialize the attention analyzer.
        
        Args:
            num_layers_to_analyze: Number of layers to analyze (from the end)
            attention_aggregation: Method to aggregate attention heads
            attention_weight_by_layer: Whether to weight attention by layer
            verbose: Whether to log detailed information
        """
        self.num_layers_to_analyze = num_layers_to_analyze
        self.attention_aggregation = attention_aggregation
        self.attention_weight_by_layer = attention_weight_by_layer
        self.verbose = verbose
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
    
    def analyze_attention(
        self,
        attentions: List[Any],
        token_ids: List[int],
        input_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[int, float]]:
        """
        Analyze attention matrices to calculate token importance.
        
        Args:
            attentions: List of attention matrices from model
            token_ids: List of token IDs in the sequence
            input_length: Length of the prompt (to separate prompt from generation)
            
        Returns:
            Tuple of (attention_flow, importance_scores)
        """
        # Check if we have attention matrices
        if not attentions:
            logger.warning("No attention matrices provided")
            return {}, {}
        
        # Convert attentions to numpy if they are torch tensors
        processed_attentions = []
        for layer_attn in attentions:
            if isinstance(layer_attn, torch.Tensor):
                processed_attentions.append(layer_attn.detach().cpu().numpy())
            elif isinstance(layer_attn, np.ndarray):
                processed_attentions.append(layer_attn)
            elif isinstance(layer_attn, (list, tuple)):
                # Handle nested lists/tuples of tensors
                layer_processed = []
                for head_attn in layer_attn:
                    if isinstance(head_attn, torch.Tensor):
                        layer_processed.append(head_attn.detach().cpu().numpy())
                    else:
                        layer_processed.append(head_attn)
                processed_attentions.append(layer_processed)
        
        # Determine which layers to analyze
        num_layers = len(processed_attentions)
        start_layer = max(0, num_layers - self.num_layers_to_analyze)
        layers_to_analyze = processed_attentions[start_layer:]
        
        # Calculate attention flow
        attention_flow = self._calculate_attention_flow(
            layers_to_analyze,
            token_ids,
            input_length
        )
        
        # Calculate importance scores
        importance_scores = self._calculate_importance_scores(
            attention_flow,
            token_ids,
            input_length
        )
        
        return attention_flow, importance_scores
    
    def _calculate_attention_flow(
        self,
        attention_layers: List[Any],
        token_ids: List[int],
        input_length: int
    ) -> Dict[str, np.ndarray]:
        """
        Calculate attention flow between tokens.
        
        Args:
            attention_layers: List of attention matrices
            token_ids: List of token IDs
            input_length: Length of the prompt
            
        Returns:
            Dictionary mapping (source, target) to attention weight
        """
        # Initialize attention flow
        attention_flow = {}
        
        # Process each layer
        for layer_idx, layer_attn in enumerate(attention_layers):
            # Calculate layer weight if enabled
            layer_weight = 1.0
            if self.attention_weight_by_layer:
                # Higher weight for later layers
                layer_weight = (layer_idx + 1) / len(attention_layers)
            
            # Process each head in the layer
            num_heads = len(layer_attn) if isinstance(layer_attn, (list, tuple)) else 1
            
            if num_heads > 1:
                # Multi-head attention
                heads_attn = layer_attn
            else:
                # Single head or already aggregated
                heads_attn = [layer_attn]
            
            # Process each head
            for head_idx, head_attn in enumerate(heads_attn):
                # Get the attention matrix for the generated tokens
                # Attention shape: [batch, heads, sequence_length, sequence_length]
                if len(head_attn.shape) == 4:
                    # Take the first batch and specified head
                    attn_matrix = head_attn[0, 0]  # [sequence_length, sequence_length]
                else:
                    attn_matrix = head_attn
                
                # Focus on attention to/from the generated tokens
                gen_start = input_length
                gen_end = len(token_ids)
                
                # Extract the relevant part of the attention matrix
                gen_attn = attn_matrix[gen_start:gen_end, :]
                
                # Record attention flow for each token pair
                for i in range(gen_attn.shape[0]):
                    source_pos = gen_start + i
                    source_token = token_ids[source_pos] if source_pos < len(token_ids) else -1
                    
                    for j in range(attn_matrix.shape[1]):
                        target_token = token_ids[j] if j < len(token_ids) else -1
                        
                        attention_weight = gen_attn[i, j] * layer_weight
                        
                        # Create key for the token pair
                        key = f"{source_token}_{target_token}"
                        
                        # Update attention flow
                        if key in attention_flow:
                            if self.attention_aggregation == "mean":
                                # Average across heads and layers
                                attention_flow[key] = (attention_flow[key] + attention_weight) / 2
                            elif self.attention_aggregation == "max":
                                # Take maximum attention weight
                                attention_flow[key] = max(attention_flow[key], attention_weight)
                            else:  # "weighted" or any other
                                # Add weighted attention
                                attention_flow[key] += attention_weight
                        else:
                            attention_flow[key] = attention_weight
        
        return attention_flow
    
    def _calculate_importance_scores(
        self,
        attention_flow: Dict[str, np.ndarray],
        token_ids: List[int],
        input_length: int
    ) -> Dict[int, float]:
        """
        Calculate importance scores for tokens based on attention flow.
        
        Args:
            attention_flow: Dictionary of attention flow
            token_ids: List of token IDs
            input_length: Length of the prompt
            
        Returns:
            Dictionary mapping token position to importance score
        """
        # Initialize importance scores
        importance_scores = {}
        
        # Focus on generated tokens
        gen_tokens = token_ids[input_length:]
        
        # Calculate incoming attention for each token
        for i, token_id in enumerate(gen_tokens):
            pos = input_length + i
            incoming_attention = 0.0
            count = 0
            
            # Sum attention flowing to this token
            for key, weight in attention_flow.items():
                source_token, target_token = map(int, key.split("_"))
                if target_token == token_id:
                    incoming_attention += weight
                    count += 1
            
            # Calculate average incoming attention
            avg_attention = incoming_attention / max(1, count)
            
            # Add position-based importance (later tokens can be more important)
            position_importance = i / max(1, len(gen_tokens) - 1)
            
            # Combine for final importance score
            importance = 0.7 * avg_attention + 0.3 * position_importance
            
            # Store the importance score
            importance_scores[pos] = importance
        
        # Normalize importance scores to [0, 1]
        if importance_scores:
            max_importance = max(importance_scores.values())
            min_importance = min(importance_scores.values())
            range_importance = max_importance - min_importance
            
            if range_importance > 0:
                normalized_scores = {
                    pos: (score - min_importance) / range_importance 
                    for pos, score in importance_scores.items()
                }
                importance_scores = normalized_scores
        
        return importance_scores
    
    def update_tree_with_attention(
        self, 
        tree_builder,
        paths: List[Dict[str, Any]]
    ):
        """
        Update the tree with attention-based importance scores.
        
        Args:
            tree_builder: CGRTBuilder instance
            paths: List of generation results
        """
        # Check if we have paths
        if not paths:
            logger.warning("No paths provided for attention analysis")
            return
        
        # Process each path
        for path_idx, path in enumerate(paths):
            # Check if we have attention matrices
            if "attentions" not in path:
                logger.warning(f"Path {path_idx} does not have attention matrices")
                continue
            
            # Get token IDs
            if "generated_ids" not in path:
                logger.warning(f"Path {path_idx} does not have generated_ids")
                continue
                
            input_ids = path.get("input_ids", None)
            if input_ids is None:
                logger.warning(f"Path {path_idx} does not have input_ids")
                continue
                
            # Calculate the offset (prompt length)
            input_length = input_ids.shape[1] if hasattr(input_ids, "shape") else len(input_ids[0])
            
            # Get the token IDs
            generated_ids = path["generated_ids"]
            if hasattr(generated_ids, "shape"):  # For torch tensors
                token_ids = generated_ids[0].tolist()
            else:  # For lists or numpy arrays
                token_ids = generated_ids[0]
            
            # Analyze attention
            attention_flow, importance_scores = self.analyze_attention(
                path["attentions"],
                token_ids,
                input_length
            )
            
            # Update the tree nodes
            for pos, importance in importance_scores.items():
                # Find nodes at this position
                for node_id, node in tree_builder.nodes.items():
                    if node.position == pos - input_length:  # Adjust for prompt length
                        # Update attention score
                        node.attention_score = importance
                        
                        # Use attention for importance if not already set
                        if node.importance_score == 0.0:
                            node.importance_score = importance
                        
                        # Update the graph
                        if node_id in tree_builder.graph:
                            tree_builder.graph.nodes[node_id]["attention_score"] = importance
                            
                            if tree_builder.graph.nodes[node_id]["importance_score"] == 0.0:
                                tree_builder.graph.nodes[node_id]["importance_score"] = importance
        
        logger.info("Updated tree with attention-based importance scores")