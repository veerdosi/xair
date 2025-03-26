"""
Divergence Detection module for the XAIR system.
Identifies branch points in reasoning paths where different paths diverge.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import torch
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import logging

logger = logging.getLogger(__name__)

class DivergenceDetector:
    """Detects divergence points between multiple reasoning paths."""
    
    def __init__(
        self,
        kl_threshold: float = 0.5,
        min_prob_diff: float = 0.3,
        context_window_size: int = 5,
        adaptive_threshold: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the divergence detector.
        
        Args:
            kl_threshold: Base threshold for KL divergence
            min_prob_diff: Minimum probability difference to consider
            context_window_size: Number of tokens to include before and after a divergence point
            adaptive_threshold: Whether to use adaptive thresholding
            verbose: Whether to log detailed information
        """
        self.kl_threshold = kl_threshold
        self.min_prob_diff = min_prob_diff
        self.context_window_size = context_window_size
        self.adaptive_threshold = adaptive_threshold
        self.verbose = verbose
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
    
    def calculate_kl_divergence(
        self, 
        dist1: List[Dict[str, float]], 
        dist2: List[Dict[str, float]]
    ) -> float:
        """
        Calculate KL divergence between two token distributions.
        
        Args:
            dist1: First distribution (list of token-probability pairs)
            dist2: Second distribution (list of token-probability pairs)
            
        Returns:
            KL divergence value
        """
        # Convert to dictionary format for easier access
        d1 = {item["token_id"]: item["probability"] for item in dist1}
        d2 = {item["token_id"]: item["probability"] for item in dist2}
        
        # Get the union of all tokens
        all_tokens = set(d1.keys()) | set(d2.keys())
        
        # Create proper probability vectors (adding zeros for missing tokens)
        p = np.array([d1.get(token, 1e-10) for token in all_tokens])
        q = np.array([d2.get(token, 1e-10) for token in all_tokens])
        
        # Normalize to ensure they sum to 1
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate KL divergence
        return entropy(p, q)
    
    def calculate_jensen_shannon_distance(
        self, 
        dist1: List[Dict[str, float]], 
        dist2: List[Dict[str, float]]
    ) -> float:
        """
        Calculate Jensen-Shannon distance between two token distributions.
        
        Args:
            dist1: First distribution (list of token-probability pairs)
            dist2: Second distribution (list of token-probability pairs)
            
        Returns:
            Jensen-Shannon distance value
        """
        # Convert to dictionary format for easier access
        d1 = {item["token_id"]: item["probability"] for item in dist1}
        d2 = {item["token_id"]: item["probability"] for item in dist2}
        
        # Get the union of all tokens
        all_tokens = set(d1.keys()) | set(d2.keys())
        
        # Create proper probability vectors (adding zeros for missing tokens)
        p = np.array([d1.get(token, 1e-10) for token in all_tokens])
        q = np.array([d2.get(token, 1e-10) for token in all_tokens])
        
        # Normalize to ensure they sum to 1
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate Jensen-Shannon distance
        return jensenshannon(p, q)
    
    def get_adaptive_threshold(self, token_alternatives: List[List[Dict]]) -> float:
        """
        Calculate an adaptive threshold based on the distribution properties.
        
        Args:
            token_alternatives: List of alternative tokens and their probabilities
            
        Returns:
            Adaptive threshold value
        """
        # Calculate entropy for each position
        entropies = []
        for pos_alternatives in token_alternatives:
            probs = [alt["probability"] for alt in pos_alternatives]
            # Normalize probabilities
            prob_sum = sum(probs)
            if prob_sum < 0.99 or prob_sum > 1.01:
                probs = [p / prob_sum for p in probs]
            
            # Calculate entropy
            pos_entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
            entropies.append(pos_entropy)
        
        # Base threshold on the average entropy
        if entropies:
            avg_entropy = np.mean(entropies)
            # Scale the threshold: higher entropy → higher threshold
            return self.kl_threshold * (1 + avg_entropy / 2)
        else:
            return self.kl_threshold
    
    def detect_divergences(
        self, 
        paths: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect divergence points between multiple reasoning paths.
        
        Args:
            paths: List of generation results from LlamaInterface
            
        Returns:
            List of divergence points with metadata
        """
        if len(paths) < 2:
            logger.warning("Need at least 2 paths to detect divergences")
            return []
        
        divergence_points = []
        
        # Compare each pair of paths
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                path1 = paths[i]
                path2 = paths[j]
                
                # Check if we have token alternatives
                if "token_alternatives" not in path1 or "token_alternatives" not in path2:
                    logger.warning("Token alternatives not available in paths")
                    continue
                
                # Get the token alternatives
                alts1 = path1["token_alternatives"]
                alts2 = path2["token_alternatives"]
                
                # Determine the minimum length to compare
                min_length = min(len(alts1), len(alts2))
                
                # Set threshold
                threshold = self.kl_threshold
                if self.adaptive_threshold:
                    threshold = self.get_adaptive_threshold(alts1[:min_length])
                
                # Detect divergences
                for pos in range(min_length):
                    # Skip if we don't have enough alternatives
                    if not alts1[pos] or not alts2[pos]:
                        continue
                    
                    # Calculate the divergence
                    kl_div = self.calculate_kl_divergence(alts1[pos], alts2[pos])
                    js_dist = self.calculate_jensen_shannon_distance(alts1[pos], alts2[pos])
                    
                    # Check if this is a divergence point
                    if kl_div > threshold or js_dist > threshold/2:
                        # Check the top tokens
                        top1 = alts1[pos][0]["token_id"] if alts1[pos] else None
                        top2 = alts2[pos][0]["token_id"] if alts2[pos] else None
                        
                        # Only consider it a divergence if the top tokens are different
                        if top1 != top2:
                            # Calculate probability difference
                            prob1 = alts1[pos][0]["probability"] if alts1[pos] else 0
                            prob2 = alts2[pos][0]["probability"] if alts2[pos] else 0
                            prob_diff = abs(prob1 - prob2)
                            
                            if prob_diff >= self.min_prob_diff:
                                # Create a divergence point record
                                divergence_point = {
                                    "position": pos,
                                    "path_indices": (i, j),
                                    "kl_divergence": kl_div,
                                    "js_distance": js_dist,
                                    "probability_diff": prob_diff,
                                    "tokens": {
                                        "path1": {
                                            "token": alts1[pos][0]["token"],
                                            "token_id": alts1[pos][0]["token_id"],
                                            "probability": alts1[pos][0]["probability"]
                                        },
                                        "path2": {
                                            "token": alts2[pos][0]["token"],
                                            "token_id": alts2[pos][0]["token_id"],
                                            "probability": alts2[pos][0]["probability"]
                                        }
                                    },
                                    "context_window": self._extract_context_window(path1, path2, pos)
                                }
                                
                                divergence_points.append(divergence_point)
        
        # Sort divergence points by position
        divergence_points.sort(key=lambda x: x["position"])
        
        return divergence_points
    
    def _extract_context_window(
        self,
        path1: Dict[str, Any],
        path2: Dict[str, Any],
        position: int
    ) -> Dict[str, List[str]]:
        """
        Extract context window around a divergence point.
        
        Args:
            path1: First path
            path2: Second path
            position: Position of the divergence
            
        Returns:
            Dictionary with context windows for both paths
        """
        # Get the token IDs
        generated_ids1 = path1.get("generated_ids", None)
        generated_ids2 = path2.get("generated_ids", None)
        
        if generated_ids1 is None or generated_ids2 is None:
            return {"path1": [], "path2": []}
        
        # Ensure we have torch tensors
        if not isinstance(generated_ids1, torch.Tensor):
            generated_ids1 = torch.tensor(generated_ids1)
        if not isinstance(generated_ids2, torch.Tensor):
            generated_ids2 = torch.tensor(generated_ids2)
        
        # Calculate the actual positions in the sequence
        # We need to account for the prompt tokens
        input_length = path1.get("input_ids", torch.tensor([[]])).shape[1]
        seq_position = input_length + position
        
        # Define the window
        start = max(input_length, seq_position - self.context_window_size)
        end = min(seq_position + self.context_window_size + 1, generated_ids1.shape[1])
        
        # Extract the token IDs for the window
        window_ids1 = generated_ids1[0, start:end].tolist()
        window_ids2 = generated_ids2[0, start:end].tolist()
        
        # Decode the token IDs
        tokenizer = path1.get("tokenizer", None)
        if tokenizer:
            window_tokens1 = tokenizer.convert_ids_to_tokens(window_ids1)
            window_tokens2 = tokenizer.convert_ids_to_tokens(window_ids2)
        else:
            # We don't have a tokenizer, so we just return the IDs
            window_tokens1 = [str(token_id) for token_id in window_ids1]
            window_tokens2 = [str(token_id) for token_id in window_ids2]
        
        return {
            "path1": window_tokens1,
            "path2": window_tokens2
        }