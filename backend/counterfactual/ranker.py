"""
Counterfactual Ranker module for the XAIR system.
Ranks counterfactuals by impact and provides advanced analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class RankingResult:
    """Stores results of counterfactual ranking."""
    ranked_counterfactuals: List[Any]
    impact_distribution: Dict[str, float]
    position_distribution: Dict[int, int]
    top_flipping_positions: List[int]
    timestamp: float = field(default_factory=time.time)

class CounterfactualRanker:
    """Ranks and analyzes counterfactuals."""
    
    def __init__(
        self,
        impact_weight: float = 1.0,
        flip_weight: float = 1.5,
        position_weight: float = 0.5,
        attention_weight: float = 0.8,
        verbose: bool = False
    ):
        """
        Initialize the counterfactual ranker.
        
        Args:
            impact_weight: Weight for impact score
            flip_weight: Weight for output flips
            position_weight: Weight for position (earlier is better)
            attention_weight: Weight for attention score
            verbose: Whether to log detailed information
        """
        self.impact_weight = impact_weight
        self.flip_weight = flip_weight
        self.position_weight = position_weight
        self.attention_weight = attention_weight
        self.verbose = verbose
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        
        # Last ranking result
        self.last_result = None
    
    def rank_counterfactuals(
        self,
        counterfactuals: List[Any],
        prefer_early_positions: bool = True
    ) -> RankingResult:
        """
        Rank counterfactuals by weighted criteria.
        
        Args:
            counterfactuals: List of CounterfactualCandidate objects
            prefer_early_positions: Whether to prefer earlier positions
            
        Returns:
            RankingResult object
        """
        if not counterfactuals:
            logger.warning("No counterfactuals to rank")
            return RankingResult(
                ranked_counterfactuals=[],
                impact_distribution={},
                position_distribution={},
                top_flipping_positions=[]
            )
        
        # Calculate max position for normalization
        max_position = max(cf.position for cf in counterfactuals) if counterfactuals else 1
        
        # Define ranking function
        def rank_score(cf):
            # Normalize position (0-1 range, earlier is higher)
            position_normalized = 1.0 - (cf.position / max_position) if prefer_early_positions else (cf.position / max_position)
            
            # Calculate weighted score
            score = (
                cf.impact_score * self.impact_weight +
                (1.0 if cf.flipped_output else 0.0) * self.flip_weight +
                position_normalized * self.position_weight +
                cf.attention_score * self.attention_weight
            )
            
            return score
        
        # Calculate scores for each counterfactual
        scored_counterfactuals = [(cf, rank_score(cf)) for cf in counterfactuals]
        
        # Sort by score
        scored_counterfactuals.sort(key=lambda x: x[1], reverse=True)
        
        # Extract ranked counterfactuals
        ranked_counterfactuals = [cf for cf, _ in scored_counterfactuals]
        
        # Calculate impact distribution (binned)
        impact_scores = [cf.impact_score for cf in counterfactuals]
        impact_bins = {
            "very_high": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "very_low": 0
        }
        
        for score in impact_scores:
            if score >= 0.8:
                impact_bins["very_high"] += 1
            elif score >= 0.6:
                impact_bins["high"] += 1
            elif score >= 0.4:
                impact_bins["medium"] += 1
            elif score >= 0.2:
                impact_bins["low"] += 1
            else:
                impact_bins["very_low"] += 1
        
        # Convert to percentages
        total_cfs = len(counterfactuals)
        impact_distribution = {
            key: (count / total_cfs) if total_cfs > 0 else 0.0 
            for key, count in impact_bins.items()
        }
        
        # Calculate position distribution
        position_distribution = {}
        for cf in counterfactuals:
            pos = cf.position
            if pos not in position_distribution:
                position_distribution[pos] = 0
            position_distribution[pos] += 1
        
        # Find top flipping positions
        flipping_cfs = [cf for cf in counterfactuals if cf.flipped_output]
        position_flip_count = {}
        for cf in flipping_cfs:
            pos = cf.position
            if pos not in position_flip_count:
                position_flip_count[pos] = 0
            position_flip_count[pos] += 1
        
        # Sort positions by flip count
        top_flipping_positions = sorted(position_flip_count.keys(), 
                                        key=lambda pos: position_flip_count[pos], 
                                        reverse=True)
        
        # Create ranking result
        result = RankingResult(
            ranked_counterfactuals=ranked_counterfactuals,
            impact_distribution=impact_distribution,
            position_distribution=position_distribution,
            top_flipping_positions=top_flipping_positions
        )
        
        self.last_result = result
        
        return result
    
    def get_critical_positions(
        self,
        counterfactuals: List[Any],
        min_flip_count: int = 1
    ) -> List[int]:
        """
        Identify critical positions that have a high impact on output.
        
        Args:
            counterfactuals: List of CounterfactualCandidate objects
            min_flip_count: Minimum number of flips for a position to be considered critical
            
        Returns:
            List of critical positions
        """
        if not counterfactuals:
            return []
        
        # Count flips per position
        position_flip_count = {}
        for cf in counterfactuals:
            if cf.flipped_output:
                pos = cf.position
                if pos not in position_flip_count:
                    position_flip_count[pos] = 0
                position_flip_count[pos] += 1
        
        # Filter positions by minimum flip count
        critical_positions = [
            pos for pos, count in position_flip_count.items()
            if count >= min_flip_count
        ]
        
        # Sort by position (earliest first)
        critical_positions.sort()
        
        return critical_positions
    
    def get_token_impact_ranking(
        self,
        counterfactuals: List[Any]
    ) -> Dict[str, float]:
        """
        Rank tokens by their impact when substituted.
        
        Args:
            counterfactuals: List of CounterfactualCandidate objects
            
        Returns:
            Dictionary mapping tokens to impact scores
        """
        if not counterfactuals:
            return {}
        
        # Group by original token
        token_impacts = {}
        token_counts = {}
        
        for cf in counterfactuals:
            token = cf.original_token
            if token not in token_impacts:
                token_impacts[token] = 0.0
                token_counts[token] = 0
            
            token_impacts[token] += cf.impact_score
            token_counts[token] += 1
        
        # Calculate average impact per token
        avg_impacts = {
            token: impacts / token_counts[token]
            for token, impacts in token_impacts.items()
        }
        
        # Sort by impact
        sorted_tokens = sorted(avg_impacts.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_tokens)
    
    def get_counterfactual_clusters(
        self,
        counterfactuals: List[Any],
        max_clusters: int = 3
    ) -> List[List[Any]]:
        """
        Cluster counterfactuals by position and impact.
        
        Args:
            counterfactuals: List of CounterfactualCandidate objects
            max_clusters: Maximum number of clusters to return
            
        Returns:
            List of clustered counterfactuals
        """
        if not counterfactuals or len(counterfactuals) <= max_clusters:
            return [counterfactuals] if counterfactuals else []
        
        try:
            from sklearn.cluster import KMeans
            
            # Extract features for clustering
            features = np.array([
                [cf.position, cf.impact_score, cf.attention_score, 1.0 if cf.flipped_output else 0.0]
                for cf in counterfactuals
            ])
            
            # Normalize features
            features_mean = np.mean(features, axis=0)
            features_std = np.std(features, axis=0)
            features_std[features_std == 0] = 1  # Avoid division by zero
            normalized_features = (features - features_mean) / features_std
            
            # Determine optimal number of clusters (up to max_clusters)
            num_clusters = min(max_clusters, len(counterfactuals) // 2)
            
            # Cluster counterfactuals
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(normalized_features)
            
            # Group counterfactuals by cluster
            clusters = [[] for _ in range(num_clusters)]
            for i, cf in enumerate(counterfactuals):
                cluster_idx = cluster_labels[i]
                clusters[cluster_idx].append(cf)
            
            # Sort clusters by average impact
            clusters.sort(key=lambda cluster: np.mean([cf.impact_score for cf in cluster]), reverse=True)
            
            return clusters
            
        except ImportError:
            logger.warning("scikit-learn not available for clustering, using position-based grouping instead")
            
            # Simple alternative: group by position range
            positions = [cf.position for cf in counterfactuals]
            min_pos, max_pos = min(positions), max(positions)
            
            if max_pos == min_pos:
                return [counterfactuals]
            
            # Create position ranges
            range_size = (max_pos - min_pos) / max_clusters
            clusters = [[] for _ in range(max_clusters)]
            
            for cf in counterfactuals:
                # Determine which range this position falls into
                range_idx = min(int((cf.position - min_pos) / range_size), max_clusters - 1)
                clusters[range_idx].append(cf)
            
            # Remove empty clusters
            clusters = [cluster for cluster in clusters if cluster]
            
            return clusters
    
    def generate_summary_report(
        self,
        counterfactuals: List[Any],
        tree_builder=None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of counterfactuals.
        
        Args:
            counterfactuals: List of CounterfactualCandidate objects
            tree_builder: CGRTBuilder instance (optional)
            
        Returns:
            Dictionary with summary report
        """
        if not counterfactuals:
            return {"status": "No counterfactuals available"}
        
        # Rank counterfactuals
        ranking_result = self.rank_counterfactuals(counterfactuals)
        
        # Calculate statistics
        flip_rate = sum(1 for cf in counterfactuals if cf.flipped_output) / len(counterfactuals)
        avg_impact = np.mean([cf.impact_score for cf in counterfactuals])
        critical_positions = self.get_critical_positions(counterfactuals)
        token_impacts = self.get_token_impact_ranking(counterfactuals)
        
        # Get top counterfactuals
        top_counterfactuals = ranking_result.ranked_counterfactuals[:5]
        
        # Prepare the report
        report = {
            "summary": {
                "total_counterfactuals": len(counterfactuals),
                "flip_rate": flip_rate,
                "average_impact": avg_impact,
                "critical_positions": critical_positions[:5]  # Top 5 critical positions
            },
            "impact_distribution": ranking_result.impact_distribution,
            "top_counterfactuals": [
                {
                    "position": cf.position,
                    "original_token": cf.original_token,
                    "alternative_token": cf.alternative_token,
                    "impact_score": cf.impact_score,
                    "flipped_output": cf.flipped_output
                }
                for cf in top_counterfactuals
            ],
            "token_impact_ranking": {
                k: v for k, v in list(token_impacts.items())[:10]  # Top 10 tokens
            }
        }
        
        # If tree_builder is provided, add node information
        if tree_builder:
            # Get node information for critical positions
            critical_nodes = []
            for pos in critical_positions[:5]:  # Top 5 positions
                # Find nodes at this position
                pos_nodes = [node for node_id, node in tree_builder.nodes.items() if node.position == pos]
                
                for node in pos_nodes:
                    critical_nodes.append({
                        "node_id": node.id,
                        "token": node.token,
                        "position": node.position,
                        "importance_score": node.importance_score,
                        "attention_score": node.attention_score,
                        "is_divergence_point": node.is_divergence_point
                    })
            
            report["critical_nodes"] = critical_nodes
        
        return report