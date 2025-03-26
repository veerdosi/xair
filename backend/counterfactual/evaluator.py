"""
Counterfactual Evaluator module for the XAIR system.
Analyzes and ranks counterfactuals based on impact.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
import logging
import time
import json
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Stores evaluation metrics for counterfactuals."""
    cfr: float = 0.0  # Counterfactual Flip Rate
    avg_impact: float = 0.0  # Average impact score
    max_impact: float = 0.0  # Maximum impact score
    semantic_coherence: float = 0.0  # Semantic coherence of counterfactuals
    min_token_changes: int = 0  # Minimum number of token changes to flip
    timestamp: float = field(default_factory=time.time)

class CounterfactualEvaluator:
    """Evaluates counterfactual alternatives."""
    
    def __init__(
        self,
        use_semantic_analysis: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the counterfactual evaluator.
        
        Args:
            use_semantic_analysis: Whether to use semantic analysis for evaluation
            verbose: Whether to log detailed information
        """
        self.use_semantic_analysis = use_semantic_analysis
        self.verbose = verbose
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        
        # Try to import semantic analysis libraries
        if self.use_semantic_analysis:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for semantic analysis")
            except (ImportError, OSError):
                logger.warning("Could not load spaCy model. Installing with: python -m spacy download en_core_web_sm")
                self.use_semantic_analysis = False
        
        # Evaluation metrics
        self.metrics = EvaluationMetrics()
    
    def evaluate_counterfactuals(
        self,
        counterfactuals: List[Any],
        original_output: str
    ) -> EvaluationMetrics:
        """
        Evaluate a list of counterfactuals.
        
        Args:
            counterfactuals: List of CounterfactualCandidate objects
            original_output: Original model output
            
        Returns:
            EvaluationMetrics object
        """
        if not counterfactuals:
            logger.warning("No counterfactuals to evaluate")
            return self.metrics
        
        # Calculate Counterfactual Flip Rate (CFR)
        flipped_count = sum(1 for cf in counterfactuals if cf.flipped_output)
        self.metrics.cfr = flipped_count / len(counterfactuals) if counterfactuals else 0.0
        
        # Calculate impact statistics
        impact_scores = [cf.impact_score for cf in counterfactuals]
        self.metrics.avg_impact = np.mean(impact_scores) if impact_scores else 0.0
        self.metrics.max_impact = max(impact_scores) if impact_scores else 0.0
        
        # Find minimum token changes needed to flip output
        if flipped_count > 0:
            # Sort by position to find earliest token change that flips
            flipped_cfs = [cf for cf in counterfactuals if cf.flipped_output]
            flipped_cfs.sort(key=lambda cf: cf.position)
            
            # The earliest position that flips
            self.metrics.min_token_changes = 1  # Minimum is 1 token
        else:
            self.metrics.min_token_changes = 0  # No flips found
        
        # Analyze semantic coherence if enabled
        if self.use_semantic_analysis:
            self.metrics.semantic_coherence = self._analyze_semantic_coherence(
                counterfactuals, 
                original_output
            )
        
        self.metrics.timestamp = time.time()
        
        logger.info(f"Evaluated {len(counterfactuals)} counterfactuals:")
        logger.info(f"  CFR: {self.metrics.cfr:.2f}")
        logger.info(f"  Avg Impact: {self.metrics.avg_impact:.2f}")
        logger.info(f"  Max Impact: {self.metrics.max_impact:.2f}")
        logger.info(f"  Min Token Changes: {self.metrics.min_token_changes}")
        logger.info(f"  Semantic Coherence: {self.metrics.semantic_coherence:.2f}")
        
        return self.metrics
    
    def _analyze_semantic_coherence(
        self,
        counterfactuals: List[Any],
        original_output: str
    ) -> float:
        """
        Analyze the semantic coherence of counterfactuals.
        
        Args:
            counterfactuals: List of CounterfactualCandidate objects
            original_output: Original model output
            
        Returns:
            Semantic coherence score (0-1)
        """
        try:
            if not self.nlp:
                return 0.5  # Default value if NLP model not available
                
            # Analyze a subset of counterfactuals for efficiency
            sample_size = min(10, len(counterfactuals))
            sample = sorted(counterfactuals, key=lambda cf: cf.impact_score, reverse=True)[:sample_size]
            
            coherence_scores = []
            
            # Get the original document
            original_doc = self.nlp(original_output)
            
            for cf in sample:
                if not cf.modified_path:
                    continue
                    
                # Get the modified document
                modified_doc = self.nlp(cf.modified_path)
                
                # Compare the document similarity
                similarity = original_doc.similarity(modified_doc)
                
                # For counterfactuals, we want them to be similar enough to be coherent
                # but different enough to have impact
                # Optimal range: 0.7-0.9 similarity for good counterfactuals
                
                # Convert to coherence score (1.0 = perfect coherence)
                if similarity < 0.4:
                    # Too different, likely incoherent
                    coherence = similarity
                elif similarity > 0.95:
                    # Too similar, not a meaningful counterfactual
                    coherence = 2.0 - similarity  # Maps 1.0 to 1.0, but penalizes values close to 1.0
                else:
                    # Good range for counterfactuals
                    # Rescale 0.4-0.95 to 0.7-1.0
                    coherence = 0.7 + (similarity - 0.4) * (0.3 / 0.55)
                
                coherence_scores.append(coherence)
            
            # Return average coherence
            return np.mean(coherence_scores) if coherence_scores else 0.5
        
        except Exception as e:
            logger.error(f"Error in semantic coherence analysis: {e}")
            return 0.5  # Default value on error
    
    def rank_counterfactuals(
        self,
        counterfactuals: List[Any],
        prioritize_flips: bool = True
    ) -> List[Any]:
        """
        Rank counterfactuals by their impact and importance.
        
        Args:
            counterfactuals: List of CounterfactualCandidate objects
            prioritize_flips: Whether to prioritize counterfactuals that flip the output
            
        Returns:
            Sorted list of counterfactuals
        """
        if not counterfactuals:
            return []
        
        # Define ranking function
        def rank_key(cf):
            # Base score is impact score
            score = cf.impact_score
            
            # Bonus for flipping the output
            if prioritize_flips and cf.flipped_output:
                score += 0.5  # Significant bonus for flips
            
            # Bonus for high attention tokens
            score += cf.attention_score * 0.2
            
            # Smaller bonus for higher probability alternatives
            score += cf.alternative_probability * 0.1
            
            return score
        
        # Sort by rank key
        ranked = sorted(counterfactuals, key=rank_key, reverse=True)
        
        return ranked
    
    def save_evaluation(self, output_path: str):
        """
        Save the evaluation metrics to a file.
        
        Args:
            output_path: Path to save the evaluation
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert metrics to dictionary
        metrics_dict = {
            "cfr": self.metrics.cfr,
            "avg_impact": self.metrics.avg_impact,
            "max_impact": self.metrics.max_impact,
            "semantic_coherence": self.metrics.semantic_coherence,
            "min_token_changes": self.metrics.min_token_changes,
            "timestamp": self.metrics.timestamp
        }
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Saved evaluation metrics to {output_path}")
    
    def load_evaluation(self, input_path: str):
        """
        Load evaluation metrics from a file.
        
        Args:
            input_path: Path to load the evaluation from
        """
        # Check if the file exists
        if not os.path.exists(input_path):
            logger.error(f"Evaluation file {input_path} not found")
            return
        
        # Load the data
        with open(input_path, "r") as f:
            metrics_dict = json.load(f)
        
        # Update metrics
        self.metrics.cfr = metrics_dict.get("cfr", 0.0)
        self.metrics.avg_impact = metrics_dict.get("avg_impact", 0.0)
        self.metrics.max_impact = metrics_dict.get("max_impact", 0.0)
        self.metrics.semantic_coherence = metrics_dict.get("semantic_coherence", 0.0)
        self.metrics.min_token_changes = metrics_dict.get("min_token_changes", 0)
        self.metrics.timestamp = metrics_dict.get("timestamp", time.time())
        
        logger.info(f"Loaded evaluation metrics from {input_path}")
    
    def get_top_counterfactuals(
        self,
        counterfactuals: List[Any],
        n: int = 5,
        require_flip: bool = False
    ) -> List[Any]:
        """
        Get the top N counterfactuals.
        
        Args:
            counterfactuals: List of CounterfactualCandidate objects
            n: Number of counterfactuals to return
            require_flip: Whether to only include counterfactuals that flip the output
            
        Returns:
            List of top counterfactuals
        """
        # Rank the counterfactuals
        ranked = self.rank_counterfactuals(counterfactuals, prioritize_flips=True)
        
        # Filter if required
        if require_flip:
            ranked = [cf for cf in ranked if cf.flipped_output]
        
        # Return top N
        return ranked[:n]