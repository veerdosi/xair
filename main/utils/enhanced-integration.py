# enhanced_integration.py

import torch
import numpy as np
import shap
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
import pandas as pd

@dataclass
class ExplanationMetrics:
    shap_values: np.ndarray
    feature_importance: Dict[str, float]
    counterfactual_examples: List[Dict]
    uncertainty: float
    rule_confidence: float

class EnhancedIntegrationSystem:
    def __init__(
        self,
        model,
        tokenizer,
        background_data: torch.Tensor,
        feature_names: List[str],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.background_data = background_data
        self.feature_names = feature_names
        
        # Initialize SHAP explainer
        self.explainer = shap.DeepExplainer(
            model=lambda x: model.forward(x)[0],
            data=background_data
        )
        
        # Initialize density-based feature selector
        self.feature_selector = DensityBasedFeatureSelector()
        
        # Initialize rule extractor
        self.rule_extractor = HierarchicalRuleExtractor(
            model=model,
            feature_names=feature_names
        )
        
        # Initialize counterfactual generator
        self.counterfactual_generator = CounterfactualGenerator(
            model=model,
            feature_names=feature_names
        )
    
    def generate_comprehensive_explanation(
        self,
        input_data: torch.Tensor,
        user_expertise: int
    ) -> Dict:
        """Generate comprehensive explanation based on user expertise"""
        # Get model prediction and base explanation
        output, base_explanation = self.model(input_data)
        
        # Generate SHAP values
        shap_values = self.explainer.shap_values(input_data)
        
        # Select important features based on density
        important_features = self.feature_selector.select_features(
            input_data,
            self.feature_names
        )
        
        # Generate counterfactuals
        counterfactuals = self.counterfactual_generator.generate(
            input_data,
            output,
            n_examples=3
        )
        
        # Extract rules at appropriate level
        rules = self.rule_extractor.extract_rules(
            input_data,
            expertise_level=user_expertise
        )
        
        # Combine explanations based on expertise level
        return self._combine_explanations(
            base_explanation=base_explanation,
            shap_values=shap_values,
            important_features=important_features,
            counterfactuals=counterfactuals,
            rules=rules,
            expertise_level=user_expertise
        )
    
    def _combine_explanations(
        self,
        base_explanation: torch.Tensor,
        shap_values: np.ndarray,
        important_features: List[str],
        counterfactuals: List[Dict],
        rules: Dict,
        expertise_level: int
    ) -> Dict:
        """Combine different explanation types based on user expertise"""
        explanation = {
            'base_explanation': base_explanation,
            'important_features': important_features
        }
        
        if expertise_level >= 1:
            # Add intermediate level explanations
            explanation.update({
                'shap_values': shap_values,
                'counterfactuals': counterfactuals[:1],  # Limit to one example
                'simple_rules': rules['simple']
            })
        
        if expertise_level >= 2:
            # Add expert level explanations
            explanation.update({
                'detailed_shap': self._generate_detailed_shap(shap_values),
                'all_counterfactuals': counterfactuals,
                'hierarchical_rules': rules['hierarchical'],
                'feature_interactions': self._analyze_feature_interactions(shap_values)
            })
        
        return explanation

class DensityBasedFeatureSelector:
    def __init__(self, eps: float = 0.3, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
    
    def select_features(
        self,
        features: torch.Tensor,
        feature_names: List[str]
    ) -> List[str]:
        """Select features based on density estimation"""
        densities = self._compute_densities(features)
        
        # Cluster features based on density
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        ).fit(densities.reshape(-1, 1))
        
        # Select representatives from each cluster
        selected_indices = self._select_from_clusters(
            clustering.labels_,
            densities
        )
        
        return [feature_names[i] for i in selected_indices]
    
    def _compute_densities(self, features: torch.Tensor) -> np.ndarray:
        """Compute density for each feature"""
        densities = []
        for i in range(features.size(1)):
            kernel = gaussian_kde(features[:, i].cpu().numpy())
            densities.append(kernel.evaluate(features[:, i].cpu().numpy()).mean())
        return np.array(densities)
    
    def _select_from_clusters(
        self,
        labels: np.ndarray,
        densities: np.ndarray
    ) -> np.ndarray:
        """Select representative features from each cluster"""
        selected = []
        for label in np.unique(labels):
            if label == -1:  # Skip noise
                continue
            cluster_indices = np.where(labels == label)[0]
            # Select highest density feature from cluster
            best_in_cluster = cluster_indices[np.argmax(densities[cluster_indices])]
            selected.append(best_in_cluster)
        return np.array(selected)

class HierarchicalRuleExtractor:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
    
    def extract_rules(
        self,
        input_data: torch.Tensor,
        expertise_level: int
    ) -> Dict:
        """Extract hierarchical rules at appropriate complexity"""
        # Extract base rules
        base_rules = self._extract_base_rules(input_data)
        
        rules = {
            'simple': self._simplify_rules(base_rules),
            'hierarchical': self._create_rule_hierarchy(base_rules)
        }
        
        return rules

class CounterfactualGenerator:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
    
    def generate(
        self,
        input_data: torch.Tensor,
        original_output: torch.Tensor,
        n_examples: int = 3
    ) -> List[Dict]:
        """Generate counterfactual examples"""
        counterfactuals = []
        
        # Implementation of counterfactual generation logic
        # This would include:
        # 1. Gradient-based optimization to find minimal changes
        # 2. Ensuring realistic counterfactuals
        # 3. Diversity in generated examples
        
        return counterfactuals
