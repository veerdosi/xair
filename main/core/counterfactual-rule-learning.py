import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.autograd import grad

@dataclass
class CounterfactualExample:
    original_input: torch.Tensor
    counterfactual_input: torch.Tensor
    original_output: torch.Tensor
    counterfactual_output: torch.Tensor
    changes: Dict[str, Tuple[float, float]]  # feature: (from_value, to_value)
    proximity_score: float
    sparsity_score: float
    validity_score: float

class EnhancedCounterfactualGenerator:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: List[str],
        feature_ranges: Dict[str, Tuple[float, float]],
        categorical_features: List[str] = None
    ):
        self.model = model
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges
        self.categorical_features = categorical_features or []
        
        # Hyperparameters for optimization
        self.lr = 0.01
        self.max_iterations = 100
        self.diversity_weight = 0.1
        self.sparsity_weight = 0.1
        
    def generate(
        self,
        input_data: torch.Tensor,
        desired_output: Optional[torch.Tensor] = None,
        n_examples: int = 3
    ) -> List[CounterfactualExample]:
        """Generate diverse counterfactual examples"""
        counterfactuals = []
        original_output = self.model(input_data)
        
        if desired_output is None:
            # If no desired output specified, flip the prediction
            desired_output = 1 - torch.argmax(original_output, dim=1)
        
        # Generate multiple diverse counterfactuals
        for i in range(n_examples):
            # Add diversity penalty based on existing counterfactuals
            diversity_penalty = self._compute_diversity_penalty(counterfactuals)
            
            # Generate counterfactual with diversity consideration
            counterfactual = self._optimize_counterfactual(
                input_data=input_data,
                desired_output=desired_output,
                diversity_penalty=diversity_penalty
            )
            
            if counterfactual is not None:
                counterfactuals.append(counterfactual)
        
        return counterfactuals
    
    def _optimize_counterfactual(
        self,
        input_data: torch.Tensor,
        desired_output: torch.Tensor,
        diversity_penalty: Optional[torch.Tensor] = None
    ) -> Optional[CounterfactualExample]:
        """Optimize for a single counterfactual example"""
        # Initialize counterfactual as copy of input
        counterfactual = input_data.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([counterfactual], lr=self.lr)
        
        best_loss = float('inf')
        best_counterfactual = None
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Compute model output for current counterfactual
            output = self.model(counterfactual)
            
            # Compute various loss components
            prediction_loss = F.cross_entropy(output, desired_output)
            proximity_loss = self._compute_proximity_loss(input_data, counterfactual)
            sparsity_loss = self._compute_sparsity_loss(input_data, counterfactual)
            
            # Combine losses
            total_loss = (
                prediction_loss +
                self.proximity_weight * proximity_loss +
                self.sparsity_weight * sparsity_loss
            )
            
            if diversity_penalty is not None:
                total_loss += self.diversity_weight * diversity_penalty
            
            # Update counterfactual
            total_loss.backward()
            optimizer.step()
            
            # Project back to valid feature ranges
            self._project_to_valid_ranges(counterfactual)
            
            # Track best result
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_counterfactual = counterfactual.clone().detach()
        
        if best_counterfactual is not None:
            return self._create_counterfactual_example(
                original_input=input_data,
                counterfactual_input=best_counterfactual
            )
        return None
    
    def _compute_proximity_loss(
        self,
        original: torch.Tensor,
        counterfactual: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance between original and counterfactual"""
        # Use weighted Manhattan distance for interpretability
        diff = torch.abs(original - counterfactual)
        # Scale by feature ranges
        ranges = torch.tensor([self.feature_ranges[f][1] - self.feature_ranges[f][0] 
                             for f in self.feature_names])
        return torch.mean(diff / ranges)
    
    def _compute_sparsity_loss(
        self,
        original: torch.Tensor,
        counterfactual: torch.Tensor
    ) -> torch.Tensor:
        """Compute sparsity of changes"""
        diff = torch.abs(original - counterfactual)
        return torch.mean(torch.log(1 + diff))
    
    def _compute_diversity_penalty(
        self,
        existing_counterfactuals: List[CounterfactualExample]
    ) -> Optional[torch.Tensor]:
        """Compute diversity penalty based on existing counterfactuals"""
        if not existing_counterfactuals:
            return None
        
        existing_tensor = torch.stack([cf.counterfactual_input 
                                     for cf in existing_counterfactuals])
        
        # Encourage diversity by penalizing similarity to existing counterfactuals
        return -torch.min(torch.pdist(existing_tensor))

class HierarchicalRuleLearner:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: List[str],
        max_depth: int = 5,
        min_samples_leaf: int = 10
    ):
        self.model = model
        self.feature_names = feature_names
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        
        # Initialize rule extractors for different levels
        self.decision_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )
        
        self.random_forest = RandomForestClassifier(
            n_estimators=10,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )
    
    def extract_rules(
        self,
        input_data: torch.Tensor,
        expertise_level: int
    ) -> Dict[str, List[Dict]]:
        """Extract hierarchical rules at appropriate complexity level"""
        # Get model predictions
        with torch.no_grad():
            predictions = self.model(input_data).argmax(dim=1)
        
        # Convert to numpy for sklearn
        X = input_data.cpu().numpy()
        y = predictions.cpu().numpy()
        
        rules = {
            'simple': self._extract_simple_rules(X, y),
            'intermediate': self._extract_intermediate_rules(X, y),
            'detailed': self._extract_detailed_rules(X, y)
        }
        
        # Filter based on expertise level
        return self._filter_rules_by_expertise(rules, expertise_level)
    
    def _extract_simple_rules(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[Dict]:
        """Extract simple decision rules"""
        # Train a shallow decision tree
        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)
        
        return self._convert_tree_to_rules(tree)
    
    def _extract_intermediate_rules(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[Dict]:
        """Extract intermediate complexity rules"""
        self.decision_tree.fit(X, y)
        return self._convert_tree_to_rules(self.decision_tree)
    
    def _extract_detailed_rules(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[Dict]:
        """Extract detailed rules with interactions"""
        self.random_forest.fit(X, y)
        
        rules = []
        for tree in self.random_forest.estimators_:
            rules.extend(self._convert_tree_to_rules(tree))
        
        return self._consolidate_rules(rules)
    
    def _convert_tree_to_rules(
        self,
        tree: DecisionTreeClassifier
    ) -> List[Dict]:
        """Convert decision tree to list of rules"""
        rules = []
        n_nodes = tree.tree_.node_count
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        value = tree.tree_.value
        
        def recurse(node, path):
            if node < 0:
                return
                
            # If leaf node
            if children_left[node] == children_right[node]:
                if len(path) > 0:
                    rules.append({
                        'conditions': path.copy(),
                        'prediction': np.argmax(value[node]),
                        'confidence': float(np.max(value[node]) / np.sum(value[node]))
                    })
                return
            
            # Add this decision node to the path
            feature_name = self.feature_names[feature[node]]
            
            # Recurse left
            path.append({
                'feature': feature_name,
                'operator': '<=',
                'threshold': float(threshold[node])
            })
            recurse(children_left[node], path)
            path.pop()
            
            # Recurse right
            path.append({
                'feature': feature_name,
                'operator': '>',
                'threshold': float(threshold[node])
            })
            recurse(children_right[node], path)
            path.pop()
        
        recurse(0, [])
        return rules
    
    def _consolidate_rules(
        self,
        rules: List[Dict]
    ) -> List[Dict]:
        """Consolidate and simplify redundant rules"""
        # Group rules by prediction
        grouped_rules = {}
        for rule in rules:
            pred = rule['prediction']
            if pred not in grouped_rules:
                grouped_rules[pred] = []
            grouped_rules[pred].append(rule)
        
        # Consolidate rules with similar conditions
        consolidated = []
        for pred, pred_rules in grouped_rules.items():
            consolidated.extend(self._merge_similar_rules(pred_rules))
        
        return consolidated
    
    def _merge_similar_rules(
        self,
        rules: List[Dict]
    ) -> List[Dict]:
        """Merge rules with similar conditions"""
        merged = []
        used = set()
        
        for i, rule1 in enumerate(rules):
            if i in used:
                continue
                
            similar_rules = [rule1]
            used.add(i)
            
            # Find similar rules
            for j, rule2 in enumerate(rules[i+1:], i+1):
                if j in used:
                    continue
                    
                if self._rules_are_similar(rule1, rule2):
                    similar_rules.append(rule2)
                    used.add(j)
            
            # Merge similar rules
            if len(similar_rules) > 1:
                merged.append(self._create_merged_rule(similar_rules))
            else:
                merged.append(rule1)
        
        return merged
    
    def _rules_are_similar(
        self,
        rule1: Dict,
        rule2: Dict
    ) -> bool:
        """Check if two rules are similar enough to merge"""
        # Compare conditions
        conditions1 = {(c['feature'], c['operator']) for c in rule1['conditions']}
        conditions2 = {(c['feature'], c['operator']) for c in rule2['conditions']}
        
        # Rules are similar if they share most conditions
        common = conditions1.intersection(conditions2)
        return len(common) >= min(len(conditions1), len(conditions2)) * 0.8
    
    def _create_merged_rule(
        self,
        rules: List[Dict]
    ) -> Dict:
        """Create a new rule by merging similar rules"""
        # Take the most common conditions
        all_conditions = []
        for rule in rules:
            all_conditions.extend(rule['conditions'])
        
        # Count condition occurrences
        condition_counts = {}
        for condition in all_conditions:
            key = (condition['feature'], condition['operator'])
            if key not in condition_counts:
                condition_counts[key] = {'count': 0, 'thresholds': []}
            condition_counts[key]['count'] += 1
            condition_counts[key]['thresholds'].append(condition['threshold'])
        
        # Keep conditions that appear in majority of rules
        threshold = len(rules) / 2
        merged_conditions = []
        for (feature, operator), data in condition_counts.items():
            if data['count'] > threshold:
                merged_conditions.append({
                    'feature': feature,
                    'operator': operator,
                    'threshold': float(np.median(data['thresholds']))
                })
        
        # Calculate average confidence
        avg_confidence = np.mean([rule['confidence'] for rule in rules])
        
        return {
            'conditions': merged_conditions,
            'prediction': rules[0]['prediction'],
            'confidence': float(avg_confidence)
        }
    
    def _filter_rules_by_expertise(
        self,
        rules: Dict[str, List[Dict]],
        expertise_level: int
    ) -> Dict[str, List[Dict]]:
        """Filter rules based on user expertise level"""
        if expertise_level == 0:
            return {'simple': rules['simple']}
        elif expertise_level == 1:
            return {
                'simple': rules['simple'],
                'intermediate': rules['intermediate']
            }
        else:
            return rules
