from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import numpy as np

class DashboardBackend:
    def __init__(
        self,
        model,
        counterfactual_generator,
        rule_learner,
        time_series_analyzer,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.counterfactual_generator = counterfactual_generator
        self.rule_learner = rule_learner
        self.time_series_analyzer = time_series_analyzer
        self.device = device
        
        self.app = FastAPI()
        self.setup_routes()
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def setup_routes(self):
        @self.app.post("/analyze")
        async def analyze_data(
            data: Dict[str, List[float]],
            expertise_level: int,
            explanation_type: str
        ):
            try:
                # Convert input data to tensor
                input_tensor = torch.tensor(data['values']).to(self.device)
                
                # Get model prediction
                with torch.no_grad():
                    output = self.model(input_tensor)
                
                # Generate explanation based on type and expertise level
                explanation = self.generate_explanation(
                    input_tensor,
                    output,
                    explanation_type,
                    expertise_level
                )
                
                return {
                    "prediction": output.cpu().numpy().tolist(),
                    "explanation": explanation
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/counterfactual")
        async def generate_counterfactual(
            data: Dict[str, List[float]],
            target_output: Optional[List[float]] = None
        ):
            try:
                input_tensor = torch.tensor(data['values']).to(self.device)
                if target_output:
                    target_tensor = torch.tensor(target_output).to(self.device)
                else:
                    target_tensor = None
                
                counterfactuals = self.counterfactual_generator.generate(
                    input_tensor,
                    target_tensor
                )
                
                return {
                    "counterfactuals": [cf.dict() for cf in counterfactuals]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/temporal")
        async def analyze_temporal(
            data: Dict[str, List[float]],
            window_size: int = 10
        ):
            try:
                sequence_tensor = torch.tensor(data['values']).to(self.device)
                
                analysis = self.time_series_analyzer.analyze_temporal_data(
                    sequence_tensor
                )
                
                return {
                    "patterns": [pattern.dict() for pattern in analysis.patterns],
                    "importance_scores": {
                        k: v.tolist() for k, v in analysis.importance_scores.items()
                    },
                    "seasonal_components": {
                        k: v.tolist() for k, v in analysis.seasonal_components.items()
                    },
                    "trend_components": {
                        k: v.tolist() for k, v in analysis.trend_components.items()
                    },
                    "anomalies": analysis.anomalies,
                    "feature_interactions": analysis.feature_interactions
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/rules")
        async def extract_rules(
            data: Dict[str, List[float]],
            expertise_level: int
        ):
            try:
                input_tensor = torch.tensor(data['values']).to(self.device)
                
                rules = self.rule_learner.extract_rules(
                    input_tensor,
                    expertise_level
                )
                
                return {
                    "rules": rules
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def generate_explanation(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor,
        explanation_type: str,
        expertise_level: int
    ) -> Dict:
        """Generate comprehensive explanation based on type and expertise level"""
        explanation = {}
        
        if explanation_type == "basic":
            explanation["basic"] = self._generate_basic_explanation(
                input_tensor,
                output
            )
            
        elif explanation_type == "counterfactual":
            explanation["counterfactual"] = self._generate_counterfactual_explanation(
                input_tensor,
                output
            )
            
        elif explanation_type == "temporal":
            explanation["temporal"] = self._generate_temporal_explanation(
                input_tensor
            )
            
        elif explanation_type == "rule":
            explanation["rules"] = self._generate_rule_explanation(
                input_tensor,
                expertise_level
            )
        
        # Add expertise-level specific components
        if expertise_level >= 1:
            explanation["intermediate"] = self._generate_intermediate_explanation(
                input_tensor,
                output
            )
        
        if expertise_level >= 2:
            explanation["expert"] = self._generate_expert_explanation(
                input_tensor,
                output
            )
        
        return explanation
    
    def _generate_basic_explanation(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor
    ) -> Dict:
        """Generate basic explanation"""
        # Implementation details...
        return {}
    
    def _generate_counterfactual_explanation(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor
    ) -> Dict:
        """Generate counterfactual explanation"""
        # Implementation details...
        return {}
    
    def _generate_temporal_explanation(
        self,
        input_tensor: torch.Tensor
    ) -> Dict:
        """Generate temporal explanation"""
        # Implementation details...
        return {}
    
    def _generate_rule_explanation(
        self,
        input_tensor: torch.Tensor,
        expertise_level: int
    ) -> Dict:
        """Generate rule-based explanation"""
        # Implementation details...
        return {}
    
    def _generate_intermediate_explanation(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor
    ) -> Dict:
        """Generate intermediate level explanation"""
        # Implementation details...
        return {}
    
    def _generate_expert_explanation(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor
    ) -> Dict:
        """Generate expert level explanation with detailed analysis"""
        explanation = {}
        
        # Feature importance and interactions
        feature_importance = self._calculate_feature_importance(input_tensor, output)
        feature_interactions = self._analyze_feature_interactions(input_tensor)
        
        # Uncertainty analysis
        uncertainty = self._analyze_uncertainty(input_tensor, output)
        
        # Model behavior analysis
        model_behavior = self._analyze_model_behavior(input_tensor)
        
        # Decision boundary analysis
        decision_boundary = self._analyze_decision_boundary(input_tensor, output)
        
        explanation.update({
            'feature_importance': feature_importance,
            'feature_interactions': feature_interactions,
            'uncertainty_analysis': uncertainty,
            'model_behavior': model_behavior,
            'decision_boundary': decision_boundary
        })
        
        return explanation
    
    def _calculate_feature_importance(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate detailed feature importance scores"""
        importance_scores = {}
        
        # Gradient-based importance
        gradients = torch.autograd.grad(
            output.sum(),
            input_tensor,
            create_graph=True
        )[0]
        
        # Average gradient magnitude for each feature
        grad_importance = torch.abs(gradients).mean(dim=0)
        
        # SHAP-based importance
        background = torch.zeros_like(input_tensor)  # Can be replaced with actual background data
        explainer = shap.DeepExplainer(self.model, background)
        shap_values = explainer.shap_values(input_tensor)
        
        # Combine different importance metrics
        for idx, feature_name in enumerate(self.feature_names):
            importance_scores[feature_name] = {
                'gradient_importance': float(grad_importance[idx]),
                'shap_importance': float(np.abs(shap_values[idx]).mean()),
                'combined_score': float((grad_importance[idx] + np.abs(shap_values[idx]).mean()) / 2)
            }
        
        return importance_scores
    
    def _analyze_feature_interactions(
        self,
        input_tensor: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Analyze interactions between features"""
        interactions = {}
        
        # Calculate pairwise interactions using H-statistic
        for i, feature1 in enumerate(self.feature_names):
            interactions[feature1] = {}
            for j, feature2 in enumerate(self.feature_names):
                if i != j:
                    interaction_score = self._calculate_h_statistic(
                        input_tensor,
                        i,
                        j
                    )
                    interactions[feature1][feature2] = float(interaction_score)
        
        return interactions
    
    def _analyze_uncertainty(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze model uncertainty"""
        uncertainty = {}
        
        # Model confidence
        probs = F.softmax(output, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        uncertainty['entropy'] = float(entropy.mean())
        
        # MC Dropout uncertainty (if model supports it)
        if hasattr(self.model, 'enable_dropout'):
            mc_samples = self._get_mc_dropout_samples(input_tensor)
            uncertainty['epistemic'] = float(torch.std(mc_samples, dim=0).mean())
        
        # Distance-based uncertainty
        distance_to_training = self._calculate_distance_to_training(input_tensor)
        uncertainty['data_uncertainty'] = float(distance_to_training)
        
        return uncertainty
    
    def _analyze_model_behavior(
        self,
        input_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze local model behavior"""
        behavior = {}
        
        # Local linearity analysis
        linearity = self._check_local_linearity(input_tensor)
        behavior['local_linearity'] = float(linearity)
        
        # Decision boundary distance
        boundary_distance = self._calculate_boundary_distance(input_tensor)
        behavior['boundary_distance'] = float(boundary_distance)
        
        # Local feature sensitivity
        sensitivity = self._calculate_local_sensitivity(input_tensor)
        behavior['feature_sensitivity'] = {
            feature: float(score) 
            for feature, score in sensitivity.items()
        }
        
        return behavior
    
    def _analyze_decision_boundary(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze decision boundary characteristics"""
        boundary_analysis = {}
        
        # Find nearest decision boundary points
        boundary_points = self._find_boundary_points(input_tensor)
        boundary_analysis['nearest_boundary_points'] = boundary_points
        
        # Calculate boundary curvature
        curvature = self._calculate_boundary_curvature(input_tensor)
        boundary_analysis['boundary_curvature'] = float(curvature)
        
        # Analyze boundary stability
        stability = self._analyze_boundary_stability(input_tensor)
        boundary_analysis['boundary_stability'] = float(stability)
        
        return boundary_analysis
    
    def _get_mc_dropout_samples(
        self,
        input_tensor: torch.Tensor,
        n_samples: int = 30
    ) -> torch.Tensor:
        """Get Monte Carlo Dropout samples"""
        self.model.enable_dropout()
        samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(input_tensor)
                samples.append(output)
        
        self.model.disable_dropout()
        return torch.stack(samples)
    
    def _calculate_distance_to_training(
        self,
        input_tensor: torch.Tensor
    ) -> float:
        """Calculate distance to training data"""
        # This would normally use a pre-computed training data representation
        # For now, return a placeholder value
        return 0.5
    
    def _check_local_linearity(
        self,
        input_tensor: torch.Tensor,
        epsilon: float = 1e-5
    ) -> float:
        """Check local linearity of model"""
        # Generate small perturbations
        perturbed = input_tensor + torch.randn_like(input_tensor) * epsilon
        
        # Compare gradients
        grad1 = torch.autograd.grad(
            self.model(input_tensor).sum(),
            input_tensor,
            create_graph=True
        )[0]
        
        grad2 = torch.autograd.grad(
            self.model(perturbed).sum(),
            perturbed,
            create_graph=True
        )[0]
        
        # Calculate gradient similarity
        return float(F.cosine_similarity(grad1.view(-1), grad2.view(-1), dim=0))
    
    def _calculate_h_statistic(
        self,
        input_tensor: torch.Tensor,
        feature_idx1: int,
        feature_idx2: int
    ) -> float:
        """Calculate H-statistic for feature interactions"""
        # This is a simplified version of the H-statistic calculation
        # Full implementation would involve multiple model evaluations
        return 0.5  # Placeholder
    
    def _calculate_boundary_distance(
        self,
        input_tensor: torch.Tensor
    ) -> float:
        """Calculate distance to decision boundary"""
        # This would normally use optimization to find nearest boundary point
        # Return placeholder for now
        return 1.0
    
    def _calculate_local_sensitivity(
        self,
        input_tensor: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate local feature sensitivity"""
        sensitivity = {}
        
        # Calculate gradient of output with respect to each feature
        gradients = torch.autograd.grad(
            self.model(input_tensor).sum(),
            input_tensor,
            create_graph=True
        )[0]
        
        for idx, feature_name in enumerate(self.feature_names):
            sensitivity[feature_name] = float(torch.abs(gradients[idx]).mean())
        
        return sensitivity
    
    def _find_boundary_points(
        self,
        input_tensor: torch.Tensor
    ) -> List[Dict[str, float]]:
        """Find nearest decision boundary points"""
        # This would normally involve optimization to find boundary points
        # Return placeholder for now
        return [{'distance': 1.0, 'direction': [0.0] * len(self.feature_names)}]
    
    def _calculate_boundary_curvature(
        self,
        input_tensor: torch.Tensor
    ) -> float:
        """Calculate decision boundary curvature"""
        # This would normally involve analyzing the Hessian
        # Return placeholder for now
        return 0.0
    
    def _analyze_boundary_stability(
        self,
        input_tensor: torch.Tensor
    ) -> float:
        """Analyze decision boundary stability"""
        # This would normally involve perturbation analysis
        # Return placeholder for now
        return 0.8

# Usage:
"""
backend = DashboardBackend(
    model=model,
    counterfactual_generator=counterfactual_generator,
    rule_learner=rule_learner,
    time_series_analyzer=time_series_analyzer
)

# Run the FastAPI application
import uvicorn
uvicorn.run(backend.app, host="0.0.0.0", port=8000)
"""