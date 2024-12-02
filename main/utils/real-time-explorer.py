# utils/real_time_explorer.py

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import plotly.graph_objects as go
from torch.nn import functional as F

@dataclass
class ExplorationResult:
    confidence: float
    alternative_outputs: List[str]
    attention_flow: Dict[str, torch.Tensor]
    decision_boundary: Optional[torch.Tensor]
    feature_interactions: Dict[str, float]

class RealTimeModelExplorer:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.exploration_history = []
        
    def explore_decision_process(
        self,
        input_text: str,
        num_alternatives: int = 5,
        confidence_threshold: float = 0.1
    ) -> ExplorationResult:
        """Explore model's decision-making process in real-time"""
        
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
            
        # Analyze confidence and alternatives
        confidence = self._analyze_confidence(outputs.logits)
        alternatives = self._generate_alternatives(outputs.logits, num_alternatives)
        
        # Analyze attention flow
        attention_flow = self._analyze_attention_flow(outputs.attentions)
        
        # Analyze decision boundaries
        decision_boundary = self._analyze_decision_boundary(outputs.logits, confidence_threshold)
        
        # Analyze feature interactions
        interactions = self._analyze_feature_interactions(outputs.attentions)
        
        result = ExplorationResult(
            confidence=confidence,
            alternative_outputs=alternatives,
            attention_flow=attention_flow,
            decision_boundary=decision_boundary,
            feature_interactions=interactions
        )
        
        self.exploration_history.append(result)
        return result
    
    def _analyze_confidence(self, logits: torch.Tensor) -> float:
        """Analyze model's confidence in its predictions"""
        probs = F.softmax(logits, dim=-1)
        max_prob = torch.max(probs, dim=-1).values
        return max_prob.item()
    
    def _generate_alternatives(
        self,
        logits: torch.Tensor,
        num_alternatives: int
    ) -> List[str]:
        """Generate alternative outputs with their probabilities"""
        probs = F.softmax(logits, dim=-1)
        top_k = torch.topk(probs, k=num_alternatives, dim=-1)
        
        alternatives = []
        for idx, prob in zip(top_k.indices[0], top_k.values[0]):
            token = self.tokenizer.decode(idx)
            alternatives.append(f"{token} ({prob:.3f})")
            
        return alternatives
    
    def _analyze_attention_flow(
        self,
        attentions: Tuple[torch.Tensor, ...]
    ) -> Dict[str, torch.Tensor]:
        """Analyze the flow of attention through the model"""
        flow_analysis = {
            'layer_wise': [],
            'head_importance': [],
            'cross_layer': []
        }
        
        for layer_idx, layer_attention in enumerate(attentions):
            # Layer-wise attention flow
            layer_flow = layer_attention.mean(dim=1)  # Average across heads
            flow_analysis['layer_wise'].append(layer_flow)
            
            # Head importance
            head_importance = torch.var(layer_attention, dim=-1)  # Variance across sequence
            flow_analysis['head_importance'].append(head_importance)
            
            # Cross-layer attention (if not first layer)
            if layer_idx > 0:
                cross_attention = self._compute_cross_layer_attention(
                    attentions[layer_idx-1],
                    layer_attention
                )
                flow_analysis['cross_layer'].append(cross_attention)
        
        return flow_analysis
    
    def _analyze_decision_boundary(
        self,
        logits: torch.Tensor,
        threshold: float
    ) -> Optional[torch.Tensor]:
        """Analyze decision boundaries in the model's output space"""
        probs = F.softmax(logits, dim=-1)
        
        # Find regions where probability differences are small
        prob_diffs = torch.abs(probs.unsqueeze(-1) - probs.unsqueeze(-2))
        boundary_regions = (prob_diffs < threshold).float()
        
        if torch.any(boundary_regions):
            return boundary_regions
        return None
    
    def _analyze_feature_interactions(
        self,
        attentions: Tuple[torch.Tensor, ...]
    ) -> Dict[str, float]:
        """Analyze interactions between different features/tokens"""
        interactions = {}
        
        # Aggregate attention patterns
        mean_attention = torch.mean(torch.stack(attentions), dim=0)
        
        # Compute interaction strengths
        for i in range(mean_attention.size(-2)):
            for j in range(i+1, mean_attention.size(-1)):
                interaction_strength = torch.mean(mean_attention[..., i, j])
                interactions[f"{i}-{j}"] = interaction_strength.item()
        
        return interactions
    
    def _compute_cross_layer_attention(
        self,
        prev_layer: torch.Tensor,
        curr_layer: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention patterns between consecutive layers"""
        # Normalize attention weights
        prev_norm = F.normalize(prev_layer, p=2, dim=-1)
        curr_norm = F.normalize(curr_layer, p=2, dim=-1)
        
        # Compute cross-attention
        cross_attention = torch.matmul(prev_norm, curr_norm.transpose(-2, -1))
        
        return cross_attention
    
    def visualize_exploration(self, result: ExplorationResult) -> Dict[str, go.Figure]:
        """Create visualizations for exploration results"""
        visualizations = {}
        
        # Confidence visualization
        visualizations['confidence'] = self._create_confidence_viz(
            result.confidence,
            result.alternative_outputs
        )
        
        # Attention flow visualization
        visualizations['attention_flow'] = self._create_attention_flow_viz(
            result.attention_flow
        )
        
        # Decision boundary visualization
        if result.decision_boundary is not None:
            visualizations['decision_boundary'] = self._create_decision_boundary_viz(
                result.decision_boundary
            )
        
        # Feature interactions visualization
        visualizations['interactions'] = self._create_interactions_viz(
            result.feature_interactions
        )
        
        return visualizations
    
    def _create_confidence_viz(
        self,
        confidence: float,
        alternatives: List[str]
    ) -> go.Figure:
        """Create confidence visualization"""
        fig = go.Figure()
        
        # Add confidence gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = confidence * 100,
            title = {'text': "Model Confidence"},
            gauge = {
                'axis': {'range': [None, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        # Add alternatives list
        y_pos = np.linspace(0, 1, len(alternatives))
        fig.add_trace(go.Scatter(
            x=[1]*len(alternatives),
            y=y_pos,
            mode='text',
            text=alternatives,
            textposition="middle right"
        ))
        
        fig.update_layout(
            title="Model Confidence and Alternatives",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def _create_attention_flow_viz(
        self,
        attention_flow: Dict[str, List[torch.Tensor]]
    ) -> go.Figure:
        """Create attention flow visualization"""
        fig = go.Figure()
        
        # Add layer-wise attention patterns
        for i, layer_attention in enumerate(attention_flow['layer_wise']):
            fig.add_trace(go.Heatmap(
                z=layer_attention.cpu().numpy(),
                name=f'Layer {i}',
                colorscale='Viridis',
                showscale=(i == 0)
            ))
        
        fig.update_layout(
            title="Attention Flow Across Layers",
            updatemenus=[{
                'buttons': [
                    {'label': f'Layer {i}',
                     'method': 'update',
                     'args': [{'visible': [j == i for j in range(len(attention_flow['layer_wise']))]}]}
                    for i in range(len(attention_flow['layer_wise']))
                ],
                'direction': 'down',
                'showactive': True,
            }],
            height=600
        )
        
        return fig
    
    def _create_decision_boundary_viz(
        self,
        boundary: torch.Tensor
    ) -> go.Figure:
        """Create decision boundary visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=boundary.cpu().numpy(),
            colorscale='RdBu',
            showscale=True
        ))
        
        fig.update_layout(
            title="Decision Boundaries",
            height=500
        )
        
        return fig
    
    def _create_interactions_viz(
        self,
        interactions: Dict[str, float]
    ) -> go.Figure:
        """Create feature interactions visualization"""
        fig = go.Figure()
        
        # Sort interactions by strength
        sorted_interactions = sorted(
            interactions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Create network graph
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for interaction, strength in sorted_interactions:
            i, j = map(int, interaction.split('-'))
            edge_x.extend([i, j, None])
            edge_y.extend([0, 0, None])
            edge_weights.append(abs(strength))
        
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none'
        ))
        
        fig.update_layout(
            title="Feature Interactions",
            showlegend=False,
            height=400
        )
        
        return fig
