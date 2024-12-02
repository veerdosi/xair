# visualization/attention_vis.py

import plotly.graph_objects as go
import plotly.express as px
import torch
import numpy as np
from typing import List, Dict, Optional
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage

class AttentionVisualizer:
    def __init__(self):
        self.color_scales = {
            'attention': 'Viridis',
            'importance': 'RdBu',
            'uncertainty': 'Plasma'
        }
    
    def create_hierarchical_attention_view(
        self, 
        attention_weights: List[torch.Tensor],
        tokens: List[str],
        layer_names: Optional[List[str]] = None
    ) -> go.Figure:
        """Create a hierarchical view of attention patterns across layers"""
        if layer_names is None:
            layer_names = [f"Layer {i}" for i in range(len(attention_weights))]
            
        # Process attention weights
        processed_weights = []
        for layer_weights in attention_weights:
            layer_mean = layer_weights.mean(dim=1).cpu().numpy()
            processed_weights.append(layer_mean)
            
        # Create linkage matrix
        layer_patterns = np.array(processed_weights)
        linkage_matrix = linkage(layer_patterns.reshape(len(layer_patterns), -1), 
                               method='ward')
        
        # Create dendrogram
        fig = go.Figure()
        
        # Add dendrogram
        dendro = dendrogram(linkage_matrix, labels=layer_names)
        
        # Add heatmaps
        for idx, (weights, name) in enumerate(zip(processed_weights, layer_names)):
            fig.add_trace(go.Heatmap(
                z=weights,
                x=tokens,
                y=tokens,
                colorscale=self.color_scales['attention'],
                showscale=True,
                visible=False,
                name=name
            ))
        
        # Update layout
        fig.update_layout(
            title="Hierarchical Attention Analysis",
            updatemenus=[{
                'buttons': [
                    {'label': name,
                     'method': 'update',
                     'args': [{'visible': [i == idx for i in range(len(processed_weights))]}]}
                    for idx, name in enumerate(layer_names)
                ],
                'direction': 'down',
                'showactive': True,
            }],
            width=1000,
            height=800
        )
        
        return fig
    
    def create_attention_flow(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        threshold: float = 0.1
    ) -> go.Figure:
        """Create a flow diagram of attention patterns"""
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, token in enumerate(tokens):
            G.add_node(i, label=token)
        
        # Add edges for attention above threshold
        attention_numpy = attention_weights.cpu().numpy()
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if attention_numpy[i, j] > threshold:
                    G.add_edge(i, j, weight=attention_numpy[i, j])
        
        # Get position layout
        pos = nx.spring_layout(G)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges (attention flows)
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=tokens,
            textposition="top center",
            marker=dict(
                size=20,
                color='#1f77b4',
            )
        ))
        
        fig.update_layout(
            title="Attention Flow Visualization",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            width=800,
            height=600
        )
        
        return fig
    
    def create_comparative_attention_view(
        self,
        base_attention: torch.Tensor,
        counterfactual_attention: torch.Tensor,
        tokens: List[str]
    ) -> go.Figure:
        """Create a comparative view of attention patterns"""
        fig = go.Figure()
        
        # Add base attention
        fig.add_trace(go.Heatmap(
            z=base_attention.cpu().numpy(),
            x=tokens,
            y=tokens,
            colorscale=self.color_scales['attention'],
            showscale=True,
            name='Base Attention'
        ))
        
        # Add counterfactual attention
        fig.add_trace(go.Heatmap(
            z=counterfactual_attention.cpu().numpy(),
            x=tokens,
            y=tokens,
            colorscale=self.color_scales['attention'],
            showscale=True,
            visible=False,
            name='Counterfactual Attention'
        ))
        
        # Add difference
        diff_attention = (counterfactual_attention - base_attention).cpu().numpy()
        fig.add_trace(go.Heatmap(
            z=diff_attention,
            x=tokens,
            y=tokens,
            colorscale=self.color_scales['importance'],
            showscale=True,
            visible=False,
            name='Attention Difference'
        ))
        
        # Update layout
        fig.update_layout(
            updatemenus=[{
                'buttons': [
                    {'label': 'Base Attention',
                     'method': 'update',
                     'args': [{'visible': [True, False, False]}]},
                    {'label': 'Counterfactual',
                     'method': 'update',
                     'args': [{'visible': [False, True, False]}]},
                    {'label': 'Difference',
                     'method': 'update',
                     'args': [{'visible': [False, False, True]}]}
                ],
                'direction': 'down',
                'showactive': True,
            }],
            title="Comparative Attention Analysis",
            width=1000,
            height=800
        )
        
        return fig

# Add imports and additional visualization features as needed
