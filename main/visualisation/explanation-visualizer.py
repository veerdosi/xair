# visualization/explanation_vis.py

import plotly.graph_objects as go
import plotly.express as px
import torch
import numpy as np
import shap
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import List, Dict, Tuple, Optional

class ExplanationVisualizer:
    def __init__(self):
        self.color_scales = {
            'prototype': 'Viridis',
            'shap': 'RdBu',
            'density': 'Plasma',
            'hierarchy': 'Spectral'
        }
    
    def create_prototype_visualization(
        self, 
        prototypes: torch.Tensor,
        query: torch.Tensor,
        similarities: torch.Tensor,
        feature_names: List[str]
    ) -> go.Figure:
        """Create visualization of prototype-based explanations"""
        fig = go.Figure()
        
        # Add prototype points
        for i, prototype in enumerate(prototypes):
            fig.add_trace(go.Scatter(
                x=prototype[:, 0].cpu(),
                y=prototype[:, 1].cpu(),
                mode='markers+text',
                name=f'Prototype {i}',
                text=[f"{name}: {val:.2f}" for name, val in zip(feature_names, prototype[0])],
                marker=dict(size=15, symbol='circle')
            ))
        
        # Add query point
        fig.add_trace(go.Scatter(
            x=[query[0, 0].cpu()],
            y=[query[0, 1].cpu()],
            mode='markers+text',
            name='Query',
            text='Query Point',
            marker=dict(size=20, symbol='star')
        ))
        
        # Add similarity connections
        for i, sim in enumerate(similarities):
            if sim > 0.5:  # Only show strong connections
                fig.add_trace(go.Scatter(
                    x=[query[0, 0].cpu(), prototypes[i][0, 0].cpu()],
                    y=[query[0, 1].cpu(), prototypes[i][0, 1].cpu()],
                    mode='lines',
                    name=f'Similarity: {sim:.2f}',
                    line=dict(width=sim.item() * 3)
                ))
        
        fig.update_layout(
            title="Prototype-based Explanation",
            showlegend=True,
            width=800,
            height=600
        )
        
        return fig

    def create_shap_summary(
        self,
        model: torch.nn.Module,
        background_data: torch.Tensor,
        explanation_data: torch.Tensor,
        feature_names: List[str]
    ) -> go.Figure:
        """Create SHAP summary plot"""
        # Create explainer
        explainer = shap.DeepExplainer(model, background_data)
        shap_values = explainer.shap_values(explanation_data)
        
        # Convert to numpy for visualization
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        fig = go.Figure()
        
        # Add SHAP values
        for i, feature in enumerate(feature_names):
            fig.add_trace(go.Box(
                y=shap_values[:, i],
                name=feature,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title="SHAP Feature Importance",
            yaxis_title="SHAP value",
            showlegend=False,
            width=1000,
            height=600
        )
        
        return fig

class DensityBasedFeatureSelector:
    def __init__(self, eps: float = 0.3, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        
    def select_features(
        self,
        features: torch.Tensor,
        feature_names: List[str]
    ) -> Tuple[List[str], np.ndarray]:
        """Select features based on density clustering"""
        # Compute feature importance densities
        densities = self._compute_densities(features)
        
        # Cluster features based on densities
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        ).fit(densities.reshape(-1, 1))
        
        # Select features from highest density clusters
        selected_indices = self._select_from_clusters(clustering.labels_, densities)
        selected_features = [feature_names[i] for i in selected_indices]
        
        return selected_features, densities[selected_indices]
    
    def _compute_densities(self, features: torch.Tensor) -> np.ndarray:
        """Compute density for each feature"""
        densities = []
        for i in range(features.size(1)):
            kernel = stats.gaussian_kde(features[:, i].cpu().numpy())
            densities.append(kernel.evaluate(features[:, i].cpu().numpy()).mean())
        return np.array(densities)
    
    def _select_from_clusters(
        self,
        labels: np.ndarray,
        densities: np.ndarray
    ) -> np.ndarray:
        """Select features from each cluster based on density"""
        selected = []
        for label in np.unique(labels):
            if label == -1:  # Skip noise
                continue
            cluster_indices = np.where(labels == label)[0]
            # Select highest density feature from cluster
            best_in_cluster = cluster_indices[np.argmax(densities[cluster_indices])]
            selected.append(best_in_cluster)
        return np.array(selected)

class HierarchicalPrototypeOrganizer:
    def __init__(self, n_levels: int = 3):
        self.n_levels = n_levels
        
    def organize_prototypes(
        self,
        prototypes: torch.Tensor,
        feature_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Organize prototypes into hierarchical structure"""
        # Compute linkage matrix
        linkage_matrix = self._compute_linkage(prototypes)
        
        # Create hierarchy levels
        hierarchy = {}
        for level in range(self.n_levels):
            n_clusters = self.n_levels - level
            hierarchy[f'level_{level}'] = self._cluster_prototypes(
                prototypes, linkage_matrix, n_clusters
            )
        
        return hierarchy
    
    def visualize_hierarchy(
        self,
        hierarchy: Dict[str, torch.Tensor],
        feature_names: List[str]
    ) -> go.Figure:
        """Visualize hierarchical organization of prototypes"""
        fig = go.Figure()
        
        # Create sankey diagram
        nodes = []
        links = []
        
        # Add nodes for each level
        node_labels = []
        for level, prototypes in hierarchy.items():
            for i, prototype in enumerate(prototypes):
                node_labels.append(f"{level}_cluster_{i}")
        
        # Add links between levels
        for i in range(len(hierarchy) - 1):
            current_level = f'level_{i}'
            next_level = f'level_{i+1}'
            
            current_prototypes = hierarchy[current_level]
            next_prototypes = hierarchy[next_level]
            
            # Compute similarities between levels
            similarities = self._compute_similarities(current_prototypes, next_prototypes)
            
            # Add strong connections as links
            for src in range(len(current_prototypes)):
                for dst in range(len(next_prototypes)):
                    if similarities[src, dst] > 0.5:
                        links.append(dict(
                            source=src + len(current_prototypes) * i,
                            target=dst + len(next_prototypes) * (i+1),
                            value=similarities[src, dst].item()
                        ))
        
        fig.add_trace(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color="blue"
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links]
            )
        ))
        
        fig.update_layout(
            title_text="Hierarchical Prototype Organization",
            font_size=10,
            width=1000,
            height=800
        )
        
        return fig
    
    def _compute_linkage(self, prototypes: torch.Tensor) -> np.ndarray:
        """Compute linkage matrix for hierarchical clustering"""
        prototype_matrix = prototypes.cpu().numpy()
        return linkage(prototype_matrix, method='ward')
    
    def _cluster_prototypes(
        self,
        prototypes: torch.Tensor,
        linkage_matrix: np.ndarray,
        n_clusters: int
    ) -> torch.Tensor:
        """Cluster prototypes into specified number of clusters"""
        from scipy.cluster.hierarchy import fcluster
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Compute cluster centers
        clustered_prototypes = []
        for i in range(1, n_clusters + 1):
            mask = labels == i
            cluster_center = prototypes[mask].mean(dim=0)
            clustered_prototypes.append(cluster_center)
        
        return torch.stack(clustered_prototypes)
    
    def _compute_similarities(
        self,
        prototypes1: torch.Tensor,
        prototypes2: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarities between two sets of prototypes"""
        return torch.nn.functional.cosine_similarity(
            prototypes1.unsqueeze(1),
            prototypes2.unsqueeze(0),
            dim=2
        )

# Usage Example:
"""
# Initialize visualizers
exp_vis = ExplanationVisualizer()
selector = DensityBasedFeatureSelector()
organizer = HierarchicalPrototypeOrganizer()

# Create visualizations
prototype_viz = exp_vis.create_prototype_visualization(prototypes, query, similarities, feature_names)
shap_viz = exp_vis.create_shap_summary(model, background_data, explanation_data, feature_names)

# Select important features
selected_features, densities = selector.select_features(features, feature_names)

# Organize prototypes hierarchically
hierarchy = organizer.organize_prototypes(prototypes, feature_names)
hierarchy_viz = organizer.visualize_hierarchy(hierarchy, feature_names)
"""
