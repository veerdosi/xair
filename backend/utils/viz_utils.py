"""
Visualization utilities for the XAIR system.
"""

import os
import json
import logging
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_visualization_style(style: str = "whitegrid", 
                              context: str = "paper",
                              font_scale: float = 1.2,
                              palette: str = "viridis") -> None:
    """
    Set up the visualization style for all plots.
    
    Args:
        style: Seaborn style
        context: Seaborn context
        font_scale: Scale factor for the font
        palette: Color palette
    """
    # Set seaborn style
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    sns.set_palette(palette)
    
    # Configure matplotlib
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.autolayout'] = True
    
    logger.info(f"Visualization style set: {style}, context: {context}, palette: {palette}")

def plot_reasoning_tree(graph: nx.DiGraph, 
                        output_path: Optional[str] = None,
                        title: str = "Reasoning Tree",
                        node_size_factor: float = 100.0,
                        highlight_nodes: Optional[List[str]] = None,
                        show_node_labels: bool = True,
                        show_edge_labels: bool = False,
                        figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot a reasoning tree.
    
    Args:
        graph: NetworkX directed graph
        output_path: Path to save figure
        title: Plot title
        node_size_factor: Factor for node size
        highlight_nodes: List of node IDs to highlight
        show_node_labels: Whether to show node labels
        show_edge_labels: Whether to show edge labels
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    plt.figure(figsize=figsize)
    plt.title(title)
    
    # Use a hierarchical layout for tree-like structures
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot") if hasattr(nx, "nx_agraph") else nx.spring_layout(graph)
    
    # Prepare node attributes for visualization
    node_colors = []
    node_sizes = []
    edge_colors = []
    edge_widths = []
    
    # Process nodes
    for node_id in graph.nodes():
        node_data = graph.nodes[node_id]
        
        # Set node color based on importance score
        importance = node_data.get("importance_score", 0.0)
        node_colors.append(plt.cm.viridis(importance))
        
        # Set node size based on a combination of importance and attention
        attention = node_data.get("attention_score", 0.0)
        # Combined score with emphasis on importance
        combined_score = 0.7 * importance + 0.3 * attention
        node_sizes.append(node_size_factor * (0.5 + combined_score))
    
    # Process edges
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 0.5)
        edge_colors.append(plt.cm.Blues(weight))
        edge_widths.append(1.0 + 3.0 * weight)
    
    # Draw the graph
    nx.draw_networkx_nodes(
        graph, 
        pos, 
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Highlight specific nodes if requested
    if highlight_nodes:
        highlight_node_indices = [list(graph.nodes()).index(node) for node in highlight_nodes if node in graph.nodes()]
        if highlight_node_indices:
            highlighted_nodes = [list(graph.nodes())[i] for i in highlight_node_indices]
            highlighted_sizes = [node_sizes[i] * 1.2 for i in highlight_node_indices]
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=highlighted_nodes,
                node_size=highlighted_sizes,
                node_color='red',
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )
    
    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.7,
        arrowsize=15,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1'
    )
    
    # Draw node labels if requested
    if show_node_labels:
        # Create custom labels
        labels = {}
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            token = node_data.get("token", "")
            labels[node_id] = token if len(token) < 15 else token[:12] + "..."
        
        nx.draw_networkx_labels(
            graph,
            pos,
            labels=labels,
            font_size=9,
            font_family='sans-serif',
            font_weight='normal'
        )
    
    # Draw edge labels if requested
    if show_edge_labels:
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("edge_type", "")
            if edge_type:
                edge_labels[(u, v)] = edge_type
        
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            font_family='sans-serif'
        )
    
    # Add legend
    plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), 
                ax=plt.gca(), 
                label="Node Importance Score")
    
    # Remove axes
    plt.axis('off')
    
    # Save figure if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved reasoning tree visualization to {output_path}")
    
    return plt.gcf()

def plot_attention_matrix(attention_matrix: np.ndarray,
                          token_labels: List[str],
                          output_path: Optional[str] = None,
                          title: str = "Attention Matrix",
                          figsize: Tuple[int, int] = (12, 10),
                          cmap: str = "viridis") -> plt.Figure:
    """
    Plot an attention matrix as a heatmap.
    
    Args:
        attention_matrix: 2D numpy array of attention weights
        token_labels: Labels for tokens
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Create labels that fit in the plot
    short_labels = []
    for label in token_labels:
        if len(label) > 10:
            short_labels.append(label[:7] + "...")
        else:
            short_labels.append(label)
    
    # Create heatmap
    ax = sns.heatmap(
        attention_matrix,
        cmap=cmap,
        xticklabels=short_labels,
        yticklabels=short_labels,
        vmin=0.0,
        vmax=1.0,
        annot=False,
        square=True
    )
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    # Add title and labels
    plt.title(title)
    plt.xlabel("Target Tokens")
    plt.ylabel("Source Tokens")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention matrix visualization to {output_path}")
    
    return plt.gcf()

def plot_token_importance(tokens: List[str],
                          importance_scores: List[float],
                          attention_scores: Optional[List[float]] = None,
                          output_path: Optional[str] = None,
                          title: str = "Token Importance",
                          figsize: Tuple[int, int] = (12, 6),
                          highlight_threshold: float = 0.7) -> plt.Figure:
    """
    Plot token importance scores.
    
    Args:
        tokens: List of tokens
        importance_scores: Importance scores for each token
        attention_scores: Attention scores for each token (optional)
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        highlight_threshold: Threshold for highlighting high-importance tokens
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Create x positions for bars
    x = np.arange(len(tokens))
    
    # Plot importance scores
    bars = plt.bar(
        x,
        importance_scores,
        width=0.4,
        color=[plt.cm.viridis(score) for score in importance_scores],
        alpha=0.7,
        label="Importance"
    )
    
    # Highlight high-importance tokens
    for i, score in enumerate(importance_scores):
        if score >= highlight_threshold:
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2)
    
    # Plot attention scores if provided
    if attention_scores:
        plt.plot(
            x,
            attention_scores,
            'ro-',
            alpha=0.7,
            markersize=6,
            label="Attention"
        )
    
    # Add token labels
    plt.xticks(x, tokens, rotation=45, ha="right")
    
    # Add title and labels
    plt.title(title)
    plt.xlabel("Tokens")
    plt.ylabel("Score")
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limit
    plt.ylim(0, 1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved token importance visualization to {output_path}")
    
    return plt.gcf()

def plot_counterfactual_impact(counterfactuals: List[Dict[str, Any]],
                              output_path: Optional[str] = None,
                              title: str = "Counterfactual Impact",
                              figsize: Tuple[int, int] = (12, 8),
                              max_cfs: int = 10) -> plt.Figure:
    """
    Plot counterfactual impact scores.
    
    Args:
        counterfactuals: List of counterfactual dictionaries
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        max_cfs: Maximum number of counterfactuals to show
        
    Returns:
        Matplotlib figure
    """
    # Sort counterfactuals by impact score
    sorted_cfs = sorted(counterfactuals, key=lambda cf: cf.get("impact_score", 0), reverse=True)
    
    # Limit to max_cfs
    cfs_to_plot = sorted_cfs[:max_cfs]
    
    # Extract data
    cf_labels = []
    impact_scores = []
    flip_status = []
    
    for cf in cfs_to_plot:
        orig = cf.get("original_token", "")
        alt = cf.get("alternative_token", "")
        label = f"'{orig}' → '{alt}'"
        cf_labels.append(label)
        
        impact_scores.append(cf.get("impact_score", 0))
        flip_status.append(cf.get("flipped_output", False))
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Create x positions for bars
    x = np.arange(len(cf_labels))
    
    # Create bars with colors based on flip status
    bars = plt.bar(
        x,
        impact_scores,
        width=0.6,
        color=[plt.cm.Reds(0.7) if flipped else plt.cm.Blues(0.7) for flipped in flip_status],
        alpha=0.8
    )
    
    # Add labels for flipped examples
    for i, flipped in enumerate(flip_status):
        if flipped:
            plt.text(
                x[i],
                impact_scores[i] + 0.02,
                "Flipped",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                color='red'
            )
    
    # Add labels
    plt.xticks(x, cf_labels, rotation=45, ha="right")
    
    # Add title and labels
    plt.title(title)
    plt.xlabel("Counterfactual Substitutions")
    plt.ylabel("Impact Score")
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=plt.cm.Reds(0.7), alpha=0.8, label='Flipped Output'),
        plt.Rectangle((0, 0), 1, 1, color=plt.cm.Blues(0.7), alpha=0.8, label='Same Output')
    ]
    plt.legend(handles=legend_elements)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limit
    plt.ylim(0, max(impact_scores) * 1.15 if impact_scores else 1.0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved counterfactual impact visualization to {output_path}")
    
    return plt.gcf()

def plot_knowledge_graph_validation(validation_results: Dict[str, Any],
                               output_path: Optional[str] = None,
                               title: str = "Knowledge Graph Validation",
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot knowledge graph validation results.
    
    Args:
        validation_results: Validation results dictionary
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Extract data from validation results
    paths = validation_results.get("paths", {})
    path_ids = list(paths.keys())
    
    # Collect data for each path
    supports = []
    contradicts = []
    unverified = []
    trustworthiness = []
    
    for path_id in path_ids:
        path_data = paths[path_id]
        supports.append(path_data.get("supported_statements", 0))
        contradicts.append(path_data.get("contradicted_statements", 0))
        unverified.append(path_data.get("unverified_statements", 0))
        trustworthiness.append(path_data.get("trustworthiness_score", 0.0))
    
    # Create bar positions
    x = np.arange(len(path_ids))
    width = 0.25
    
    # Create grouped bars
    plt.bar(x - width, supports, width=width, label='Supported', color='green', alpha=0.7)
    plt.bar(x, contradicts, width=width, label='Contradicted', color='red', alpha=0.7)
    plt.bar(x + width, unverified, width=width, label='Unverified', color='gray', alpha=0.7)
    
    # Plot trustworthiness scores
    ax2 = plt.twinx()
    ax2.plot(x, trustworthiness, 'bo-', label='Trustworthiness', linewidth=2, alpha=0.8)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('Trustworthiness Score')
    
    # Add labels
    plt.xticks(x, [f"Path {p}" for p in path_ids], rotation=0)
    
    # Add title and labels
    plt.title(title)
    plt.xlabel("Reasoning Paths")
    plt.ylabel("Statement Count")
    
    # Add legend
    handles1, labels1 = plt.gca().get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved knowledge graph validation visualization to {output_path}")
    
    return plt.gcf()

def plot_divergence_points(divergence_points: List[Dict[str, Any]],
                         original_tokens: List[str],
                         output_path: Optional[str] = None,
                         title: str = "Divergence Points",
                         figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot divergence points in the reasoning process.
    
    Args:
        divergence_points: List of divergence point dictionaries
        original_tokens: List of original tokens
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Create a chart showing the position of divergence points
    token_positions = list(range(len(original_tokens)))
    divergence_positions = [dp["position"] for dp in divergence_points]
    
    # Create indicator array (1 for divergence point, 0 otherwise)
    is_divergence = [1 if i in divergence_positions else 0 for i in token_positions]
    
    # Calculate severity scores
    severity_scores = [0.0] * len(original_tokens)
    for dp in divergence_points:
        pos = dp["position"]
        if 0 <= pos < len(severity_scores):
            severity_scores[pos] = dp.get("severity", 0.5)
    
    # Plot tokens as a bar chart
    token_colors = ['lightgray'] * len(original_tokens)
    for pos in divergence_positions:
        if 0 <= pos < len(token_colors):
            token_colors[pos] = plt.cm.Oranges(0.7)
    
    plt.bar(
        token_positions,
        [1] * len(original_tokens),
        color=token_colors,
        alpha=0.7,
        edgecolor='none'
    )
    
    # Add token labels
    plt.xticks(token_positions, original_tokens, rotation=45, ha="right")
    
    # Plot severity as a line
    ax2 = plt.twinx()
    ax2.plot(token_positions, severity_scores, 'ro-', label='Divergence Severity', alpha=0.7)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('Severity Score')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel("Tokens")
    plt.ylabel("Divergence")
    
    # Remove y ticks on the left axis
    plt.yticks([])
    
    # Add legend
    handles1 = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.Oranges(0.7), alpha=0.7, label='Divergence Point')]
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, ['Divergence Point', 'Severity'], loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved divergence points visualization to {output_path}")
    
    return plt.gcf()

def export_visualization_report(
    tree_builder,
    counterfactuals,
    validation_results,
    output_dir: str
) -> str:
    """
    Export a comprehensive visualization report.
    
    Args:
        tree_builder: CGRT tree builder
        counterfactuals: Counterfactual results
        validation_results: Knowledge graph validation results
        output_dir: Directory to save visualizations
        
    Returns:
        Path to the report directory
    """
    # Create output directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set up visualization style
    setup_visualization_style()
    
    # Generate and save visualizations
    
    # 1. Reasoning tree
    tree_path = os.path.join(viz_dir, "reasoning_tree.png")
    plot_reasoning_tree(
        tree_builder.graph,
        output_path=tree_path,
        title="Reasoning Tree"
    )
    
    # 2. Token importance
    token_importance_path = os.path.join(viz_dir, "token_importance.png")
    
    # Extract token importance data
    tokens = []
    importance_scores = []
    attention_scores = []
    
    for node_id, node in tree_builder.nodes.items():
        tokens.append(node.token)
        importance_scores.append(node.importance_score)
        attention_scores.append(node.attention_score)
    
    if tokens:
        plot_token_importance(
            tokens,
            importance_scores,
            attention_scores,
            output_path=token_importance_path,
            title="Token Importance and Attention"
        )
    
    # 3. Counterfactual impact
    if counterfactuals:
        cf_impact_path = os.path.join(viz_dir, "counterfactual_impact.png")
        plot_counterfactual_impact(
            counterfactuals,
            output_path=cf_impact_path,
            title="Counterfactual Impact"
        )
    
    # 4. Knowledge graph validation
    if validation_results:
        kg_path = os.path.join(viz_dir, "kg_validation.png")
        plot_knowledge_graph_validation(
            validation_results,
            output_path=kg_path,
            title="Knowledge Graph Validation"
        )
    
    # 5. Divergence points
    if hasattr(tree_builder, "divergence_points") and tree_builder.divergence_points:
        divergence_path = os.path.join(viz_dir, "divergence_points.png")
        
        # Get original tokens from the first path
        original_tokens = []
        if tree_builder.paths:
            for node_id in tree_builder.paths[0]:
                if node_id in tree_builder.nodes:
                    original_tokens.append(tree_builder.nodes[node_id].token)
        
        if original_tokens:
            plot_divergence_points(
                tree_builder.divergence_points,
                original_tokens,
                output_path=divergence_path,
                title="Divergence Points"
            )
    
    # Create index.html to view all visualizations
    html_path = os.path.join(viz_dir, "index.html")
    with open(html_path, "w") as f:
        f.write("<html><head><title>XAIR Visualization Report</title>\n")
        f.write("<style>body{font-family:Arial,sans-serif;margin:20px;}\n")
        f.write("h1{color:#333;}\n")
        f.write(".viz{margin-bottom:30px;}\n")
        f.write("img{max-width:100%;border:1px solid #ddd;}</style>\n")
        f.write("</head><body>\n")
        f.write("<h1>XAIR Visualization Report</h1>\n")
        
        # Add visualizations
        for img_name, title in [
            ("reasoning_tree.png", "Reasoning Tree"),
            ("token_importance.png", "Token Importance and Attention"),
            ("counterfactual_impact.png", "Counterfactual Impact"),
            ("kg_validation.png", "Knowledge Graph Validation"),
            ("divergence_points.png", "Divergence Points")
        ]:
            img_path = os.path.join(viz_dir, img_name)
            if os.path.exists(img_path):
                f.write(f"<div class='viz'><h2>{title}</h2>\n")
                f.write(f"<img src='{img_name}' alt='{title}'>\n")
                f.write("</div>\n")
        
        f.write("</body></html>\n")
    
    logger.info(f"Visualization report exported to {viz_dir}")
    return viz_dir