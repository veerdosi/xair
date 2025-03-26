"""
Visualization utilities for the XAIR system.
Provides functions for visualizing reasoning trees and counterfactuals.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

def create_tree_visualization(
    tree_data: Dict[str, Any],
    highlight_nodes: List[str] = None,
    output_path: Optional[str] = None,
    width: int = 1200,
    height: int = 800
) -> str:
    """
    Create a visualization of the reasoning tree.
    
    Args:
        tree_data: Tree data in DependenTree format
        highlight_nodes: List of node IDs to highlight
        output_path: Path to save the visualization
        width: Width of the visualization
        height: Height of the visualization
        
    Returns:
        Path to the saved visualization
    """
    # This is a simplified version that creates a JSON file for use with D3.js
    # A complete implementation would render an actual visualization
    
    if output_path is None:
        output_path = "tree_visualization.json"
    
    # Create a version of the tree with highlights
    if highlight_nodes:
        tree_with_highlights = add_highlights_to_tree(tree_data, highlight_nodes)
    else:
        tree_with_highlights = tree_data
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tree_with_highlights, f, indent=2)
    
    logger.info(f"Saved tree visualization data to {output_path}")
    
    return output_path

def add_highlights_to_tree(
    tree_data: Dict[str, Any],
    highlight_nodes: List[str]
) -> Dict[str, Any]:
    """
    Add highlight information to tree nodes.
    
    Args:
        tree_data: Tree data
        highlight_nodes: List of node IDs to highlight
        
    Returns:
        Tree data with highlight information
    """
    # Make a copy to avoid modifying the original
    import copy
    tree_copy = copy.deepcopy(tree_data)
    
    # Function to recursively process nodes
    def process_node(node):
        # Check if this node should be highlighted
        if "id" in node and node["id"] in highlight_nodes:
            node["highlighted"] = True
        else:
            node["highlighted"] = False
        
        # Process children
        if "children" in node:
            for child in node["children"]:
                process_node(child)
    
    # Process from the root
    process_node(tree_copy)
    
    return tree_copy

def create_counterfactual_visualization(
    counterfactuals: List[Any],
    output_path: Optional[str] = None
) -> str:
    """
    Create a visualization of counterfactuals.
    
    Args:
        counterfactuals: List of CounterfactualCandidate objects
        output_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    if not counterfactuals:
        logger.warning("No counterfactuals to visualize")
        return ""
    
    if output_path is None:
        output_path = "counterfactual_visualization.png"
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Extract data for plotting
    positions = [cf.position for cf in counterfactuals]
    impact_scores = [cf.impact_score for cf in counterfactuals]
    is_flipped = [cf.flipped_output for cf in counterfactuals]
    
    # Create scatter plot
    colors = ['red' if flipped else 'blue' for flipped in is_flipped]
    plt.scatter(positions, impact_scores, c=colors, s=100, alpha=0.7)
    
    # Add labels for top counterfactuals
    top_cfs = sorted(counterfactuals, key=lambda cf: cf.impact_score, reverse=True)[:5]
    for cf in top_cfs:
        plt.annotate(
            f"{cf.original_token} → {cf.alternative_token}",
            (cf.position, cf.impact_score),
            textcoords="offset points",
            xytext=(5, 5),
            ha='center'
        )
    
    # Add legend
    plt.scatter([], [], c='red', label='Flipped Output')
    plt.scatter([], [], c='blue', label='Non-Flipped Output')
    plt.legend()
    
    # Set labels and title
    plt.xlabel('Token Position')
    plt.ylabel('Impact Score')
    plt.title('Counterfactual Impact by Position')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved counterfactual visualization to {output_path}")
    
    return output_path

def create_impact_heatmap(
    text: str,
    positions: List[int],
    impact_scores: List[float],
    output_path: Optional[str] = None,
    width: int = 1000,
    height: int = 300
) -> str:
    """
    Create a heatmap visualization of token impacts.
    
    Args:
        text: Text to visualize
        positions: List of token positions
        impact_scores: List of impact scores
        output_path: Path to save the visualization
        width: Width of the visualization
        height: Height of the visualization
        
    Returns:
        Path to the saved visualization
    """
    if not positions or not impact_scores:
        logger.warning("No impact data to visualize")
        return ""
    
    if output_path is None:
        output_path = "impact_heatmap.png"
    
    # Create a blank image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to load a font
        font = ImageFont.truetype("Arial.ttf", 14)
    except IOError:
        # Fallback to default
        font = ImageFont.load_default()
    
    # Calculate token positions and sizes
    tokens = text.split()
    token_positions = []
    current_x = 20
    line_height = 30
    line_y = 20
    max_line_width = width - 40
    
    for token in tokens:
        token_width = draw.textlength(token, font=font) + 10  # Add padding
        
        # Check if we need to wrap to next line
        if current_x + token_width > max_line_width:
            current_x = 20
            line_y += line_height
        
        token_positions.append((current_x, line_y, token_width))
        current_x += token_width
    
    # Draw tokens with impact highlighting
    for i, (token, (x, y, w)) in enumerate(zip(tokens, token_positions)):
        # Check if this token has an impact score
        if i in positions:
            idx = positions.index(i)
            score = impact_scores[idx]
            
            # Calculate color based on impact (red for high impact)
            intensity = int(255 * score)
            color = (255, 255 - intensity, 255 - intensity)
            
            # Draw background rectangle
            draw.rectangle([x, y, x + w, y + line_height], fill=color)
        
        # Draw token text
        draw.text((x + 5, y + 5), token, fill="black", font=font)
    
    # Add legend
    legend_y = height - 70
    draw.rectangle([20, legend_y, 40, legend_y + 20], fill=(255, 255, 255))
    draw.text((45, legend_y), "Low Impact", fill="black", font=font)
    
    draw.rectangle([150, legend_y, 170, legend_y + 20], fill=(255, 128, 128))
    draw.text((175, legend_y), "Medium Impact", fill="black", font=font)
    
    draw.rectangle([300, legend_y, 320, legend_y + 20], fill=(255, 0, 0))
    draw.text((325, legend_y), "High Impact", fill="black", font=font)
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    
    logger.info(f"Saved impact heatmap to {output_path}")
    
    return output_path

def export_visualization_data(
    cgrt,
    counterfactual,
    output_dir: str = "viz_data"
) -> Dict[str, Any]:
    """
    Export visualization data for the frontend.
    
    Args:
        cgrt: CGRT instance
        counterfactual: Counterfactual instance
        output_dir: Directory to save visualization data
        
    Returns:
        Dictionary with visualization data information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    viz_data = {
        "tree": {},
        "counterfactuals": {},
        "paths": [],
        "stats": {},
        "files": []
    }
    
    # Export tree data
    tree_data = cgrt.tree_builder.to_dependentree_format()
    tree_file = os.path.join(output_dir, "tree_data.json")
    with open(tree_file, "w") as f:
        json.dump(tree_data, f, indent=2)
    viz_data["tree"] = tree_data
    viz_data["files"].append(tree_file)
    
    # Export counterfactual data
    cf_data = [cf.to_dict() for cf in counterfactual.counterfactuals]
    cf_file = os.path.join(output_dir, "counterfactual_data.json")
    with open(cf_file, "w") as f:
        json.dump(cf_data, f, indent=2)
    viz_data["counterfactuals"] = cf_data
    viz_data["files"].append(cf_file)
    
    # Export path data
    paths = cgrt.get_paths_text()
    viz_data["paths"] = paths
    paths_file = os.path.join(output_dir, "path_data.json")
    with open(paths_file, "w") as f:
        json.dump({"paths": paths}, f, indent=2)
    viz_data["files"].append(paths_file)
    
    # Export stats
    tree_stats = cgrt.get_tree_stats()
    cf_summary = counterfactual.get_counterfactual_summary()
    
    stats = {
        "tree": tree_stats,
        "counterfactuals": cf_summary
    }
    
    stats_file = os.path.join(output_dir, "stats_data.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    viz_data["stats"] = stats
    viz_data["files"].append(stats_file)
    
    # Create visualizations
    # Tree visualization
    highlight_nodes = []
    if counterfactual.counterfactuals:
        # Find nodes corresponding to top counterfactuals
        top_cfs = counterfactual.get_top_counterfactuals(5)
        for cf in top_cfs:
            # Find corresponding node IDs
            for node_id, node in cgrt.tree_builder.nodes.items():
                if node.position == cf.position:
                    highlight_nodes.append(node_id)
    
    tree_viz_file = os.path.join(output_dir, "tree_viz.json")
    create_tree_visualization(tree_data, highlight_nodes, tree_viz_file)
    viz_data["files"].append(tree_viz_file)
    
    # Counterfactual visualization
    if counterfactual.counterfactuals:
        cf_viz_file = os.path.join(output_dir, "counterfactual_viz.png")
        create_counterfactual_visualization(counterfactual.counterfactuals, cf_viz_file)
        viz_data["files"].append(cf_viz_file)
    
    logger.info(f"Exported visualization data to {output_dir}")
    
    return viz_data

def image_to_data_url(image_path: str) -> str:
    """
    Convert an image to a data URL.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Data URL of the image
    """
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Get the MIME type based on extension
    mime_type = "image/png"
    if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        mime_type = "image/jpeg"
    elif image_path.endswith(".svg"):
        mime_type = "image/svg+xml"
    
    # Convert to base64 and create data URL
    encoded = base64.b64encode(image_data).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"
    
    return data_url