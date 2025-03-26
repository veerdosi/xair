"""
Tree Builder module for the XAIR system.
Constructs a directed graph representing multiple reasoning paths.
"""

import os
import json
import networkx as nx
from typing import List, Dict, Tuple, Any, Optional, Set
import numpy as np
import logging
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

@dataclass
class TreeNode:
    """Represents a node in the CGRT."""
    id: str
    token: str
    token_id: int
    position: int
    path_ids: Set[int] = field(default_factory=set)
    probability: float = 0.0
    entropy: float = 0.0
    attention_score: float = 0.0
    importance_score: float = 0.0
    is_divergence_point: bool = False
    modified_importance: Optional[float] = None
    kg_entities: List[Dict[str, Any]] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["path_ids"] = list(self.path_ids)  # Convert set to list for JSON
        return result

@dataclass
class TreeEdge:
    """Represents an edge in the CGRT."""
    source_id: str
    target_id: str
    weight: float = 1.0
    path_ids: Set[int] = field(default_factory=set)
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["path_ids"] = list(self.path_ids)  # Convert set to list for JSON
        return result

class CGRTBuilder:
    """Builds a Counterfactual Graph Reasoning Tree from multiple reasoning paths."""
    
    def __init__(
        self,
        compression_enabled: bool = True,
        compression_threshold: float = 0.05,
        verbose: bool = False
    ):
        """
        Initialize the CGRT builder.
        
        Args:
            compression_enabled: Whether to enable tree compression
            compression_threshold: Threshold for compression
            verbose: Whether to log detailed information
        """
        self.compression_enabled = compression_enabled
        self.compression_threshold = compression_threshold
        self.verbose = verbose
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        
        # Initialize the graph
        self.reset()
    
    def reset(self):
        """Reset the tree builder."""
        self.graph = nx.DiGraph()
        self.nodes = {}  # Dict[node_id, TreeNode]
        self.edges = []  # List[TreeEdge]
        self.paths = []  # List of generation results
        self.divergence_points = []  # List of divergence points
    
    def build_tree(
        self,
        paths: List[Dict[str, Any]],
        divergence_points: List[Dict[str, Any]],
        tokenizer = None
    ) -> nx.DiGraph:
        """
        Build a CGRT from the given paths and divergence points.
        
        Args:
            paths: List of generation results
            divergence_points: List of divergence points
            tokenizer: Tokenizer to use for decoding tokens
            
        Returns:
            NetworkX DiGraph representing the CGRT
        """
        self.reset()
        self.paths = paths
        self.divergence_points = divergence_points
        
        # Check if we have enough paths
        if len(paths) < 1:
            logger.error("Need at least one path to build a tree")
            return self.graph
        
        # Process each path
        for path_idx, path in enumerate(paths):
            # Get token-level information
            if "generated_ids" not in path:
                logger.error(f"Path {path_idx} does not have generated_ids")
                continue
                
            input_ids = path.get("input_ids", None)
            if input_ids is None:
                logger.error(f"Path {path_idx} does not have input_ids")
                continue
                
            # Calculate the offset (prompt length)
            offset = input_ids.shape[1] if hasattr(input_ids, "shape") else len(input_ids[0])
            generated_ids = path["generated_ids"]
            
            # Extract token probabilities
            token_probs = path.get("token_probabilities", [])
            token_alternatives = path.get("token_alternatives", [])
            
            # Extract the token IDs for the generated part
            if hasattr(generated_ids, "shape"):  # For torch tensors
                token_ids = generated_ids[0, offset:].tolist()
            else:  # For lists or numpy arrays
                token_ids = generated_ids[0][offset:]
            
            # Create nodes for each token
            prev_node_id = None
            for pos, token_id in enumerate(token_ids):
                # Skip if we're beyond the token probabilities
                if pos >= len(token_probs):
                    break
                
                # Get token probability
                token_prob = token_probs[pos] if pos < len(token_probs) else 0.0
                
                # Get alternative tokens
                alternatives = token_alternatives[pos] if pos < len(token_alternatives) else []
                
                # Check if this position is a divergence point
                is_divergence = any(
                    (dp["position"] == pos and path_idx in dp["path_indices"])
                    for dp in divergence_points
                )
                
                # Decode the token
                if tokenizer:
                    token = tokenizer.decode([token_id])
                else:
                    token = f"[Token {token_id}]"
                
                # Create a node ID
                node_id = f"path{path_idx}_pos{pos}_token{token_id}"
                
                # Create the node or update an existing one
                if node_id in self.nodes:
                    # Update the existing node
                    node = self.nodes[node_id]
                    node.path_ids.add(path_idx)
                    node.probability = max(node.probability, token_prob)  # Take the max probability
                else:
                    # Create a new node
                    node = TreeNode(
                        id=node_id,
                        token=token,
                        token_id=token_id,
                        position=pos,
                        path_ids={path_idx},
                        probability=token_prob,
                        is_divergence_point=is_divergence,
                        alternatives=alternatives
                    )
                    self.nodes[node_id] = node
                
                # Add the node to the graph
                self.graph.add_node(
                    node_id,
                    token=token,
                    token_id=token_id,
                    position=pos,
                    probability=token_prob,
                    is_divergence_point=is_divergence
                )
                
                # Connect to the previous node if it exists
                if prev_node_id:
                    # Create an edge ID
                    edge_id = f"{prev_node_id}_to_{node_id}"
                    
                    # Add the edge to the graph
                    self.graph.add_edge(
                        prev_node_id,
                        node_id,
                        weight=token_prob,
                        path_id=path_idx
                    )
                    
                    # Create the edge object
                    edge = TreeEdge(
                        source_id=prev_node_id,
                        target_id=node_id,
                        weight=token_prob,
                        path_ids={path_idx}
                    )
                    self.edges.append(edge)
                
                # Update the previous node
                prev_node_id = node_id
        
        # Apply compression if enabled
        if self.compression_enabled:
            self._compress_tree()
        
        # Calculate initial importance scores
        self._calculate_importance_scores()
        
        return self.graph
    
    def _compress_tree(self):
        """
        Compress the tree by merging similar nodes.
        """
        # Identify nodes to merge - those with similar token and position but different paths
        pos_token_nodes = {}  # Dict[(position, token_id), List[node_id]]
        
        for node_id, node in self.nodes.items():
            key = (node.position, node.token_id)
            if key not in pos_token_nodes:
                pos_token_nodes[key] = []
            pos_token_nodes[key].append(node_id)
        
        # Merge nodes with the same token at the same position
        for (pos, token_id), node_ids in pos_token_nodes.items():
            if len(node_ids) <= 1:
                continue
            
            # Choose the first node as the primary node
            primary_node_id = node_ids[0]
            primary_node = self.nodes[primary_node_id]
            
            # Merge the rest of the nodes into the primary node
            for node_id in node_ids[1:]:
                if node_id == primary_node_id:
                    continue
                    
                node = self.nodes[node_id]
                
                # Update the primary node
                primary_node.path_ids.update(node.path_ids)
                primary_node.probability = max(primary_node.probability, node.probability)
                primary_node.is_divergence_point = primary_node.is_divergence_point or node.is_divergence_point
                
                # Update incoming edges
                for edge in self.edges:
                    if edge.target_id == node_id:
                        edge.target_id = primary_node_id
                        edge.path_ids.update(node.path_ids)
                
                # Update outgoing edges
                for edge in self.edges:
                    if edge.source_id == node_id:
                        edge.source_id = primary_node_id
                        edge.path_ids.update(node.path_ids)
                
                # Remove the merged node
                del self.nodes[node_id]
                
                # Update the graph
                if self.graph.has_node(node_id):
                    # Redirect edges
                    for pred in list(self.graph.predecessors(node_id)):
                        edge_data = self.graph.get_edge_data(pred, node_id)
                        self.graph.add_edge(pred, primary_node_id, **edge_data)
                    
                    for succ in list(self.graph.successors(node_id)):
                        edge_data = self.graph.get_edge_data(node_id, succ)
                        self.graph.add_edge(primary_node_id, succ, **edge_data)
                    
                    # Remove the node
                    self.graph.remove_node(node_id)
        
        # Rebuild the list of edges
        self.edges = [edge for edge in self.edges 
                     if edge.source_id in self.nodes and edge.target_id in self.nodes]
        
        # Update the graph structure
        self._rebuild_graph()
    
    def _rebuild_graph(self):
        """
        Rebuild the graph from nodes and edges.
        """
        # Create a new graph
        new_graph = nx.DiGraph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            new_graph.add_node(
                node_id,
                **node.to_dict()
            )
        
        # Add edges
        for edge in self.edges:
            if edge.source_id in self.nodes and edge.target_id in self.nodes:
                new_graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    **edge.to_dict()
                )
        
        self.graph = new_graph
    
    def _calculate_importance_scores(self):
        """
        Calculate initial importance scores for nodes.
        """
        # Initialize importance scores based on:
        # 1. Divergence points (high importance)
        # 2. Node probability (higher prob → higher importance)
        # 3. Position in the sequence (later tokens may depend on earlier ones)
        # 4. Number of paths containing the node (more paths → higher importance)
        
        # First, normalize probabilities
        max_prob = max((node.probability for node in self.nodes.values()), default=1.0)
        
        # Calculate scores
        for node_id, node in self.nodes.items():
            # Base score from probability
            prob_score = node.probability / max_prob if max_prob > 0 else 0.5
            
            # Bonus for divergence points
            divergence_bonus = 0.5 if node.is_divergence_point else 0.0
            
            # Bonus for nodes in multiple paths
            path_ratio = len(node.path_ids) / len(self.paths) if self.paths else 0
            path_bonus = 0.3 * path_ratio
            
            # Calculate importance score
            importance = prob_score + divergence_bonus + path_bonus
            
            # Normalize to [0, 1]
            importance = min(1.0, importance)
            
            # Update the node
            node.importance_score = importance
            
            # Update the graph
            self.graph.nodes[node_id]["importance_score"] = importance
    
    def adjust_node_importance(
        self,
        node_id: str,
        new_importance: float,
        propagate: bool = True,
        propagation_factor: float = 0.5
    ):
        """
        Adjust the importance score of a node.
        
        Args:
            node_id: ID of the node to adjust
            new_importance: New importance score
            propagate: Whether to propagate the change to connected nodes
            propagation_factor: Factor for propagation (0-1)
        """
        if node_id not in self.nodes:
            logger.error(f"Node {node_id} not found")
            return
        
        # Get the node
        node = self.nodes[node_id]
        
        # Store the original importance
        original_importance = node.importance_score
        
        # Update the node
        node.modified_importance = new_importance
        
        # Update the graph
        self.graph.nodes[node_id]["modified_importance"] = new_importance
        
        # Propagate the change if requested
        if propagate:
            self._propagate_importance_change(
                node_id, 
                original_importance, 
                new_importance, 
                propagation_factor
            )
    
    def _propagate_importance_change(
            self,
            node_id: str,
            original_importance: float,
            new_importance: float,
            propagation_factor: float,
            visited: Set[str] = None
        ):
            """
            Propagate importance changes to connected nodes.
            
            Args:
                node_id: ID of the node that was changed
                original_importance: Original importance score
                new_importance: New importance score
                propagation_factor: Factor for propagation (0-1)
                visited: Set of already visited nodes
            """
            if visited is None:
                visited = set()
            
            if node_id in visited:
                return
            
            visited.add(node_id)
            
            # Calculate the importance difference
            diff = new_importance - original_importance
            
            # Propagate to successors (downstream effects)
            for succ in self.graph.successors(node_id):
                if succ in visited:
                    continue
                    
                # Calculate the propagated change
                prop_diff = diff * propagation_factor
                
                # Get the successor node
                succ_node = self.nodes[succ]
                
                # Calculate new importance
                current_importance = succ_node.modified_importance if succ_node.modified_importance is not None else succ_node.importance_score
                new_succ_importance = max(0.0, min(1.0, current_importance + prop_diff))
                
                # Update the node
                succ_node.modified_importance = new_succ_importance
                
                # Update the graph
                self.graph.nodes[succ]["modified_importance"] = new_succ_importance
                
                # Continue propagation with reduced factor
                self._propagate_importance_change(
                    succ,
                    current_importance,
                    new_succ_importance,
                    propagation_factor * 0.7,
                    visited
                )
        
    def save_tree(self, output_path: str):
        """
        Save the tree to a file.
        
        Args:
            output_path: Path to save the tree
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare the data
        tree_data = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges]
        }
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(tree_data, f, indent=2)
        
        logger.info(f"Saved tree to {output_path}")
    
    def load_tree(self, input_path: str):
        """
        Load a tree from a file.
        
        Args:
            input_path: Path to load the tree from
        """
        # Check if the file exists
        if not os.path.exists(input_path):
            logger.error(f"Tree file {input_path} not found")
            return
        
        # Load the data
        with open(input_path, "r") as f:
            tree_data = json.load(f)
        
        # Reset the tree
        self.reset()
        
        # Load nodes
        for node_data in tree_data["nodes"]:
            # Convert path_ids back to set
            path_ids = set(node_data.pop("path_ids"))
            
            # Create the node
            node = TreeNode(
                **node_data,
                path_ids=path_ids
            )
            
            # Add to nodes
            self.nodes[node.id] = node
        
        # Load edges
        for edge_data in tree_data["edges"]:
            # Convert path_ids back to set
            path_ids = set(edge_data.pop("path_ids"))
            
            # Create the edge
            edge = TreeEdge(
                **edge_data,
                path_ids=path_ids
            )
            
            # Add to edges
            self.edges.append(edge)
        
        # Rebuild the graph
        self._rebuild_graph()
        
        logger.info(f"Loaded tree from {input_path}")
    
    def get_attention_analysis(self):
        """
        Perform attention analysis on the tree.
        
        Returns:
            Dictionary with attention analysis results
        """
        # This is a placeholder for the attention analysis module
        # We'll integrate with the attention.py module once it's implemented
        return {
            "attention_flow": {},
            "importance_scores": {node_id: node.importance_score for node_id, node in self.nodes.items()}
        }
    
    def to_dependentree_format(self):
        """
        Convert the tree to DependenTree format for visualization.
        
        Returns:
            Dictionary in DependenTree format
        """
        # Create a mapping from position to nodes
        pos_nodes = {}
        for node_id, node in self.nodes.items():
            if node.position not in pos_nodes:
                pos_nodes[node.position] = []
            pos_nodes[node.position].append(node)
        
        # Sort positions
        positions = sorted(pos_nodes.keys())
        
        # Create the tree structure
        tree = {
            "text": "",
            "children": []
        }
        
        # Add nodes by position
        current_level = tree
        for pos in positions:
            # Sort nodes by importance
            nodes = sorted(pos_nodes[pos], key=lambda n: n.importance_score, reverse=True)
            
            # Create child nodes
            children = []
            for node in nodes:
                child = {
                    "id": node.id,
                    "text": node.token,
                    "importance": node.importance_score,
                    "modified_importance": node.modified_importance,
                    "is_divergence": node.is_divergence_point,
                    "children": []
                }
                children.append(child)
            
            # Add to the current level
            if children:
                current_level["children"] = children
                current_level = children[0]  # Follow the highest importance path
        
        return tree