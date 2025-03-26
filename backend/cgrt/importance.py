"""
Node Importance Adjustment module for the XAIR system.
Handles post-hoc modification of node importance values.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
import logging
import json
import os
import time  # Proper import of time module

logger = logging.getLogger(__name__)

class ImportanceAdjuster:
    """Handles adjustments to node importance in the CGRT."""
    
    def __init__(
        self,
        propagation_enabled: bool = True,
        propagation_factor: float = 0.5,
        track_history: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the importance adjuster.
        
        Args:
            propagation_enabled: Whether to propagate changes to connected nodes
            propagation_factor: Factor for propagation (0-1)
            track_history: Whether to track adjustment history
            verbose: Whether to log detailed information
        """
        self.propagation_enabled = propagation_enabled
        self.propagation_factor = propagation_factor
        self.track_history = track_history
        self.verbose = verbose
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        
        # Initialize history
        self.adjustment_history = []
    
    def adjust_node_importance(
        self,
        tree_builder,
        node_id: str,
        new_importance: float,
        reason: str = "",
        user_id: str = "user"
    ) -> bool:
        """
        Adjust the importance of a node in the tree.
        
        Args:
            tree_builder: CGRTBuilder instance
            node_id: ID of the node to adjust
            new_importance: New importance score
            reason: Reason for the adjustment
            user_id: ID of the user making the adjustment
            
        Returns:
            Whether the adjustment was successful
        """
        # Check if the node exists
        if node_id not in tree_builder.nodes:
            logger.error(f"Node {node_id} not found")
            return False
        
        # Get the node
        node = tree_builder.nodes[node_id]
        
        # Record the original importance
        original_importance = node.importance_score
        
        # Validate the new importance
        new_importance = max(0.0, min(1.0, new_importance))
        
        # Update the node
        node.modified_importance = new_importance
        
        # Update the graph
        tree_builder.graph.nodes[node_id]["modified_importance"] = new_importance
        
        # Propagate the change if enabled
        if self.propagation_enabled:
            self._propagate_importance_change(
                tree_builder,
                node_id,
                original_importance,
                new_importance,
                self.propagation_factor
            )
        
        # Track the adjustment
        if self.track_history:
            adjustment = {
                "timestamp": time.time(),  # Proper use of time module
                "node_id": node_id,
                "original_importance": original_importance,
                "new_importance": new_importance,
                "reason": reason,
                "user_id": user_id,
                "propagation_enabled": self.propagation_enabled,
                "propagation_factor": self.propagation_factor
            }
            self.adjustment_history.append(adjustment)
        
        logger.info(f"Adjusted importance of node {node_id}: {original_importance} -> {new_importance}")
        return True
    
    def _propagate_importance_change(
        self,
        tree_builder,
        node_id: str,
        original_importance: float,
        new_importance: float,
        propagation_factor: float,
        visited: Set[str] = None
    ):
        """
        Propagate importance changes to connected nodes.
        
        Args:
            tree_builder: CGRTBuilder instance
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
        for succ in tree_builder.graph.successors(node_id):
            if succ in visited:
                continue
                
            # Calculate the propagated change
            prop_diff = diff * propagation_factor
            
            # Get the successor node
            succ_node = tree_builder.nodes[succ]
            
            # Calculate new importance
            current_importance = succ_node.modified_importance if succ_node.modified_importance is not None else succ_node.importance_score
            new_succ_importance = max(0.0, min(1.0, current_importance + prop_diff))
            
            # Update the node
            succ_node.modified_importance = new_succ_importance
            
            # Update the graph
            tree_builder.graph.nodes[succ]["modified_importance"] = new_succ_importance
            
            # Continue propagation with reduced factor
            self._propagate_importance_change(
                tree_builder,
                succ,
                current_importance,
                new_succ_importance,
                propagation_factor * 0.7,
                visited
            )
    
    def batch_adjust_nodes(
        self,
        tree_builder,
        adjustments: List[Dict[str, Any]],
        user_id: str = "user"
    ) -> Dict[str, Any]:
        """
        Apply multiple adjustments in batch.
        
        Args:
            tree_builder: CGRTBuilder instance
            adjustments: List of adjustment dictionaries
            user_id: ID of the user making the adjustments
            
        Returns:
            Dictionary with results
        """
        results = {
            "success": [],
            "failed": []
        }
        
        # Apply each adjustment
        for adj in adjustments:
            node_id = adj.get("node_id")
            new_importance = adj.get("new_importance")
            reason = adj.get("reason", "")
            
            if node_id is None or new_importance is None:
                results["failed"].append({
                    "node_id": node_id,
                    "reason": "Missing node_id or new_importance"
                })
                continue
            
            # Apply the adjustment
            success = self.adjust_node_importance(
                tree_builder,
                node_id,
                new_importance,
                reason,
                user_id
            )
            
            if success:
                results["success"].append({
                    "node_id": node_id,
                    "new_importance": new_importance
                })
            else:
                results["failed"].append({
                    "node_id": node_id,
                    "reason": "Node not found or invalid importance"
                })
        
        return results
    
    def reset_node_importance(
        self,
        tree_builder,
        node_id: str,
        user_id: str = "user"
    ) -> bool:
        """
        Reset the importance of a node to its original value.
        
        Args:
            tree_builder: CGRTBuilder instance
            node_id: ID of the node to reset
            user_id: ID of the user making the reset
            
        Returns:
            Whether the reset was successful
        """
        # Check if the node exists
        if node_id not in tree_builder.nodes:
            logger.error(f"Node {node_id} not found")
            return False
        
        # Get the node
        node = tree_builder.nodes[node_id]
        
        # Check if the node has been modified
        if node.modified_importance is None:
            logger.info(f"Node {node_id} has not been modified")
            return True
        
        # Record the original values
        modified_importance = node.modified_importance
        original_importance = node.importance_score
        
        # Reset the node
        node.modified_importance = None
        
        # Update the graph
        if "modified_importance" in tree_builder.graph.nodes[node_id]:
            del tree_builder.graph.nodes[node_id]["modified_importance"]
        
        # Propagate the change if enabled
        if self.propagation_enabled:
            self._propagate_importance_change(
                tree_builder,
                node_id,
                modified_importance,
                original_importance,
                self.propagation_factor
            )
        
        # Track the adjustment
        if self.track_history:
            adjustment = {
                "timestamp": time.time(),  # Proper use of time module
                "node_id": node_id,
                "original_importance": modified_importance,
                "new_importance": original_importance,
                "reason": "Reset to original value",
                "user_id": user_id,
                "is_reset": True,
                "propagation_enabled": self.propagation_enabled,
                "propagation_factor": self.propagation_factor
            }
            self.adjustment_history.append(adjustment)
        
        logger.info(f"Reset importance of node {node_id} to original value: {original_importance}")
        return True
    
    def compare_original_vs_adjusted(self, tree_builder) -> Dict[str, Any]:
        """
        Compare original vs adjusted importance scores.
        
        Args:
            tree_builder: CGRTBuilder instance
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "modified_nodes": [],
            "total_nodes": len(tree_builder.nodes),
            "average_change": 0.0,
            "max_increase": 0.0,
            "max_decrease": 0.0
        }
        
        total_change = 0.0
        count = 0
        
        # Collect modifications
        for node_id, node in tree_builder.nodes.items():
            if node.modified_importance is not None:
                change = node.modified_importance - node.importance_score
                
                comparison["modified_nodes"].append({
                    "node_id": node_id,
                    "original": node.importance_score,
                    "modified": node.modified_importance,
                    "change": change
                })
                
                total_change += abs(change)
                count += 1
                
                if change > comparison["max_increase"]:
                    comparison["max_increase"] = change
                
                if change < comparison["max_decrease"]:
                    comparison["max_decrease"] = change
        
        # Calculate average change
        if count > 0:
            comparison["average_change"] = total_change / count
        
        comparison["percentage_modified"] = count / max(1, comparison["total_nodes"]) * 100
        
        return comparison
    
    def save_adjustment_history(self, output_path: str):
        """
        Save the adjustment history to a file.
        
        Args:
            output_path: Path to save the history
        """
        if not self.track_history:
            logger.warning("History tracking is disabled")
            return
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(self.adjustment_history, f, indent=2)
        
        logger.info(f"Saved adjustment history to {output_path}")
    
    def load_adjustment_history(self, input_path: str):
        """
        Load adjustment history from a file.
        
        Args:
            input_path: Path to load the history from
        """
        # Check if the file exists
        if not os.path.exists(input_path):
            logger.error(f"History file {input_path} not found")
            return
        
        # Load the data
        with open(input_path, "r") as f:
            self.adjustment_history = json.load(f)
        
        logger.info(f"Loaded adjustment history from {input_path}")
    
    def apply_history_to_tree(self, tree_builder, user_id: str = None):
        """
        Apply saved adjustment history to a tree.
        
        Args:
            tree_builder: CGRTBuilder instance
            user_id: Filter history by user ID if provided
        """
        if not self.adjustment_history:
            logger.warning("No adjustment history to apply")
            return
        
        # Sort history by timestamp
        sorted_history = sorted(self.adjustment_history, key=lambda x: x.get("timestamp", 0))
        
        # Track already processed nodes to avoid duplicates
        processed_nodes = set()
        
        # Apply each adjustment
        for adjustment in sorted_history:
            # Skip if not matching user_id filter
            if user_id is not None and adjustment.get("user_id") != user_id:
                continue
            
            node_id = adjustment.get("node_id")
            
            # Skip reset operations for nodes we've seen
            if adjustment.get("is_reset", False) and node_id in processed_nodes:
                continue
            
            # Apply the adjustment
            if adjustment.get("is_reset", False):
                self.reset_node_importance(
                    tree_builder,
                    node_id,
                    adjustment.get("user_id", "user")
                )
            else:
                self.adjust_node_importance(
                    tree_builder,
                    node_id,
                    adjustment.get("new_importance"),
                    adjustment.get("reason", ""),
                    adjustment.get("user_id", "user")
                )
            
            # Mark as processed
            processed_nodes.add(node_id)
        
        logger.info(f"Applied {len(sorted_history)} adjustments from history")
    
    def get_top_adjusted_nodes(self, tree_builder, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top N nodes with the largest importance adjustments.
        
        Args:
            tree_builder: CGRTBuilder instance
            n: Number of nodes to return
            
        Returns:
            List of node data dictionaries
        """
        adjusted_nodes = []
        
        for node_id, node in tree_builder.nodes.items():
            if node.modified_importance is not None:
                change = abs(node.modified_importance - node.importance_score)
                
                adjusted_nodes.append({
                    "node_id": node_id,
                    "token": node.token,
                    "position": node.position,
                    "original_importance": node.importance_score,
                    "modified_importance": node.modified_importance,
                    "change": change
                })
        
        # Sort by absolute change
        sorted_nodes = sorted(adjusted_nodes, key=lambda x: x["change"], reverse=True)
        
        # Return top N
        return sorted_nodes[:n]
    
    def get_adjustment_summary_by_user(self) -> Dict[str, Any]:
        """
        Get a summary of adjustments by user.
        
        Returns:
            Dictionary with user-based summary
        """
        if not self.adjustment_history:
            return {}
        
        summary = {}
        
        for adjustment in self.adjustment_history:
            user_id = adjustment.get("user_id", "unknown")
            
            if user_id not in summary:
                summary[user_id] = {
                    "count": 0,
                    "resets": 0,
                    "increases": 0,
                    "decreases": 0,
                    "average_change": 0.0
                }
            
            # Update count
            summary[user_id]["count"] += 1
            
            # Check if it's a reset
            if adjustment.get("is_reset", False):
                summary[user_id]["resets"] += 1
            else:
                # Calculate change
                original = adjustment.get("original_importance", 0.0)
                new = adjustment.get("new_importance", 0.0)
                change = new - original
                
                # Update summary
                if change > 0:
                    summary[user_id]["increases"] += 1
                elif change < 0:
                    summary[user_id]["decreases"] += 1
                
                # Update average change
                current_avg = summary[user_id]["average_change"]
                current_count = summary[user_id]["count"] - summary[user_id]["resets"]
                
                if current_count > 0:
                    summary[user_id]["average_change"] = (current_avg * (current_count - 1) + abs(change)) / current_count
        
        return summary