"""
Main CGRT module for the backend system.
Integrates all CGRT components into a unified interface.
"""

import os
import json
import logging
import time
from typing import List, Dict, Tuple, Any, Optional

from backend.models.llm_interface import LlamaInterface, GenerationConfig
from backend.cgrt.divergence import DivergenceDetector
from backend.cgrt.tree_builder import CGRTBuilder
from backend.cgrt.attention import AttentionAnalyzer
from backend.cgrt.importance import ImportanceAdjuster

logger = logging.getLogger(__name__)

class CGRT:
    """
    Counterfactual Graph Reasoning Tree main class.
    Integrates all CGRT components.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-3.2-1B",
        device: str = "auto",
        load_in_4bit: bool = False,  # Kept for API compatibility but ignored
        temperatures: List[float] = [0.2, 0.7, 1.0],
        paths_per_temp: int = 1,
        max_new_tokens: int = 512,
        kl_threshold: float = 0.5,
        context_window_size: int = 5,
        compression_enabled: bool = True,
        propagation_enabled: bool = True,
        propagation_factor: float = 0.5,
        output_dir: str = "output",
        use_fp16: bool = True,
        use_bettertransformer: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the CGRT system.
        
        Args:
            model_name_or_path: Path to the model or model identifier
            device: Device to load the model on ('cpu', 'cuda', 'mps', or 'auto')
            load_in_4bit: Ignored for Mac compatibility
            temperatures: List of temperature values to use
            paths_per_temp: Number of paths to generate per temperature
            max_new_tokens: Maximum number of tokens to generate
            kl_threshold: Base threshold for KL divergence
            context_window_size: Number of tokens to include before and after a divergence point
            compression_enabled: Whether to enable tree compression
            propagation_enabled: Whether to propagate importance changes
            propagation_factor: Factor for propagation
            output_dir: Directory to save outputs
            use_fp16: Whether to use half precision
            use_bettertransformer: Whether to use BetterTransformer
            verbose: Whether to log detailed information
        """
        self.temperatures = temperatures
        self.paths_per_temp = paths_per_temp
        self.max_new_tokens = max_new_tokens
        self.output_dir = output_dir
        self.verbose = verbose
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing LLM interface...")
        self.llm = LlamaInterface(
            model_name_or_path=model_name_or_path,
            device=device,
            load_in_4bit=False,  # Explicitly set to False for Mac compatibility
            use_fp16=use_fp16,
            use_bettertransformer=use_bettertransformer,
            verbose=verbose
        )
        
        logger.info("Initializing divergence detector...")
        self.divergence_detector = DivergenceDetector(
            kl_threshold=kl_threshold,
            context_window_size=context_window_size,
            adaptive_threshold=True,
            verbose=verbose
        )
        
        logger.info("Initializing tree builder...")
        self.tree_builder = CGRTBuilder(
            compression_enabled=compression_enabled,
            verbose=verbose
        )
        
        logger.info("Initializing attention analyzer...")
        self.attention_analyzer = AttentionAnalyzer(
            attention_weight_by_layer=True,
            verbose=verbose
        )
        
        logger.info("Initializing importance adjuster...")
        self.importance_adjuster = ImportanceAdjuster(
            propagation_enabled=propagation_enabled,
            propagation_factor=propagation_factor,
            track_history=True,
            verbose=verbose
        )
        
        # State variables
        self.paths = []  # Generated paths
        self.divergence_points = []  # Detected divergence points
        
        logger.info("CGRT system initialized successfully")
    
    def process_input(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        save_results: bool = True
    ):
        """
        Process an input prompt through the CGRT pipeline.
        
        Args:
            prompt: Input prompt
            generation_config: Configuration for generation
            save_results: Whether to save results to disk
            
        Returns:
            The constructed CGRT
        """
        # Create default generation config if not provided
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                output_hidden_states=True,
                output_attentions=True
            )
        
        # 1. Generate multiple reasoning paths
        logger.info("Generating multiple reasoning paths...")
        self.paths = self.llm.generate_multiple_paths(
            prompt,
            temperatures=self.temperatures,
            paths_per_temp=self.paths_per_temp,
            generation_config=generation_config
        )
        
        # Save the generation results if requested
        if save_results:
            output_path = os.path.join(self.output_dir, "generation_results.json")
            self.llm.save_generation_results(self.paths, self.output_dir)
        
        # 2. Identify divergence points
        logger.info("Identifying divergence points...")
        self.divergence_points = self.divergence_detector.detect_divergences(self.paths)
        
        if save_results:
            divergence_path = os.path.join(self.output_dir, "divergence_points.json")
            with open(divergence_path, "w") as f:
                json.dump(self.divergence_points, f, indent=2)
        
        # 3. Build the tree
        logger.info("Building the tree...")
        self.tree_builder.build_tree(
            self.paths,
            self.divergence_points,
            self.llm.tokenizer
        )
        
        # 4. Analyze attention
        logger.info("Analyzing attention patterns...")
        self.attention_analyzer.update_tree_with_attention(
            self.tree_builder,
            self.paths
        )
        
        # 5. Save the tree if requested
        if save_results:
            tree_path = os.path.join(self.output_dir, "reasoning_tree.json")
            self.tree_builder.save_tree(tree_path)
        
        logger.info("Input processing complete")
        return self.tree_builder.graph
    
    def adjust_node_importance(
        self,
        node_id: str,
        new_importance: float,
        reason: str = "",
        user_id: str = "user"
    ) -> bool:
        """
        Adjust the importance of a node in the tree.
        
        Args:
            node_id: ID of the node to adjust
            new_importance: New importance score
            reason: Reason for the adjustment
            user_id: ID of the user making the adjustment
            
        Returns:
            Whether the adjustment was successful
        """
        return self.importance_adjuster.adjust_node_importance(
            self.tree_builder,
            node_id,
            new_importance,
            reason,
            user_id
        )
    
    def reset_node_importance(
        self,
        node_id: str,
        user_id: str = "user"
    ) -> bool:
        """
        Reset the importance of a node to its original value.
        
        Args:
            node_id: ID of the node to reset
            user_id: ID of the user making the reset
            
        Returns:
            Whether the reset was successful
        """
        return self.importance_adjuster.reset_node_importance(
            self.tree_builder,
            node_id,
            user_id
        )
    
    def compare_original_vs_adjusted(self) -> Dict[str, Any]:
        """
        Compare original vs adjusted importance scores.
        
        Returns:
            Dictionary with comparison results
        """
        return self.importance_adjuster.compare_original_vs_adjusted(self.tree_builder)
    
    def get_top_adjusted_nodes(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top N nodes with the largest importance adjustments.
        
        Args:
            n: Number of nodes to return
            
        Returns:
            List of node data dictionaries
        """
        return self.importance_adjuster.get_top_adjusted_nodes(self.tree_builder, n)
    
    def save_adjustment_history(self, output_path: Optional[str] = None):
        """
        Save the adjustment history to a file.
        
        Args:
            output_path: Path to save the history
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "adjustment_history.json")
        
        self.importance_adjuster.save_adjustment_history(output_path)
    
    def load_adjustment_history(self, input_path: str):
        """
        Load adjustment history from a file.
        
        Args:
            input_path: Path to load the history from
        """
        self.importance_adjuster.load_adjustment_history(input_path)
    
    def apply_history_to_tree(self, user_id: str = None):
        """
        Apply saved adjustment history to the current tree.
        
        Args:
            user_id: Filter history by user ID if provided
        """
        self.importance_adjuster.apply_history_to_tree(self.tree_builder, user_id)
    
    def to_dependentree_format(self) -> Dict[str, Any]:
        """
        Convert the tree to DependenTree format for visualization.
        
        Returns:
            Dictionary in DependenTree format
        """
        return self.tree_builder.to_dependentree_format()
    
    def get_paths_text(self) -> List[str]:
        """
        Get the text of all generated paths.
        
        Returns:
            List of path texts
        """
        return [path.get("generated_text", "") for path in self.paths]
    
    def get_divergence_summary(self) -> Dict[str, Any]:
        """
        Get a summary of detected divergence points.
        
        Returns:
            Dictionary with divergence summary
        """
        if not self.divergence_points:
            return {"count": 0, "points": []}
        
        divergence_summary = {
            "count": len(self.divergence_points),
            "points": []
        }
        
        for dp in self.divergence_points:
            summary_point = {
                "position": dp["position"],
                "path_indices": dp["path_indices"],
                "kl_divergence": dp["kl_divergence"],
                "probability_diff": dp["probability_diff"],
                "tokens": {
                    "path1": dp["tokens"]["path1"]["token"],
                    "path2": dp["tokens"]["path2"]["token"]
                }
            }
            
            # Add context if available
            if "context_window" in dp:
                summary_point["context"] = {
                    "path1": dp["context_window"].get("path1", []),
                    "path2": dp["context_window"].get("path2", [])
                }
            
            divergence_summary["points"].append(summary_point)
        
        return divergence_summary
    
    def get_tree_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the constructed tree.
        
        Returns:
            Dictionary with tree statistics
        """
        graph = self.tree_builder.graph
        
        stats = {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "max_depth": 0,
            "divergence_points": len([n for n, data in graph.nodes(data=True) if data.get("is_divergence_point", False)]),
            "paths_count": len(self.paths),
            "avg_node_importance": 0.0,
            "modified_nodes": 0
        }
        
        # Calculate max depth
        roots = [n for n, d in graph.in_degree() if d == 0]
        if roots:
            # Use the first root for depth calculation
            max_depth = 0
            visited = set()
            
            def dfs_depth(node, depth):
                nonlocal max_depth
                if node in visited:
                    return
                visited.add(node)
                
                max_depth = max(max_depth, depth)
                
                for succ in graph.successors(node):
                    dfs_depth(succ, depth + 1)
            
            dfs_depth(roots[0], 0)
            stats["max_depth"] = max_depth
        
        # Calculate average importance
        if graph.nodes:
            importance_sum = sum(data.get("importance_score", 0.0) for n, data in graph.nodes(data=True))
            stats["avg_node_importance"] = importance_sum / len(graph.nodes)
        
        # Count modified nodes
        stats["modified_nodes"] = len([n for n, data in graph.nodes(data=True) if "modified_importance" in data])
        
        return stats
    
    def export_path_comparison(self, output_path: Optional[str] = None) -> str:
        """
        Export a comparison of all generated paths.
        
        Args:
            output_path: Path to save the comparison
            
        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "path_comparison.txt")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write("Path Comparison\n")
            f.write("==============\n\n")
            
            for i, path in enumerate(self.paths):
                temperature = path.get("temperature", "unknown")
                f.write(f"Path {i} (Temperature: {temperature}):\n")
                f.write("-" * 50 + "\n")
                f.write(path.get("generated_text", "[No text]"))
                f.write("\n\n")
            
            f.write("\nDivergence Points\n")
            f.write("===============\n\n")
            
            for i, dp in enumerate(self.divergence_points):
                f.write(f"Divergence Point {i}:\n")
                f.write(f"Position: {dp['position']}\n")
                f.write(f"Paths: {dp['path_indices']}\n")
                f.write(f"KL Divergence: {dp['kl_divergence']:.4f}\n")
                
                if "tokens" in dp:
                    f.write(f"Token in Path {dp['path_indices'][0]}: {dp['tokens']['path1']['token']} (p={dp['tokens']['path1']['probability']:.4f})\n")
                    f.write(f"Token in Path {dp['path_indices'][1]}: {dp['tokens']['path2']['token']} (p={dp['tokens']['path2']['probability']:.4f})\n")
                
                if "context_window" in dp:
                    f.write("Context in Path 1: ")
                    f.write(" ".join(dp["context_window"].get("path1", [])))
                    f.write("\n")
                    
                    f.write("Context in Path 2: ")
                    f.write(" ".join(dp["context_window"].get("path2", [])))
                    f.write("\n")
                
                f.write("\n")
        
        logger.info(f"Exported path comparison to {output_path}")
        return output_path
    
    def export_tree_visualization(self, output_path: Optional[str] = None) -> str:
        """
        Export the tree visualization data.
        
        Args:
            output_path: Path to save the visualization data
            
        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "tree_visualization.json")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Convert to visualization format
            viz_data = self.to_dependentree_format()
            
            # Save to file
            with open(output_path, "w") as f:
                json.dump(viz_data, f, indent=2)
                
        except RecursionError:
            # Fallback: create simplified tree to avoid recursion
            logger.warning("Recursion error detected. Creating simplified visualization.")
            
            # Create a simplified tree
            simple_tree = {
                "text": "Root",
                "children": []
            }
            
            # Add top-level nodes only (first few positions)
            positions = set(node.position for node in self.tree_builder.nodes.values())
            for pos in sorted(positions)[:10]:  # First 10 positions only
                pos_nodes = [n for n in self.tree_builder.nodes.values() if n.position == pos]
                for node in pos_nodes[:5]:  # Top 5 nodes per position
                    simple_tree["children"].append({
                        "id": node.id,
                        "text": node.token,
                        "importance": node.importance_score,
                        "is_divergence": node.is_divergence_point,
                    })
            
            # Save simplified tree
            with open(output_path, "w") as f:
                json.dump(simple_tree, f, indent=2)
        
        logger.info(f"Exported tree visualization to {output_path}")
        return output_path