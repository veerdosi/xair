"""
Example script demonstrating the Mac-friendly CGRT module.
"""

import os
import argparse
import logging
from backend.cgrt.cgrt_main import CGRT
from backend.models.llm_interface import GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CGRT Example')
    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-3.2-1B",
                        help='Path to the model or model identifier')
    parser.add_argument('--device', type=str, default="auto",
                        help='Device to load the model on (cpu, cuda, mps, or auto)')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.2, 0.7, 1.0],
                        help='Temperature values to use')
    parser.add_argument('--paths_per_temp', type=int, default=1,
                        help='Number of paths to generate per temperature')
    parser.add_argument('--output_dir', type=str, default="output",
                        help='Directory to save outputs')
    parser.add_argument('--load_4bit', action='store_true',
                        help='Ignored for Mac compatibility')
    parser.add_argument('--use_fp16', action='store_true', default=True,
                        help='Use half precision (float16)')
    parser.add_argument('--use_bettertransformer', action='store_true', default=True,
                        help='Use BetterTransformer optimization')
    parser.add_argument('--prompt', type=str,
                        default="Explain why climate change is happening and what solutions exist.",
                        help='Input prompt')
    parser.add_argument('--max_tokens', type=int, default=256,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize CGRT system
    logger.info("Initializing CGRT system...")
    cgrt = CGRT(
        model_name_or_path=args.model_path,
        device=args.device,
        load_in_4bit=False,  # Explicitly disable 4-bit quantization
        temperatures=args.temperatures,
        paths_per_temp=args.paths_per_temp,
        max_new_tokens=args.max_tokens,
        output_dir=args.output_dir,
        use_fp16=args.use_fp16,
        use_bettertransformer=args.use_bettertransformer,
        verbose=args.verbose
    )
    
    # Set up generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        output_hidden_states=True,
        output_attentions=True
    )
    
    # Process the input
    logger.info(f"Processing input: {args.prompt}")
    cgrt.process_input(args.prompt, gen_config, save_results=True)
    
    # Print path texts
    logger.info("Generated Paths:")
    for i, text in enumerate(cgrt.get_paths_text()):
        logger.info(f"Path {i}:")
        logger.info("-" * 50)
        logger.info(text[:100] + "..." if len(text) > 100 else text)
        logger.info("")
    
    # Print divergence summary
    divergence_summary = cgrt.get_divergence_summary()
    logger.info(f"Found {divergence_summary['count']} divergence points")
    
    for i, dp in enumerate(divergence_summary["points"][:3]):  # Show first 3
        logger.info(f"Divergence Point {i}:")
        logger.info(f"  Position: {dp['position']}")
        logger.info(f"  Path indices: {dp['path_indices']}")
        logger.info(f"  Tokens: {dp['tokens']['path1']} vs {dp['tokens']['path2']}")
    
    # Print tree stats
    tree_stats = cgrt.get_tree_stats()
    logger.info("Tree Statistics:")
    for key, value in tree_stats.items():
        logger.info(f"  {key}: {value}")
    
    # Export visualizations
    path_comp_file = cgrt.export_path_comparison()
    tree_viz_file = cgrt.export_tree_visualization()
    
    logger.info(f"Path comparison saved to: {path_comp_file}")
    logger.info(f"Tree visualization saved to: {tree_viz_file}")
    
    # Example of importance adjustment
    if tree_stats["node_count"] > 0:
        # Get a node ID from the tree
        node_id = list(cgrt.tree_builder.nodes.keys())[0]
        
        logger.info(f"Adjusting importance of node {node_id}")
        cgrt.adjust_node_importance(node_id, 0.9, "Example adjustment")
        
        # Save adjustment history
        cgrt.save_adjustment_history()
        
        # Compare original vs adjusted
        comparison = cgrt.compare_original_vs_adjusted()
        logger.info("Importance Adjustment Comparison:")
        logger.info(f"  Modified nodes: {len(comparison['modified_nodes'])}")
        logger.info(f"  Total nodes: {comparison['total_nodes']}")
        logger.info(f"  Average change: {comparison['average_change']:.4f}")
    
    logger.info("Example completed successfully")

if __name__ == "__main__":
    main()