"""
Main entry point for the XAIR system.
Integrates CGRT, Counterfactual, and Knowledge Graph components.
"""

import os
import logging
import argparse
from typing import Dict, List, Any, Optional

from backend.models.llm_interface import LlamaInterface, GenerationConfig
from backend.cgrt.cgrt_main import CGRT
from backend.counterfactual.counterfactual_main import Counterfactual
from backend.knowledge_graph.kg_main import KnowledgeGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description="XAIR: Explainable AI Reasoning")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", 
                        help="Model name or path")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device to use (cpu, cuda, mps, or auto)")
    
    # Generation settings
    parser.add_argument("--max-tokens", type=int, default=256, 
                        help="Maximum tokens to generate")
    parser.add_argument("--temperatures", type=str, default="0.2,0.7,1.0", 
                        help="Comma-separated temperatures for generation")
    parser.add_argument("--paths-per-temp", type=int, default=1, 
                        help="Paths to generate per temperature")
    
    # Counterfactual settings
    parser.add_argument("--counterfactual-tokens", type=int, default=5, 
                        help="Top-k tokens for counterfactual generation")
    parser.add_argument("--attention-threshold", type=float, default=0.3, 
                        help="Minimum attention threshold for counterfactuals")
    parser.add_argument("--max-counterfactuals", type=int, default=20, 
                        help="Maximum counterfactuals to generate")
    
    # Knowledge Graph settings
    parser.add_argument("--kg-use-local-model", action="store_true", 
                        help="Use local sentence transformer model")
    parser.add_argument("--kg-similarity-threshold", type=float, default=0.6, 
                        help="Minimum similarity threshold for KG entity mapping")
    parser.add_argument("--kg-skip", action="store_true",
                        help="Skip Knowledge Graph processing (useful for slower machines)")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default="output", 
                        help="Output directory")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    
    return parser

def main():
    """Main entry point."""
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Convert comma-separated temperatures to list
    temperatures = [float(t) for t in args.temperatures.split(",")]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    logger.info("Initializing LLM interface...")
    llm = LlamaInterface(
        model_name_or_path=args.model,
        device=args.device,
        use_fp16=True,
        verbose=args.verbose
    )
    
    logger.info("Initializing CGRT...")
    cgrt = CGRT(
        model_name_or_path=args.model,
        device=args.device,
        temperatures=temperatures,
        paths_per_temp=args.paths_per_temp,
        max_new_tokens=args.max_tokens,
        output_dir=os.path.join(args.output_dir, "cgrt"),
        verbose=args.verbose
    )
    
    logger.info("Initializing Counterfactual...")
    counterfactual = Counterfactual(
        top_k_tokens=args.counterfactual_tokens,
        min_attention_threshold=args.attention_threshold,
        max_total_candidates=args.max_counterfactuals,
        output_dir=os.path.join(args.output_dir, "counterfactual"),
        verbose=args.verbose
    )
    
    # Initialize Knowledge Graph component
    if not args.kg_skip:
        logger.info("Initializing Knowledge Graph...")
        knowledge_graph = KnowledgeGraph(
            use_local_model=args.kg_use_local_model,
            min_similarity_threshold=args.kg_similarity_threshold,
            output_dir=os.path.join(args.output_dir, "knowledge_graph"),
            verbose=args.verbose
        )
    
    # Interactive prompt loop
    while True:
        try:
            prompt = input("\nEnter your prompt (or 'quit' to exit): ")
            if prompt.lower() in ("quit", "exit", "q"):
                break
                
            # Process with CGRT
            logger.info("Processing with CGRT...")
            tree = cgrt.process_input(prompt)
            
            # Print reasoning paths
            paths = cgrt.get_paths_text()
            logger.info(f"Generated {len(paths)} reasoning paths")
            
            for i, path_text in enumerate(paths):
                print(f"\nPath {i+1}:")
                print("=" * 50)
                print(path_text)
            
            # Process with Counterfactual
            logger.info("Generating counterfactuals...")
            counterfactuals = counterfactual.generate_counterfactuals(
                cgrt.tree_builder,
                llm,
                prompt,
                cgrt.paths
            )
            
            # Evaluate counterfactuals
            if paths:
                original_output = paths[0]  # Use first path as the original output
                metrics = counterfactual.evaluate_counterfactuals(original_output)
                
                # Get top counterfactuals
                top_cfs = counterfactual.get_top_counterfactuals(3)
                
                print("\nTop Counterfactuals:")
                print("=" * 50)
                for i, cf in enumerate(top_cfs):
                    print(f"{i+1}. Original: '{cf.original_token}' → Alternative: '{cf.alternative_token}'")
                    print(f"   Position: {cf.position}, Impact: {cf.impact_score:.2f}, Flipped: {'Yes' if cf.flipped_output else 'No'}")
                
                # Export comparison
                comparison_path = counterfactual.export_counterfactual_comparison()
                logger.info(f"Exported counterfactual comparison to {comparison_path}")
                
                # Print CFR
                cfr = counterfactual.generator.calculate_cfr()
                print(f"\nCounterfactual Flip Rate (CFR): {cfr:.2f}")
                
                # Process with Knowledge Graph (if not skipped)
                if not args.kg_skip:
                    logger.info("Processing with Knowledge Graph...")
                    try:
                        # Map nodes to entities and validate reasoning
                        entity_mapping, validation_results = knowledge_graph.process_reasoning_tree(
                            cgrt.tree_builder,
                            cgrt.paths
                        )
                        
                        # Get validation summary
                        summary = knowledge_graph.get_validation_summary()
                        
                        # Print validation summary
                        print("\nKnowledge Graph Validation:")
                        print("=" * 50)
                        print(f"Average trustworthiness score: {summary['average_trustworthiness']:.2f}")
                        print(f"Supported statements: {summary['total_supported_statements']}")
                        print(f"Contradicted statements: {summary['total_contradicted_statements']}")
                        print(f"Unverified statements: {summary['total_unverified_statements']}")
                        
                        # Print most trustworthy path
                        most_trustworthy_path_id = summary['most_trustworthy_path']['path_id']
                        most_trustworthy_score = summary['most_trustworthy_path']['score']
                        print(f"Most trustworthy path: Path {most_trustworthy_path_id} (Score: {most_trustworthy_score:.2f})")
                        
                        # Export detailed report
                        report_path = os.path.join(args.output_dir, "knowledge_graph", "validation_report.txt")
                        knowledge_graph.validator.export_validation_report(report_path)
                        logger.info(f"Exported knowledge graph validation report to {report_path}")
                        
                    except Exception as e:
                        logger.error(f"Error in Knowledge Graph processing: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Save states
                cgrt.tree_builder.save_tree(os.path.join(args.output_dir, "reasoning_tree.json"))
                counterfactual.save_state()
                
                logger.info("Processing complete! Results saved to output directory.")
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()