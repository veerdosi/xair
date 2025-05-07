"""
Main entry point for the XAIR system.
Integrates CGRT, Counterfactual, and Knowledge Graph components.
"""

import os
import logging
import argparse
import json
from typing import Dict, List, Any, Optional
import traceback

# Import components
from backend.models.llm_interface import LlamaInterface, GenerationConfig
from backend.cgrt.cgrt_main import CGRT, get_performance_preset
from backend.counterfactual.counterfactual_main import Counterfactual
from backend.knowledge_graph.kg_main import KnowledgeGraph

# Import utilities
from backend.utils.config import XAIRConfig
from backend.utils.logging_utils import setup_logger, TimingLogger
from backend.utils.device_utils import get_optimal_device, optimize_for_device
from backend.utils.error_utils import handle_exceptions, XAIRError
from backend.utils.viz_utils import export_visualization_report

# Setup logger
logger = setup_logger(name="xair", level=logging.INFO, use_rich=True)
timer = TimingLogger(logger=logger)

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

    # Performance settings
    parser.add_argument("--performance", type=str, choices=["max_speed", "balanced", "max_quality"],
                        default="balanced", help="Performance preset (max_speed, balanced, max_quality)")
    parser.add_argument("--fast-mode", action="store_true",
                        help="Skip hidden states and attention collection for faster generation")
    parser.add_argument("--fast-init", action="store_true",
                        help="Skip non-essential initialization steps for faster startup")

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

    # Config settings
    parser.add_argument("--config", type=str,
                        help="Path to configuration file")
    parser.add_argument("--save-config", type=str,
                        help="Save configuration to the specified file path")

    # Visualization settings
    parser.add_argument("--generate-visualizations", action="store_true",
                        help="Generate visualizations for the results")

    return parser

@handle_exceptions(reraise=True)
def initialize_components(config: XAIRConfig):
    """
    Initialize system components based on configuration.

    Args:
        config: System configuration

    Returns:
        Tuple of (llm, cgrt, counterfactual, knowledge_graph)
    """
    # Get optimal device and model settings
    device, dtype = get_optimal_device(config.device)
    device_settings = optimize_for_device(device)

    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Apply performance preset if specified
    if hasattr(config, 'performance') and config.performance:
        logger.info(f"Using performance preset: {config.performance}")
        preset = get_performance_preset(config.performance)
        # Update config with preset values, preserving explicit settings
        if not hasattr(config, 'fast_mode') or config.fast_mode is None:
            config.fast_mode = preset["fast_mode"]
        if not hasattr(config, 'fast_init') or config.fast_init is None:
            config.fast_init = preset["fast_init"]
        if preset["temperatures"] and len(preset["temperatures"]) > 0:
            config.cgrt.temperatures = preset["temperatures"]
        if preset["paths_per_temp"]:
            config.cgrt.paths_per_temp = preset["paths_per_temp"]
    else:
        # Set defaults if not using a preset
        if not hasattr(config, 'fast_mode'):
            config.fast_mode = False
        if not hasattr(config, 'fast_init'):
            config.fast_init = False

    logger.info(f"Fast mode: {'Enabled' if config.fast_mode else 'Disabled'}")
    logger.info(f"Fast init: {'Enabled' if config.fast_init else 'Disabled'}")

    # Initialize LLM interface
    timer.start("init_llm")
    logger.info("Initializing LLM interface...")
    llm = LlamaInterface(
        model_name_or_path=config.model_name_or_path,
        device=device,
        use_fp16=device_settings["use_fp16"],
        use_bettertransformer=device_settings["use_bettertransformer"],
        verbose=config.verbose,
        fast_init=config.fast_init
    )
    timer.stop("init_llm")

    # Initialize CGRT
    timer.start("init_cgrt")
    logger.info("Initializing CGRT...")
    cgrt = CGRT(
        model_name_or_path=config.model_name_or_path,
        device=device,
        temperatures=config.cgrt.temperatures,
        paths_per_temp=config.cgrt.paths_per_temp,
        max_new_tokens=config.max_tokens,
        output_dir=config.cgrt.output_dir,
        verbose=config.verbose,
        fast_mode=config.fast_mode,
        fast_init=config.fast_init
    )
    timer.stop("init_cgrt")

    # Initialize Counterfactual
    timer.start("init_counterfactual")
    logger.info("Initializing Counterfactual...")
    counterfactual = Counterfactual(
        top_k_tokens=config.counterfactual.top_k_tokens,
        min_attention_threshold=config.counterfactual.min_attention_threshold,
        max_total_candidates=config.counterfactual.max_total_candidates,
        output_dir=config.counterfactual.output_dir,
        verbose=config.verbose
    )
    timer.stop("init_counterfactual")

    # Initialize Knowledge Graph (if not skipped)
    knowledge_graph = None
    if not config.skip_kg:
        timer.start("init_kg")
        logger.info("Initializing Knowledge Graph...")
        knowledge_graph = KnowledgeGraph(
            use_local_model=config.knowledge_graph.use_local_model,
            min_similarity_threshold=config.knowledge_graph.min_similarity_threshold,
            output_dir=config.knowledge_graph.output_dir,
            verbose=config.verbose
        )
        timer.stop("init_kg")

    return llm, cgrt, counterfactual, knowledge_graph

@handle_exceptions(reraise=False)
def process_prompt(
    prompt: str,
    llm: LlamaInterface,
    cgrt: CGRT,
    counterfactual: Counterfactual,
    knowledge_graph: Optional[KnowledgeGraph],
    config: XAIRConfig,
    generate_visualizations: bool = False
):
    """
    Process a prompt through the XAIR pipeline.

    Args:
        prompt: Input prompt
        llm: LLM interface
        cgrt: CGRT component
        counterfactual: Counterfactual component
        knowledge_graph: Knowledge Graph component (or None if skipped)
        config: System configuration
        generate_visualizations: Whether to generate visualizations
    """
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # 1. Process with CGRT
    timer.start("cgrt_processing")
    logger.info("Processing with CGRT...")

    # Set up generation config with appropriate settings
    gen_config = GenerationConfig(
        max_new_tokens=config.max_tokens,
        output_hidden_states=not getattr(config, 'fast_mode', False),
        output_attentions=not getattr(config, 'fast_mode', False)
    )

    tree = cgrt.process_input(
        prompt,
        generation_config=gen_config
    )
    timer.stop("cgrt_processing")

    # Print reasoning paths
    paths = cgrt.get_paths_text()
    logger.info(f"Generated {len(paths)} reasoning paths")

    for i, path_text in enumerate(paths):
        print(f"\nPath {i+1}:")
        print("=" * 50)
        print(path_text)

    # 2. Process with Counterfactual
    timer.start("counterfactual_processing")
    logger.info("Generating counterfactuals...")
    counterfactuals = counterfactual.generate_counterfactuals(
        cgrt.tree_builder,
        llm,
        prompt,
        cgrt.paths
    )
    timer.stop("counterfactual_processing")

    validation_results = None

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

        # 3. Process with Knowledge Graph (if not skipped)
        if knowledge_graph:
            timer.start("kg_processing")
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
                report_path = os.path.join(config.output_dir, "knowledge_graph", "validation_report.txt")
                knowledge_graph.validator.export_validation_report(report_path)
                logger.info(f"Exported knowledge graph validation report to {report_path}")

            except Exception as e:
                logger.error(f"Error in Knowledge Graph processing: {e}")
                logger.debug(traceback.format_exc())
            finally:
                timer.stop("kg_processing")

        # Save states
        cgrt.tree_builder.save_tree(os.path.join(config.output_dir, "reasoning_tree.json"))
        counterfactual.save_state()

        # Generate visualizations if requested
        if generate_visualizations:
            timer.start("visualizations")
            try:
                logger.info("Generating visualizations...")
                viz_dir = export_visualization_report(
                    cgrt.tree_builder,
                    counterfactuals,
                    validation_results,
                    config.output_dir
                )
                logger.info(f"Visualizations saved to {viz_dir}")
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")
                logger.debug(traceback.format_exc())
            finally:
                timer.stop("visualizations")

        # Print timing summary
        print("\nPerformance Timing:")
        print("=" * 50)
        print(timer.summary())

        logger.info("Processing complete! Results saved to output directory.")
    else:
        logger.warning("No reasoning paths generated. Check model configuration.")

def main():
    """Main entry point."""
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()

    # Create configuration
    config = None

    # If config file is specified, load it
    if args.config:
        try:
            logger.info(f"Loading configuration from {args.config}...")
            config = XAIRConfig.load(args.config)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return
    else:
        # Create config from args
        logger.info("Creating configuration from command line arguments...")
        config = XAIRConfig.from_args(args)

    # Save config if requested
    if args.save_config:
        try:
            logger.info(f"Saving configuration to {args.save_config}...")
            config.save(args.save_config)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize components
    try:
        llm, cgrt, counterfactual, knowledge_graph = initialize_components(config)
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        logger.debug(traceback.format_exc())
        return

    # Interactive prompt loop
    while True:
        try:
            prompt = input("\nEnter your prompt (or 'quit' to exit): ")
            if prompt.lower() in ("quit", "exit", "q"):
                break

            process_prompt(
                prompt,
                llm,
                cgrt,
                counterfactual,
                knowledge_graph,
                config,
                args.generate_visualizations
            )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.debug(traceback.format_exc())

if __name__ == "__main__":
    main()
