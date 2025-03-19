import asyncio
import os
import logging
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Import XAIR modules
from backend.llm_interface import LLMInterface
from backend.cgrt import CGRTGenerator, CrossGeneration, GenerationPath
from backend.cgrt_tree import CGRTTree, CGRTNode
from backend.counterfactual import CounterfactualGenerator, Counterfactual
from backend.counterfactual_integrator import CounterfactualIntegrator
from backend.impact_analyzer import ImpactAnalyzer
from backend.error_handling import XAIRError, log_exception

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main(args):
    # Get API key from environment or command line
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or use --api-key")
    
    # Initialize LLM interface with specified model
    async with LLMInterface(
        api_key=api_key,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    ) as llm:
        logger.info(f"Initialized LLM interface with model: {args.model}")
        
        # Use the prompt provided as argument
        prompt = args.prompt
        logger.info(f"Using prompt: {prompt}")
        
        # Step 1: Generate cross-generational paths
        logger.info("Generating cross-generational paths...")
        cgrt_generator = CGRTGenerator(
            llm,
            temperature_range=(args.temperature * 0.7, args.temperature * 1.3),
            min_probability_threshold=args.min_probability
        )
        cross_gen = await cgrt_generator.generate_cross_paths(
            prompt, 
            max_tokens=args.max_tokens
        )
        
        # Step 2: Build CGRT tree from generations
        logger.info("Building reasoning tree...")
        tree_builder = CGRTTree(
            merge_threshold=0.8,
            importance_threshold=args.min_probability
        )
        reasoning_tree = tree_builder.build_from_generations(cross_gen)
        
        # Get tree statistics
        node_count = reasoning_tree.number_of_nodes()
        edge_count = reasoning_tree.number_of_edges()
        logger.info(f"Tree statistics: {node_count} nodes, {edge_count} edges")
        
        # Step 3: Generate counterfactuals
        logger.info("Generating counterfactuals...")
        cf_generator = CounterfactualGenerator(llm, cgrt_generator)
        counterfactuals = await cf_generator.generate_counterfactuals(reasoning_tree)
        logger.info(f"Generated {len(counterfactuals)} counterfactuals")
        
        # Step 4: Integrate counterfactuals into the tree
        logger.info("Integrating counterfactuals...")
        cf_integrator = CounterfactualIntegrator(cgrt_generator, cf_generator)
        integrated_tree = await cf_integrator.integrate_counterfactuals(reasoning_tree, counterfactuals)
        logger.info(f"Integrated tree has {integrated_tree.number_of_nodes()} nodes")
        
        # Step 5: Analyze impact of counterfactuals
        logger.info("Analyzing impact of counterfactuals...")
        impact_analyzer = ImpactAnalyzer(llm, cgrt_generator, cf_generator, cf_integrator)
        impact_scores = await impact_analyzer.analyze_impacts(integrated_tree, counterfactuals)
        logger.info(f"Generated {len(impact_scores)} impact scores")
        
        # Get ranked impacts
        ranked_impacts = impact_analyzer.get_ranked_impacts(impact_scores)
        logger.info("Top impacts:")
        for i, (cf_id, score) in enumerate(ranked_impacts[:5]):
            logger.info(f"  {i+1}. Counterfactual {cf_id}: {score:.3f}")
        
        # Get visualization data for integrated tree
        viz_data = cf_integrator.get_visualization_data(integrated_tree)
        logger.info(f"Visualization data: {len(viz_data['nodes'])} nodes, {len(viz_data['edges'])} edges")

        # Return success
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAIR - Explainable AI Reasoning")
    parser.add_argument("--prompt", type=str, default="Explain the concept of explainable AI", 
                        help="The prompt to generate a reasoning tree for")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (can also be set as OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum number of tokens to generate")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth of the reasoning tree")
    parser.add_parameter = parser.add_argument("--min-probability", type=float, default=0.1, 
                          help="Minimum probability threshold for node inclusion")
    
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(main(args))
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        exit(1)