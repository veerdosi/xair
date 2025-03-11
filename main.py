import asyncio
import numpy as np
from ver3_0.response_llm import LLMInterface
from ver3_0.cgrt import CGRTGenerator, CrossGeneration, GenerationPath
from ver3_0.cgrt_tree import CGRTTree, CGRTNode
from ver3_0.counterfactual import Counterfactual, CounterfactualGenerator, VCNet
from ver3_0.counterfactual_integrator import CounterfactualIntegrator, CounterfactualOverlay
from ver3_0.impact_analyzer import ImpactAnalyzer, ImpactScore

async def main():
    # Initialize LLM interface
    async with LLMInterface(api_key="your_api_key_here") as llm:
        # Example prompt
        prompt = ""
        
        # Initialize components
        tree_generator = CGRTGenerator(llm)
        cf_generator = CounterfactualGenerator(llm, tree_generator)
        integrator = CounterfactualIntegrator(tree_generator, cf_generator)
        analyzer = ImpactAnalyzer(llm, tree_generator, cf_generator, integrator)

        # Step 1: Generate base reasoning tree
        print("Generating reasoning tree...")
        cross_gen = await tree_generator.generate_cross_paths(prompt)
        reasoning_tree = CGRTTree().build_from_generations(cross_gen)

        # Step 2: Generate counterfactuals
        print("\nGenerating counterfactuals...")
        counterfactuals = await cf_generator.generate_counterfactuals(reasoning_tree)

        # Step 3: Integrate counterfactuals
        print("\nIntegrating counterfactuals...")
        integrated_tree = await integrator.integrate_counterfactuals(reasoning_tree, counterfactuals)

        # Step 4: Analyze impacts
        print("\nAnalyzing impacts...")
        impact_scores = await analyzer.analyze_impacts(integrated_tree, counterfactuals)

        # Display results
        print("\n=== Top Impacts ===")
        for score in impact_scores[:3]:
            print(f"Counterfactual {score.counterfactual_id}")
            print(f"Composite Score: {score.composite_score:.2f}")
            print(f"Affected Nodes: {len(score.affected_nodes)}\n")

if __name__ == "__main__":
    asyncio.run(main())