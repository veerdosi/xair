"""
Main Counterfactual module for the XAIR system.
Integrates counterfactual generation and evaluation.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Any, Optional, Set
import time

from backend.counterfactual.generator import CounterfactualGenerator, CounterfactualCandidate
from backend.counterfactual.evaluator import CounterfactualEvaluator
from backend.counterfactual.ranker import CounterfactualRanker
from backend.utils.progress_monitor import ProgressMonitor, Stage

logger = logging.getLogger(__name__)

class Counterfactual:
    """
    Main Counterfactual class that integrates generation and evaluation.
    """

    def __init__(
        self,
        top_k_tokens: int = 5,
        min_attention_threshold: float = 0.3,
        use_semantic_analysis: bool = True,
        max_candidates_per_node: int = 3,
        max_total_candidates: int = 20,
        output_dir: str = "output",
        verbose: bool = False
    ):
        """
        Initialize the Counterfactual system.

        Args:
            top_k_tokens: Number of top tokens to consider for substitution
            min_attention_threshold: Minimum attention score for tokens to consider
            use_semantic_analysis: Whether to use semantic analysis for evaluation
            max_candidates_per_node: Maximum number of candidates per node
            max_total_candidates: Maximum total candidates to generate
            output_dir: Directory to save outputs
            verbose: Whether to log detailed information
        """
        self.output_dir = output_dir
        self.verbose = verbose

        if self.verbose:
            logging.basicConfig(level=logging.INFO)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        logger.info("Initializing Counterfactual Generator...")
        self.generator = CounterfactualGenerator(
            top_k_tokens=top_k_tokens,
            min_attention_threshold=min_attention_threshold,
            max_candidates_per_node=max_candidates_per_node,
            max_total_candidates=max_total_candidates,
            verbose=verbose
        )

        logger.info("Initializing Counterfactual Evaluator...")
        self.evaluator = CounterfactualEvaluator(
            use_semantic_analysis=use_semantic_analysis,
            verbose=verbose
        )

        logger.info("Initializing Counterfactual Ranker...")
        self.ranker = CounterfactualRanker(
            impact_weight=1.0,
            flip_weight=1.5,
            position_weight=0.5,
            attention_weight=0.8,
            verbose=verbose
        )

        # State variables
        self.counterfactuals = []  # Generated counterfactuals
        self.metrics = None  # Evaluation metrics

        # Create progress monitor
        self.progress_monitor = ProgressMonitor(verbose=verbose)

        logger.info("Counterfactual system initialized successfully")

    def generate_counterfactuals(
        self,
        tree_builder,
        llm_interface,
        prompt: str,
        paths: List[Dict[str, Any]] = None,
        focus_nodes: List[str] = None,
        save_results: bool = True
    ) -> List[CounterfactualCandidate]:
        """
        Generate counterfactual alternatives.

        Args:
            tree_builder: CGRTBuilder instance
            llm_interface: LlamaInterface instance
            prompt: Original prompt
            paths: Generated paths (optional)
            focus_nodes: List of node IDs to focus on (optional)
            save_results: Whether to save results to disk

        Returns:
            List of counterfactual candidates
        """
        logger.info("Generating counterfactuals...")
        self.progress_monitor.start_stage(Stage.COUNTERFACTUAL_GENERATION)

        start_time = time.time()
        total_nodes = len(tree_builder.nodes) if tree_builder and hasattr(tree_builder, 'nodes') else 0

        # Generate counterfactuals
        self.counterfactuals = self.generator.generate_counterfactuals(
            tree_builder,
            llm_interface,
            prompt,
            paths,
            focus_nodes
        )

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Generated {len(self.counterfactuals)} counterfactuals in {duration:.2f}s")
        self.progress_monitor.complete_stage(Stage.COUNTERFACTUAL_GENERATION)

        # Save the results if requested
        if save_results:
            output_path = os.path.join(self.output_dir, "counterfactuals.json")
            self.generator.save_counterfactuals(output_path)

        return self.counterfactuals

    def evaluate_counterfactuals(
        self,
        original_output: str,
        save_results: bool = True
    ):
        """
        Evaluate the generated counterfactuals.

        Args:
            original_output: Original model output
            save_results: Whether to save results to disk

        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating counterfactuals...")
        self.progress_monitor.start_stage(Stage.COUNTERFACTUAL_EVALUATION)

        # Track progress
        start_time = time.time()
        total_cf = len(self.counterfactuals)

        # Evaluate counterfactuals
        self.metrics = self.evaluator.evaluate_counterfactuals(
            self.counterfactuals,
            original_output
        )

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Counterfactual evaluation complete in {duration:.2f}s")
        self.progress_monitor.complete_stage(Stage.COUNTERFACTUAL_EVALUATION)

        # Save the results if requested
        if save_results:
            output_path = os.path.join(self.output_dir, "counterfactual_evaluation.json")
            self.evaluator.save_evaluation(output_path)

        return self.metrics

    def get_top_counterfactuals(
        self,
        n: int = 5,
        require_flip: bool = False
    ) -> List[CounterfactualCandidate]:
        """
        Get the top N counterfactuals.

        Args:
            n: Number of counterfactuals to return
            require_flip: Whether to only include counterfactuals that flip the output

        Returns:
            List of top counterfactuals
        """
        # Use the ranker for more sophisticated `ranking`
        result = self.ranker.rank_counterfactuals(self.counterfactuals)
        ranked_counterfactuals = result.ranked_counterfactuals

        # Filter if required
        if require_flip:
            ranked_counterfactuals = [cf for cf in ranked_counterfactuals if cf.flipped_output]

        # Return top N
        return ranked_counterfactuals[:n]

    def get_counterfactual_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the counterfactual analysis.

        Returns:
            Dictionary with counterfactual summary
        """
        if not self.counterfactuals:
            return {"count": 0, "cfr": 0.0, "counterfactuals": []}

        # Calculate CFR
        cfr = self.generator.calculate_cfr()

        # Use the ranker to get a more comprehensive report
        ranking_result = self.ranker.rank_counterfactuals(self.counterfactuals)

        # Get advanced analytics from the ranker
        critical_positions = self.ranker.get_critical_positions(self.counterfactuals)

        # Create summary
        summary = {
            "count": len(self.counterfactuals),
            "cfr": cfr,
            "avg_impact": self.metrics.avg_impact if self.metrics else 0.0,
            "max_impact": self.metrics.max_impact if self.metrics else 0.0,
            "min_token_changes": self.metrics.min_token_changes if self.metrics else 0,
            "critical_positions": critical_positions[:3],  # Top 3 critical positions
            "impact_distribution": ranking_result.impact_distribution,
            "counterfactuals": []
        }

        # Add top counterfactuals
        top_cfs = ranking_result.ranked_counterfactuals[:5]  # Use already ranked counterfactuals
        for cf in top_cfs:
            summary["counterfactuals"].append({
                "position": cf.position,
                "original_token": cf.original_token,
                "alternative_token": cf.alternative_token,
                "impact_score": cf.impact_score,
                "flipped_output": cf.flipped_output
            })

        return summary

    def export_counterfactual_comparison(self, output_path: Optional[str] = None) -> str:
        """
        Export a comparison of original vs counterfactual outputs.

        Args:
            output_path: Path to save the comparison

        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "counterfactual_comparison.txt")

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Get advanced summary from ranker
        report = self.ranker.generate_summary_report(self.counterfactuals)

        with open(output_path, "w") as f:
            f.write("Counterfactual Comparison\n")
            f.write("========================\n\n")

            f.write(f"Total Counterfactuals: {len(self.counterfactuals)}\n")
            f.write(f"Counterfactual Flip Rate (CFR): {self.generator.calculate_cfr():.2f}\n\n")

            # Add impact distribution
            f.write("Impact Distribution:\n")
            for level, percentage in report.get("impact_distribution", {}).items():
                f.write(f"  {level.replace('_', ' ').title()}: {percentage*100:.1f}%\n")
            f.write("\n")

            # Add critical positions information
            if "critical_positions" in report.get("summary", {}):
                f.write("Critical Positions:\n")
                for pos in report["summary"]["critical_positions"]:
                    f.write(f"  Position {pos}\n")
                f.write("\n")

            # Get top counterfactuals from ranking results
            top_cfs = self.ranker.rank_counterfactuals(self.counterfactuals).ranked_counterfactuals[:5]

            for i, cf in enumerate(top_cfs):
                f.write(f"Counterfactual {i+1}:\n")
                f.write("----------------\n")
                f.write(f"Position: {cf.position}\n")
                f.write(f"Original Token: '{cf.original_token}'\n")
                f.write(f"Alternative Token: '{cf.alternative_token}'\n")
                f.write(f"Impact Score: {cf.impact_score:.2f}\n")
                f.write(f"Flipped Output: {'Yes' if cf.flipped_output else 'No'}\n\n")

                if cf.modified_path:
                    # Show a snippet of the modified path (truncated)
                    max_snippet_length = 300
                    snippet = cf.modified_path
                    if len(snippet) > max_snippet_length:
                        snippet = snippet[:max_snippet_length] + "..."

                    f.write("Output Snippet:\n")
                    f.write(f"{snippet}\n\n")

            f.write("\nMetrics:\n")
            f.write("--------\n")
            if self.metrics:
                f.write(f"Average Impact: {self.metrics.avg_impact:.2f}\n")
                f.write(f"Max Impact: {self.metrics.max_impact:.2f}\n")
                f.write(f"Semantic Coherence: {self.metrics.semantic_coherence:.2f}\n")
                f.write(f"Min Token Changes: {self.metrics.min_token_changes}\n")

        logger.info(f"Exported counterfactual comparison to {output_path}")
        return output_path

    def save_state(self, output_path: Optional[str] = None):
        """
        Save the current state of the counterfactual analysis.

        Args:
            output_path: Path to save the state
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "counterfactual_state.json")

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save state
        state = {
            "counterfactuals": [cf.to_dict() for cf in self.counterfactuals],
            "metrics": {
                "cfr": self.metrics.cfr if self.metrics else 0.0,
                "avg_impact": self.metrics.avg_impact if self.metrics else 0.0,
                "max_impact": self.metrics.max_impact if self.metrics else 0.0,
                "semantic_coherence": self.metrics.semantic_coherence if self.metrics else 0.0,
                "min_token_changes": self.metrics.min_token_changes if self.metrics else 0
            },
            "timestamp": time.time()
        }

        with open(output_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved counterfactual state to {output_path}")

    def load_state(self, input_path: str):
        """
        Load state from a file.

        Args:
            input_path: Path to load the state from
        """
        # Check if the file exists
        if not os.path.exists(input_path):
            logger.error(f"State file {input_path} not found")
            return

        # Load the data
        with open(input_path, "r") as f:
            state = json.load(f)

        # Load counterfactuals
        self.counterfactuals = []
        for cf_data in state.get("counterfactuals", []):
            try:
                candidate = CounterfactualCandidate(
                    position=cf_data["position"],
                    original_token=cf_data["original_token"],
                    original_token_id=cf_data["original_token_id"],
                    alternative_token=cf_data["alternative_token"],
                    alternative_token_id=cf_data["alternative_token_id"],
                    original_probability=cf_data["original_probability"],
                    alternative_probability=cf_data["alternative_probability"],
                    attention_score=cf_data["attention_score"],
                    impact_score=cf_data["impact_score"],
                    semantic_similarity=cf_data["semantic_similarity"],
                    flipped_output=cf_data["flipped_output"],
                    modified_path=cf_data["modified_path"],
                    timestamp=cf_data["timestamp"]
                )
                self.counterfactuals.append(candidate)
            except KeyError as e:
                logger.error(f"Missing key in counterfactual data: {e}")

        # Load metrics
        if "metrics" in state:
            from backend.counterfactual.evaluator import EvaluationMetrics
            metrics_data = state["metrics"]
            self.metrics = EvaluationMetrics(
                cfr=metrics_data.get("cfr", 0.0),
                avg_impact=metrics_data.get("avg_impact", 0.0),
                max_impact=metrics_data.get("max_impact", 0.0),
                semantic_coherence=metrics_data.get("semantic_coherence", 0.0),
                min_token_changes=metrics_data.get("min_token_changes", 0)
            )

        logger.info(f"Loaded counterfactual state from {input_path}")
