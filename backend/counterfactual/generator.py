"""
Counterfactual Generator module for the XAIR system.
Identifies key tokens and generates counterfactual alternatives.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
import logging
from dataclasses import dataclass, field
import time
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

@dataclass
class CounterfactualCandidate:
    """Represents a counterfactual token substitution candidate."""
    position: int
    original_token: str
    original_token_id: int
    alternative_token: str
    alternative_token_id: int
    original_probability: float
    alternative_probability: float
    attention_score: float = 0.0
    impact_score: float = 0.0
    semantic_similarity: float = 0.0
    flipped_output: bool = False
    modified_path: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "position": self.position,
            "original_token": self.original_token,
            "original_token_id": self.original_token_id,
            "alternative_token": self.alternative_token,
            "alternative_token_id": self.alternative_token_id,
            "original_probability": self.original_probability,
            "alternative_probability": self.alternative_probability,
            "attention_score": self.attention_score,
            "impact_score": self.impact_score,
            "semantic_similarity": self.semantic_similarity,
            "flipped_output": self.flipped_output,
            "modified_path": self.modified_path,
            "timestamp": self.timestamp
        }

class CounterfactualGenerator:
    """Generates counterfactual alternatives for reasoning paths."""

    def __init__(
        self,
        top_k_tokens: int = 5,
        min_attention_threshold: float = 0.3,
        min_semantic_similarity: float = 0.5,
        max_candidates_per_node: int = 3,
        max_total_candidates: int = 20,
        filter_by_pos: bool = True,  # Filter substitutions by part of speech
        verbose: bool = False
    ):
        """
        Initialize the counterfactual generator.

        Args:
            top_k_tokens: Number of top tokens to consider for substitution
            min_attention_threshold: Minimum attention score for tokens to consider
            min_semantic_similarity: Minimum semantic similarity for alternatives
            max_candidates_per_node: Maximum number of candidates per node
            max_total_candidates: Maximum total candidates to generate
            filter_by_pos: Whether to filter substitutions by part of speech
            verbose: Whether to log detailed information
        """
        self.top_k_tokens = top_k_tokens
        self.min_attention_threshold = min_attention_threshold
        self.min_semantic_similarity = min_semantic_similarity
        self.max_candidates_per_node = max_candidates_per_node
        self.max_total_candidates = max_total_candidates
        self.filter_by_pos = filter_by_pos
        self.verbose = verbose

        if self.verbose:
            logging.basicConfig(level=logging.INFO)

        # Import spaCy if part-of-speech filtering is enabled
        if self.filter_by_pos:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for part-of-speech filtering")
            except (ImportError, OSError):
                logger.warning("Could not load spaCy model. Installing with: python -m spacy download en_core_web_sm")
                self.filter_by_pos = False

        # Generated counterfactuals
        self.counterfactuals = []

    def generate_counterfactuals(
        self,
        tree_builder,
        llm_interface,
        prompt: str,
        paths: List[Dict[str, Any]] = None,
        focus_nodes: List[str] = None
    ) -> List[CounterfactualCandidate]:
        """
        Generate counterfactual alternatives based on the reasoning tree.

        Args:
            tree_builder: CGRTBuilder instance
            llm_interface: LlamaInterface instance
            prompt: Original prompt
            paths: Generated paths (optional, will use tree_builder paths if not provided)
            focus_nodes: List of node IDs to focus on (optional)

        Returns:
            List of counterfactual candidates
        """
        logger.info("Generating counterfactual alternatives...")

        # Reset counterfactuals
        self.counterfactuals = []

        # Use existing paths if not provided
        if paths is None:
            paths = tree_builder.paths

        if not paths:
            logger.error("No paths available")
            return []

        # 1. Identify important nodes (based on attention and importance scores)
        target_nodes = self._identify_target_nodes(tree_builder, focus_nodes)

        if not target_nodes:
            logger.warning("No suitable target nodes found")
            return []

        logger.info(f"Identified {len(target_nodes)} target nodes for counterfactual generation")

        # 2. Generate token alternatives for each target node
        for node_id in target_nodes:
            node = tree_builder.nodes[node_id]

            # Skip if we already have enough counterfactuals
            if len(self.counterfactuals) >= self.max_total_candidates:
                break

            # Find the path that contains this node
            node_paths = []
            for path_idx in node.path_ids:
                if path_idx < len(paths):
                    node_paths.append(paths[path_idx])

            if not node_paths:
                continue

            # Use the first path containing this node
            path = node_paths[0]

            # Calculate the actual position in the sequence
            # We need to account for the prompt tokens
            input_length = path.get("input_ids", torch.tensor([[]])).shape[1]
            seq_position = input_length + node.position

            # Generate alternative tokens for this position
            alternatives = self._generate_alternatives(
                llm_interface,
                prompt,
                path,
                seq_position,
                node
            )

            logger.info(f"Generated {len(alternatives)} alternatives for node {node_id}")

            # Add to counterfactuals
            self.counterfactuals.extend(alternatives)

            # Limit the number of counterfactuals
            if len(self.counterfactuals) >= self.max_total_candidates:
                logger.info(f"Reached maximum number of counterfactuals ({self.max_total_candidates})")
                break

        # 3. Evaluate the impact of counterfactuals
        self._evaluate_counterfactuals(
            llm_interface,
            prompt,
            tree_builder
        )

        # 4. Sort by impact score
        self.counterfactuals.sort(key=lambda x: x.impact_score, reverse=True)

        return self.counterfactuals

    def _identify_target_nodes(
        self,
        tree_builder,
        focus_nodes: Optional[List[str]] = None
    ) -> List[str]:
        """
        Identify nodes to target for counterfactual generation.

        Args:
            tree_builder: CGRTBuilder instance
            focus_nodes: List of node IDs to focus on (optional)

        Returns:
            List of node IDs to target
        """
        if focus_nodes:
            # Filter out any nodes that don't exist
            return [node_id for node_id in focus_nodes if node_id in tree_builder.nodes]

        # Prioritize nodes with:
        # 1. High attention scores
        # 2. Divergence points
        # 3. High importance scores

        # Get all nodes
        node_scores = []
        for node_id, node in tree_builder.nodes.items():
            # Calculate combined score
            attention_weight = 2.0  # Weight attention more heavily
            importance_weight = 1.0
            divergence_weight = 1.5  # Weight divergence points

            combined_score = (
                node.attention_score * attention_weight +
                node.importance_score * importance_weight +
                (divergence_weight if node.is_divergence_point else 0.0)
            )

            # Add to list if above threshold
            if node.attention_score >= self.min_attention_threshold:
                node_scores.append((node_id, combined_score))

        # Sort by combined score
        node_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top nodes
        return [node_id for node_id, _ in node_scores[:self.max_total_candidates]]

    def _generate_alternatives(
        self,
        llm_interface,
        prompt: str,
        path: Dict[str, Any],
        position: int,
        node
    ) -> List[CounterfactualCandidate]:
        """
        Generate alternative tokens for a specific position.

        Args:
            llm_interface: LlamaInterface instance
            prompt: Original prompt
            path: Path containing the node
            position: Position in the sequence
            node: The target node

        Returns:
            List of counterfactual candidates
        """
        alternatives = []

        # Get token alternatives from the path
        token_alternatives = path.get("token_alternatives", [])

        if not token_alternatives or node.position >= len(token_alternatives):
            logger.warning(f"No token alternatives available for position {node.position}")
            return alternatives

        # Get alternatives at this position
        pos_alternatives = token_alternatives[node.position]

        # Skip if there are no alternatives
        if not pos_alternatives or len(pos_alternatives) <= 1:
            return alternatives

        # Get original token
        original_token = node.token
        original_token_id = node.token_id
        original_probability = pos_alternatives[0]["probability"] if pos_alternatives else 0.0

        # Generate alternatives by masked prediction
        alternatives_from_path = pos_alternatives[1:self.top_k_tokens+1]  # Skip the first one (original token)

        alternative_candidates = []

        # Process alternatives from path
        for alt in alternatives_from_path:
            alternative_token = alt["token"]
            alternative_token_id = alt["token_id"]
            alternative_probability = alt["probability"]

            # Check if this is a good alternative
            if self._is_valid_alternative(original_token, alternative_token):
                candidate = CounterfactualCandidate(
                    position=node.position,
                    original_token=original_token,
                    original_token_id=original_token_id,
                    alternative_token=alternative_token,
                    alternative_token_id=alternative_token_id,
                    original_probability=original_probability,
                    alternative_probability=alternative_probability,
                    attention_score=node.attention_score
                )

                alternative_candidates.append(candidate)

        # Sort by probability and take top candidates
        alternative_candidates.sort(key=lambda x: x.alternative_probability, reverse=True)
        return alternative_candidates[:self.max_candidates_per_node]

    def _is_valid_alternative(
        self,
        original_token: str,
        alternative_token: str
    ) -> bool:
        """
        Check if an alternative token is valid.

        Args:
            original_token: Original token
            alternative_token: Alternative token

        Returns:
            Whether the alternative is valid
        """
        # Skip if same token
        if original_token == alternative_token:
            return False

        # Skip very short tokens
        if len(alternative_token.strip()) <= 1:
            return False

        # Check part of speech if enabled
        if self.filter_by_pos:
            try:
                # Get POS tags
                original_doc = self.nlp(original_token)
                alternative_doc = self.nlp(alternative_token)

                if not original_doc or not alternative_doc:
                    return True  # In case of parsing issues, allow it

                # Get the main part of speech
                original_pos = original_doc[0].pos_ if len(original_doc) > 0 else ""
                alternative_pos = alternative_doc[0].pos_ if len(alternative_doc) > 0 else ""

                # Allow only if they have the same part of speech
                # This ensures grammatical coherence
                if original_pos and alternative_pos and original_pos != alternative_pos:
                    return False
            except Exception as e:
                logger.warning(f"Error in POS checking: {e}")
                # In case of errors, fallback to allow
                return True

        return True

    def _evaluate_counterfactuals(
        self,
        llm_interface,
        prompt: str,
        tree_builder
    ):
        """
        Evaluate the impact of counterfactual alternatives.

        Args:
            llm_interface: LlamaInterface instance
            prompt: Original prompt
            tree_builder: CGRTBuilder instance
        """
        logger.info(f"Evaluating {len(self.counterfactuals)} counterfactual candidates...")

        # Get the main original output
        original_outputs = [path.get("generated_text", "") for path in tree_builder.paths]
        original_output = original_outputs[0] if original_outputs else ""

        for i, candidate in enumerate(self.counterfactuals):
            logger.info(f"Evaluating counterfactual {i+1}/{len(self.counterfactuals)}")

            # Generate new output with the substitution
            try:
                modified_output = self._generate_with_substitution(
                    llm_interface,
                    prompt,
                    candidate
                )

                # Update the candidate
                candidate.modified_path = modified_output

                # Calculate impact score
                impact_score = self._calculate_impact_score(original_output, modified_output)
                candidate.impact_score = impact_score

                # Determine if this flipped the output
                candidate.flipped_output = self._check_if_flipped(original_output, modified_output)

                logger.info(f"Impact score: {impact_score}, Flipped: {candidate.flipped_output}")
            except Exception as e:
                logger.error(f"Error evaluating counterfactual: {e}")
                # Mark as low impact if there was an error
                candidate.impact_score = 0.0
                candidate.flipped_output = False

    def _generate_with_substitution(
        self,
        llm_interface,
        prompt: str,
        candidate: CounterfactualCandidate
    ) -> str:
        """
        Generate a new output by substituting a token.

        Args:
            llm_interface: LlamaInterface instance
            prompt: Original prompt
            candidate: Counterfactual candidate

        Returns:
            Modified output with token substitution
        """
        # For token substitution, we need to:
        # 1. Generate up to the position of substitution
        # 2. Replace the token
        # 3. Continue generation from that point

        try:
            # Tokenize the prompt
            inputs = llm_interface.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids

            # Position is relative to generation, not the full sequence
            # We need to adjust for the prompt length
            prompt_length = input_ids.shape[1]
            absolute_position = prompt_length + candidate.position

            # Step 1: Generate up to the position
            from transformers import GenerationConfig
            gen_config = GenerationConfig(
                max_new_tokens=candidate.position + 1,  # +1 to include the position we want to replace
                temperature=0.2,  # Low temperature for deterministic generation
                do_sample=False,  # No sampling for consistency
                output_scores=False,
                return_dict_in_generate=True
            )

            # Generate up to the position
            first_part = llm_interface.generate(prompt, gen_config, fast_mode=True)
            first_part_text = first_part["generated_text"]

            # Get the generated tokens up to the position (excluding the position itself)
            generated_ids = first_part["generated_ids"]
            generated_prefix = llm_interface.tokenizer.decode(
                generated_ids[0, prompt_length:absolute_position],
                skip_special_tokens=True
            )

            # Step 2: Create a new prompt with the prefix and alternative token
            new_prompt = prompt + generated_prefix + candidate.alternative_token

            # Step 3: Continue generation from this new prompt
            continuation_config = GenerationConfig(
                max_new_tokens=512,  # Generate a reasonably long continuation
                temperature=0.2,  # Keep temperature low for consistency
                do_sample=True
            )

            continuation = llm_interface.generate(new_prompt, continuation_config, fast_mode=True)
            continuation_text = continuation["generated_text"]

            # Combine for final result
            # Remove the prompt from the continuation to avoid duplication
            final_output = prompt + generated_prefix + candidate.alternative_token + continuation_text

            return final_output

        except Exception as e:
            logger.error(f"Error in token substitution: {e}")
            return f"Error in token substitution: {str(e)}"

    def _calculate_impact_score(
        self,
        original_output: str,
        modified_output: str
    ) -> float:
        """
        Calculate the impact score of a counterfactual.

        Args:
            original_output: Original generated output
            modified_output: Modified output with token substitution

        Returns:
            Impact score (0-1)
        """
        # Simple impact score based on textual difference
        # A more sophisticated approach would analyze semantic differences

        # If outputs are identical, no impact
        if original_output == modified_output:
            return 0.0

        try:
            # Use ROUGE score as a measure of similarity
            from rouge import Rouge
            rouge = Rouge()
            scores = rouge.get_scores(modified_output, original_output)

            # Get ROUGE-L score for longest common subsequence
            rouge_l = scores[0]["rouge-l"]["f"]

            # Convert similarity to difference (impact)
            impact = 1.0 - rouge_l

            return impact
        except ImportError:
            # Fallback if Rouge is not available
            logger.warning("Rouge not available, falling back to simple difference")

            # Use simple character-level difference as fallback
            total_chars = max(len(original_output), len(modified_output))
            if total_chars == 0:
                return 0.0

            # Calculate edit distance (Levenshtein distance)
            def levenshtein(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein(s2, s1)
                if len(s2) == 0:
                    return len(s1)

                previous_row = range(len(s2) + 1)
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row

                return previous_row[-1]

            distance = levenshtein(original_output, modified_output)
            impact = min(1.0, distance / total_chars)

            return impact
        except Exception as e:
            logger.error(f"Error calculating impact score: {e}")
            return 0.5  # Default to medium impact on error

    def _check_if_flipped(
        self,
        original_output: str,
        modified_output: str
    ) -> bool:
        """
        Check if the counterfactual flipped the output conclusion.

        Args:
            original_output: Original generated output
            modified_output: Modified output with token substitution

        Returns:
            Whether the output conclusion flipped
        """
        # A simple approach looks for sentiment changes or yes/no flips
        # A more sophisticated approach would use NLI or other semantic analysis

        # Check for simple yes/no flips
        yes_indicators = ["yes", "correct", "right", "true", "agree", "confirmed"]
        no_indicators = ["no", "incorrect", "wrong", "false", "disagree", "denied"]

        # Function to determine if text contains indicators
        def contains_indicators(text, indicators):
            text_lower = text.lower()

            # Check for indicators with boundaries
            for indicator in indicators:
                # Look for the indicator as a standalone word
                if f" {indicator} " in f" {text_lower} ":
                    return True

                # Also check if it ends with the indicator
                if text_lower.endswith(f" {indicator}"):
                    return True

                # Or starts with it
                if text_lower.startswith(f"{indicator} "):
                    return True

            return False

        # Check if original and modified outputs have opposite indicators
        original_has_yes = contains_indicators(original_output[-100:], yes_indicators)
        original_has_no = contains_indicators(original_output[-100:], no_indicators)
        modified_has_yes = contains_indicators(modified_output[-100:], yes_indicators)
        modified_has_no = contains_indicators(modified_output[-100:], no_indicators)

        # If they have opposite indicators, it's a flip
        if (original_has_yes and modified_has_no) or (original_has_no and modified_has_yes):
            return True

        # Also look for explicit contrasts at the end of the text
        contradiction_phrases = [
            "however,", "on the contrary,", "instead,", "but,", "contrary to",
            "in contrast,", "nonetheless,", "nevertheless,"
        ]

        # If the modified output ends with a contradiction, it might be a flip
        modified_end = modified_output[-200:].lower()
        for phrase in contradiction_phrases:
            if phrase in modified_end:
                return True

        # Calculate the difference in sentiment as another flip indicator
        try:
            from textblob import TextBlob

            original_sentiment = TextBlob(original_output).sentiment.polarity
            modified_sentiment = TextBlob(modified_output).sentiment.polarity

            # If sentiment flipped from positive to negative or vice versa
            if (original_sentiment > 0.2 and modified_sentiment < -0.2) or \
               (original_sentiment < -0.2 and modified_sentiment > 0.2):
                return True
        except ImportError:
            logger.warning("TextBlob not available for sentiment analysis")
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")

        # Default to not flipped
        return False

    def calculate_cfr(self) -> float:
        """
        Calculate the Counterfactual Flip Rate (CFR).

        Returns:
            CFR value (0-1)
        """
        if not self.counterfactuals:
            return 0.0

        # Count flipped counterfactuals
        flipped_count = sum(1 for cf in self.counterfactuals if cf.flipped_output)

        # Calculate CFR
        cfr = flipped_count / len(self.counterfactuals)

        return cfr

    def save_counterfactuals(self, output_path: str):
        """
        Save the counterfactuals to a file.

        Args:
            output_path: Path to save the counterfactuals
        """
        import os
        import json

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert to dictionary
        data = {
            "counterfactuals": [cf.to_dict() for cf in self.counterfactuals],
            "cfr": self.calculate_cfr(),
            "timestamp": time.time()
        }

        # Save to file
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved counterfactuals to {output_path}")

    def load_counterfactuals(self, input_path: str):
        """
        Load counterfactuals from a file.

        Args:
            input_path: Path to load the counterfactuals from
        """
        import os
        import json

        # Check if the file exists
        if not os.path.exists(input_path):
            logger.error(f"Counterfactuals file {input_path} not found")
            return

        # Load the data
        with open(input_path, "r") as f:
            data = json.load(f)

        # Convert to CounterfactualCandidate objects
        self.counterfactuals = []
        for cf_data in data.get("counterfactuals", []):
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

        logger.info(f"Loaded {len(self.counterfactuals)} counterfactuals from {input_path}")
