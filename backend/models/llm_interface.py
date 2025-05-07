"""
LLM Interface module for the XAIR system.
Provides an interface to the Llama 3 model with Mac-friendly optimizations.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    TextIteratorStreamer
)
from threading import Thread
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
    max_new_tokens: int = 512
    repetition_penalty: float = 1.1
    do_sample: bool = True
    output_hidden_states: bool = True
    output_attentions: bool = True
    return_dict_in_generate: bool = True

class LlamaInterface:
    """Interface for the Llama 3 model."""

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-3.2-1B",
        device: str = "auto",
        load_in_4bit: bool = False,
        cpu_offloading: bool = False,
        use_fp16: bool = True,
        use_bettertransformer: bool = False,
        verbose: bool = False,
        fast_init: bool = False  # New parameter for faster initialization
    ):
        """
        Initialize the Llama 3 interface.

        Args:
            model_name_or_path: Path to the model or model identifier
            device: Device to load the model on ('cpu', 'cuda', 'mps', or 'auto')
            load_in_4bit: Ignored for Mac compatibility
            cpu_offloading: Whether to offload some layers to CPU
            use_fp16: Whether to use half precision (float16)
            use_bettertransformer: Whether to use BetterTransformer for optimization
            verbose: Whether to log detailed information
            fast_init: Skip non-essential parts of initialization for faster startup
        """
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.INFO)

        # Determine the right device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            logger.info(f"Using device: {device}")

        self.device = device
        self.model_name = model_name_or_path

        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Load model with Mac-friendly optimizations
        logger.info(f"Loading model from {model_name_or_path}")

        # Configure model loading
        dtype = torch.float16 if use_fp16 and device != "cpu" else torch.float32

        logger.info(f"Using dtype: {dtype}")

        # Define device map
        if cpu_offloading:
            device_map = "auto"
            logger.info("Using automatic device mapping with CPU offloading")
        else:
            device_map = device
            logger.info(f"Using device: {device_map}")

        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation="eager"
            )

            # Apply BetterTransformer if requested (helps with MPS performance)
            if use_bettertransformer:
                try:
                    # Check if we're using a version that supports native attention
                    import pkg_resources
                    transformers_version = pkg_resources.get_distribution("transformers").version
                    torch_version = torch.__version__

                    # Check if we have a recent enough version for native optimizations
                    transformers_major = int(transformers_version.split('.')[0])
                    transformers_minor = int(transformers_version.split('.')[1])
                    torch_major = int(torch_version.split('.')[0])
                    torch_minor = int(torch_version.split('.')[1])

                    if (transformers_major >= 4 and transformers_minor >= 36 and
                        torch_major >= 2 and torch_minor >= 1):
                        logger.info("Using native PyTorch SDPA optimizations (no BetterTransformer needed)")
                    else:
                        # Use BetterTransformer for older versions
                        from optimum.bettertransformer import BetterTransformer
                        logger.info("Applying BetterTransformer optimization")
                        self.model = BetterTransformer.transform(self.model)
                except ImportError:
                    logger.warning("optimum package not found. Install with: pip install optimum")
                    logger.warning("Continuing without BetterTransformer optimization")
                except Exception as e:
                    logger.warning(f"Failed to apply BetterTransformer: {e}")
                    logger.warning("Continuing without BetterTransformer optimization")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to CPU with minimal memory settings")

            # Last resort: Load with minimal settings on CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )

        # Set padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model and tokenizer loaded successfully")

        # Perform a tiny warmup generation to initialize parts of the model
        if not fast_init:
            with torch.no_grad():
                logger.info("Warming up model with a small generation...")
                _ = self.model.generate(
                    self.tokenizer("Hello", return_tensors="pt").to(self.device).input_ids,
                    max_new_tokens=1
                )
        else:
            logger.info("Skipping warmup generation (fast init enabled)")

    def generate_multiple_paths(
        self,
        prompt: str,
        temperatures: List[float] = [0.2, 0.7, 1.0],
        paths_per_temp: int = 1,
        generation_config: Optional[GenerationConfig] = None,
        fast_mode: bool = False  # New parameter
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple reasoning paths with different temperature settings.

        Args:
            prompt: The input prompt
            temperatures: List of temperature values to use
            paths_per_temp: Number of paths to generate per temperature
            generation_config: Configuration for generation
            fast_mode: If True, skip collecting hidden states and attention

        Returns:
            List of dictionaries containing the generated paths and associated metadata
        """
        if generation_config is None:
            generation_config = GenerationConfig()

        paths = []

        for temp in temperatures:
            for _ in range(paths_per_temp):
                config = GenerationConfig(**vars(generation_config))
                config.temperature = temp

                result = self.generate(prompt, config, fast_mode=fast_mode)

                # Add temperature info to the result
                result["temperature"] = temp
                paths.append(result)

        return paths

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response from the model.

        Args:
            prompt: Input prompt
            config: Generation configuration
            stream: Whether to stream the output

        Returns:
            Dictionary containing the generated text and associated metadata
        """
        if config is None:
            config = GenerationConfig()

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Set up streaming if requested
        streamer = None
        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            generation_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Always pass attention_mask
                streamer=streamer,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
            )

            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            return {"streamer": streamer}

        # Generate without streaming
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                output_hidden_states=config.output_hidden_states,
                output_attentions=config.output_attentions,
                return_dict_in_generate=config.return_dict_in_generate,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                output_scores = True,
                attention_mask = attention_mask,
            )

        # Get the generated text
        generated_ids = outputs.sequences
        generated_text = self.tokenizer.batch_decode(
            generated_ids[:, input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        # Create the result dictionary
        result = {
            "generated_text": generated_text,
            "prompt": prompt,
            "input_ids": input_ids,
            "generated_ids": generated_ids,
        }

        # Include hidden states and attentions if requested
        if config.output_hidden_states and hasattr(outputs, "hidden_states"):
            # Extract hidden states - these are tuples of tensors
            hidden_states = outputs.hidden_states

            # We need to format them for easier processing
            # Convert to list of numpy arrays to make them serializable
            # We'll also detach from GPU
            processed_hidden_states = []

            # Format depends on model type and generate function implementation
            if isinstance(hidden_states, tuple):
                for layer_states in hidden_states:
                    if isinstance(layer_states, tuple):
                        # For sequence generation with multiple steps
                        layer_processed = [state.detach().cpu().numpy() for state in layer_states]
                        processed_hidden_states.append(layer_processed)
                    else:
                        # Single tensor
                        processed_hidden_states.append(layer_states.detach().cpu().numpy())

            result["hidden_states"] = processed_hidden_states

        if config.output_attentions and hasattr(outputs, "attentions"):
            # Similar processing for attention matrices
            attentions = outputs.attentions
            processed_attentions = []

            if isinstance(attentions, tuple):
                for layer_attentions in attentions:
                    if isinstance(layer_attentions, tuple):
                        layer_processed = [state.detach().cpu().numpy() for state in layer_attentions]
                        processed_attentions.append(layer_processed)
                    else:
                        processed_attentions.append(layer_attentions.detach().cpu().numpy())

            result["attentions"] = processed_attentions

        # Add token probabilities
        if hasattr(outputs, "scores") and outputs.scores is not None:
            scores = [score.detach().cpu() for score in outputs.scores]
            # Convert to probabilities
            probs = [torch.softmax(score, dim=-1) for score in scores]

            # For each token, get the probability that was assigned to the chosen token
            token_probs = []
            for i, prob in enumerate(probs):
                # Get the index of the next token in the sequence
                token_id = generated_ids[0, input_ids.shape[1] + i].item()
                token_prob = prob[0, token_id].item()
                token_probs.append(token_prob)

            result["token_probabilities"] = token_probs

            # Get top-k alternatives for each position
            k = min(5, prob.shape[-1])  # Get top 5 or fewer if vocab is smaller
            topk_result = []

            for i, prob in enumerate(probs):
                topk_probs, topk_indices = torch.topk(prob[0], k)

                alternatives = []
                for j in range(k):
                    token_id = topk_indices[j].item()
                    token = self.tokenizer.decode([token_id])
                    prob_val = topk_probs[j].item()

                    alternatives.append({
                        "token": token,
                        "token_id": token_id,
                        "probability": prob_val
                    })

                topk_result.append(alternatives)

            result["token_alternatives"] = topk_result

        return result

    def get_token_entropies(self, token_alternatives: List[List[Dict]]) -> List[float]:
        """
        Calculate entropy for each token position based on alternatives.

        Args:
            token_alternatives: List of alternative tokens and their probabilities

        Returns:
            List of entropy values
        """
        entropies = []

        for pos_alternatives in token_alternatives:
            probs = [alt["probability"] for alt in pos_alternatives]
            # Normalize probabilities if they don't sum to 1
            prob_sum = sum(probs)
            if prob_sum < 0.99 or prob_sum > 1.01:  # Allow small floating point errors
                probs = [p / prob_sum for p in probs]

            # Calculate entropy: -sum(p_i * log(p_i))
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
            entropies.append(entropy)

        return entropies

    def save_generation_results(self, results: List[Dict], output_dir: str):
        """
        Save generation results to disk.

        Args:
            results: List of generation results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data for saving (remove non-serializable elements)
        serializable_results = []

        for i, result in enumerate(results):
            clean_result = {
                "prompt": result["prompt"],
                "generated_text": result["generated_text"],
                "temperature": result.get("temperature", 0.7),
            }

            # Add token probabilities if available
            if "token_probabilities" in result:
                clean_result["token_probabilities"] = result["token_probabilities"]

            # Add token alternatives if available
            if "token_alternatives" in result:
                # Convert token alternatives to serializable format
                serializable_alternatives = []
                for pos_alts in result["token_alternatives"]:
                    serializable_alternatives.append([
                        {"token": alt["token"], "probability": alt["probability"]}
                        for alt in pos_alts
                    ])
                clean_result["token_alternatives"] = serializable_alternatives

            serializable_results.append(clean_result)

        # Save to file
        output_file = os.path.join(output_dir, "generation_results.json")
        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved generation results to {output_file}")
