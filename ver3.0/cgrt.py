from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
from uuid import uuid4
import networkx as nx
from collections import defaultdict
import torch
import asyncio

@dataclass
class GenerationPath:
    tokens: List[str]
    probabilities: List[float]
    attention_maps: List[np.ndarray]
    divergence_points: List[int]
    importance_score: float

@dataclass
class CrossGeneration:
    paths: List[GenerationPath]
    shared_prefix: List[str]
    divergence_map: Dict[int, Set[str]]
    attention_flow: np.ndarray

class CGRTGenerator:
    def __init__(
        self,
        llm,
        num_generations: int = 5,
        temperature_range: Tuple[float, float] = (0.7, 1.3),
        min_probability_threshold: float = 0.1,
        max_branches_per_node: int = 3,
        attention_threshold: float = 0.2
    ):
        self.llm = llm
        self.num_generations = num_generations
        self.temperature_range = temperature_range
        self.min_probability_threshold = min_probability_threshold
        self.max_branches_per_node = max_branches_per_node
        self.attention_threshold = attention_threshold
        self.generation_cache = {}

    async def generate_cross_paths(
        self,
        prompt: str,
        max_tokens: int = 50
    ) -> CrossGeneration:
        # Generate multiple paths with different temperatures
        temperatures = np.linspace(
            self.temperature_range[0],
            self.temperature_range[1],
            self.num_generations
        )
        
        generation_tasks = [
            self._generate_single_path(prompt, temp, max_tokens)
            for temp in temperatures
        ]
        
        paths = await asyncio.gather(*generation_tasks)
        
        # Find common prefix across all generations
        shared_prefix = self._find_shared_prefix(paths)
        
        # Identify divergence points
        divergence_map = self._map_divergences(paths, shared_prefix)
        
        # Calculate aggregate attention flow
        attention_flow = self._aggregate_attention(paths)
        
        return CrossGeneration(
            paths=paths,
            shared_prefix=shared_prefix,
            divergence_map=divergence_map,
            attention_flow=attention_flow
        )

    async def _generate_single_path(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> GenerationPath:
        tokens = []
        probabilities = []
        attention_maps = []
        divergence_points = []
        
        current_text = prompt
        
        for _ in range(max_tokens):
            # Generate next token
            response = await self.llm.query(
                current_text,
                temperature=temperature,
                return_token_probs=True,
                return_attention=True
            )
            
            # Get token probabilities
            token_probs = await self.llm.get_token_probabilities(current_text)
            
            # Get attention map
            attention_map = await self.llm.get_attention_flow(current_text)
            
            # Find high-probability alternative tokens
            alternatives = [
                (token, prob) for token, prob in token_probs.items()
                if prob >= self.min_probability_threshold
            ]
            
            # If we have multiple viable alternatives, mark as divergence
            if len(alternatives) > 1:
                divergence_points.append(len(tokens))
            
            # Select next token
            next_token = response.text.split()[-1]
            current_text = f"{current_text} {next_token}"
            
            tokens.append(next_token)
            probabilities.append(token_probs[next_token])
            attention_maps.append(attention_map)
        
        # Calculate path importance
        importance_score = self._calculate_path_importance(
            probabilities,
            attention_maps
        )
        
        return GenerationPath(
            tokens=tokens,
            probabilities=probabilities,
            attention_maps=attention_maps,
            divergence_points=divergence_points,
            importance_score=importance_score
        )

    def _find_shared_prefix(
        self,
        paths: List[GenerationPath]
    ) -> List[str]:
        """Find the common token prefix across all generations"""
        min_len = min(len(path.tokens) for path in paths)
        shared = []
        
        for i in range(min_len):
            tokens = set(path.tokens[i] for path in paths)
            if len(tokens) > 1:
                break
            shared.append(list(tokens)[0])
            
        return shared

    def _map_divergences(
        self,
        paths: List[GenerationPath],
        shared_prefix: List[str]
    ) -> Dict[int, Set[str]]:
        """Map positions to sets of divergent tokens"""
        divergence_map = defaultdict(set)
        
        prefix_len = len(shared_prefix)
        max_len = max(len(path.tokens) for path in paths)
        
        for pos in range(prefix_len, max_len):
            tokens = set()
            for path in paths:
                if pos < len(path.tokens):
                    tokens.add(path.tokens[pos])
            if len(tokens) > 1:
                divergence_map[pos] = tokens
                
        return divergence_map

    def _aggregate_attention(
        self,
        paths: List[GenerationPath]
    ) -> np.ndarray:
        """Aggregate attention maps across all paths"""
        attention_arrays = [
            np.stack(path.attention_maps)
            for path in paths
        ]
        
        # Average attention across all paths
        return np.mean(attention_arrays, axis=0)

    def _calculate_path_importance(
        self,
        probabilities: List[float],
        attention_maps: List[np.ndarray]
    ) -> float:
        """Calculate overall path importance score"""
        prob_score = np.mean(probabilities)
        attention_score = np.mean([
            np.mean(attention_map) 
            for attention_map in attention_maps
        ])
        
        return 0.7 * prob_score + 0.3 * attention_score