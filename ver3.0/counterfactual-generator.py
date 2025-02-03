from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
from uuid import uuid4
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from llm_interface import DeepSeekInterface
from reasoning_tree import ReasoningTreeGenerator, TreeNode
import asyncio
from functools import lru_cache

@dataclass
class Counterfactual:
    id: str
    original_text: str
    modified_text: str
    modification_type: str
    target_outcome: str
    actual_outcome: str
    probability: float
    attention_score: float
    parent_node_id: str
    embedding: Optional[np.ndarray] = None
    
    def effectiveness_score(self) -> float:
        """Calculate how effective this counterfactual is at achieving its goal"""
        return self.probability * self.attention_score

class CounterfactualCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        
    def add(self, cf: Counterfactual):
        if len(self.cache) >= self.max_size:
            # Remove lowest effectiveness score
            min_key = min(self.cache.items(), 
                         key=lambda x: x[1].effectiveness_score())
            del self.cache[min_key[0]]
        self.cache[cf.id] = cf
        
    def get_similar(self, embedding: np.ndarray, threshold: float = 0.8) -> List[Counterfactual]:
        similar = []
        for cf in self.cache.values():
            if cf.embedding is not None:
                similarity = cosine_similarity([embedding], [cf.embedding])[0][0]
                if similarity > threshold:
                    similar.append(cf)
        return similar

class CounterfactualGenerator:
    def __init__(
        self,
        llm: DeepSeekInterface,
        tree_generator: ReasoningTreeGenerator,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        similarity_threshold: float = 0.8,
        max_counterfactuals_per_node: int = 5,
        min_probability: float = 0.1,
        cache_size: int = 1000
    ):
        self.llm = llm
        self.tree_generator = tree_generator
        self.encoder = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_counterfactuals_per_node = max_counterfactuals_per_node
        self.min_probability = min_probability
        self.cache = CounterfactualCache(cache_size)
        
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string"""
        return self.encoder.encode(text)
    
    async def generate_counterfactuals(
        self,
        reasoning_tree: nx.DiGraph
    ) -> List[Counterfactual]:
        """Generate counterfactuals for all significant nodes in the tree"""
        counterfactuals = []
        
        # Process nodes in breadth-first order
        for node_id in nx.bfs_tree(reasoning_tree, 
                                  list(reasoning_tree.nodes())[0]):
            node = reasoning_tree.nodes[node_id]['node']
            
            # Skip nodes with low attention or probability
            if (node.attention_weight < self.min_probability or 
                node.probability < self.min_probability):
                continue
                
            new_counterfactuals = await self._generate_node_counterfactuals(
                node,
                reasoning_tree
            )
            counterfactuals.extend(new_counterfactuals)
            
        return counterfactuals
    
    async def _generate_node_counterfactuals(
        self,
        node: TreeNode,
        tree: nx.DiGraph
    ) -> List[Counterfactual]:
        """Generate counterfactuals for a specific node"""
        counterfactuals = []
        
        # Get the full context up to this node
        context = self._get_node_context(node, tree)
        
        # Generate different types of modifications
        modifications = await asyncio.gather(
            self._generate_token_substitutions(context),
            self._generate_context_modifications(context),
            self._generate_semantic_alternatives(context)
        )
        
        for mod_list in modifications:
            for mod in mod_list:
                # Create counterfactual
                cf = await self._create_counterfactual(
                    original_text=context,
                    modified_text=mod['text'],
                    modification_type=mod['type'],
                    node=node
                )
                
                # Check similarity with existing counterfactuals
                if cf and not self._is_too_similar(cf, counterfactuals):
                    counterfactuals.append(cf)
                    
                # Stop if we have enough counterfactuals for this node
                if len(counterfactuals) >= self.max_counterfactuals_per_node:
                    break
                    
        return counterfactuals
    
    def _get_node_context(self, node: TreeNode, tree: nx.DiGraph) -> str:
        """Get the full text context leading to a node"""
        path = []
        current = node
        
        while current.parent_id is not None:
            path.append(current.text)
            current = tree.nodes[current.parent_id]['node']
            
        path.append(current.text)  # Add root node
        return " ".join(reversed(path))
    
    async def _generate_token_substitutions(
        self,
        context: str
    ) -> List[Dict[str, str]]:
        """Generate alternatives by substituting key tokens"""
        try:
            # Get token probabilities
            token_probs = await self.llm.get_token_probabilities(context)
            
            # Filter for significant tokens
            significant_tokens = {
                token: prob for token, prob in token_probs.items()
                if prob >= self.min_probability
            }
            
            modifications = []
            for token in significant_tokens:
                # Generate alternative using LLM
                prompt = f"Replace '{token}' in this context with a different word that changes the meaning: {context}"
                response = await self.llm.query(prompt)
                
                modifications.append({
                    'text': response.text,
                    'type': 'token_substitution'
                })
                
            return modifications
        except Exception as e:
            print(f"Error generating token substitutions: {e}")
            return []
    
    async def _generate_context_modifications(
        self,
        context: str
    ) -> List[Dict[str, str]]:
        """Generate alternatives by modifying the context"""
        try:
            prompts = [
                f"Modify this text to lead to a different conclusion: {context}",
                f"What's an alternative version of this that changes the outcome: {context}",
                f"Rewrite this text to explore a different possibility: {context}"
            ]
            
            responses = await asyncio.gather(
                *(self.llm.query(prompt) for prompt in prompts)
            )
            
            return [
                {'text': resp.text, 'type': 'context_modification'}
                for resp in responses
            ]
        except Exception as e:
            print(f"Error generating context modifications: {e}")
            return []
    
    async def _generate_semantic_alternatives(
        self,
        context: str
    ) -> List[Dict[str, str]]:
        """Generate semantically different alternatives"""
        try:
            prompts = [
                f"Express this idea in a completely different way: {context}",
                f"What's a contrasting perspective on this: {context}"
            ]
            
            responses = await asyncio.gather(
                *(self.llm.query(prompt) for prompt in prompts)
            )
            
            return [
                {'text': resp.text, 'type': 'semantic_alternative'}
                for resp in responses
            ]
        except Exception as e:
            print(f"Error generating semantic alternatives: {e}")
            return []
    
    async def _create_counterfactual(
        self,
        original_text: str,
        modified_text: str,
        modification_type: str,
        node: TreeNode
    ) -> Optional[Counterfactual]:
        """Create a counterfactual instance with necessary metadata"""
        try:
            # Get LLM response for modified text
            response = await self.llm.query(modified_text)
            attention_flow = await self.llm.get_attention_flow(modified_text)
            
            # Calculate attention score
            attention_score = float(np.mean(attention_flow))
            
            cf = Counterfactual(
                id=str(uuid4()),
                original_text=original_text,
                modified_text=modified_text,
                modification_type=modification_type,
                target_outcome="alternative_path",  # Could be more specific
                actual_outcome=response.text,
                probability=response.logits[0] if response.logits is not None else 0.0,
                attention_score=attention_score,
                parent_node_id=node.id,
                embedding=self.get_embedding(modified_text)
            )
            
            # Add to cache
            self.cache.add(cf)
            
            return cf
        except Exception as e:
            print(f"Error creating counterfactual: {e}")
            return None
    
    def _is_too_similar(
        self,
        new_cf: Counterfactual,
        existing_cfs: List[Counterfactual]
    ) -> bool:
        """Check if a new counterfactual is too similar to existing ones"""
        if not existing_cfs:
            return False
            
        for cf in existing_cfs:
            if cf.embedding is not None and new_cf.embedding is not None:
                similarity = cosine_similarity([cf.embedding], 
                                            [new_cf.embedding])[0][0]
                if similarity > self.similarity_threshold:
                    return True
        return False
    
    def get_most_diverse_counterfactuals(
        self,
        counterfactuals: List[Counterfactual],
        n: int = 5
    ) -> List[Counterfactual]:
        """Select the most diverse set of counterfactuals"""
        if not counterfactuals or n <= 0:
            return []
            
        selected = [counterfactuals[0]]  # Start with the first one
        
        while len(selected) < n and len(selected) < len(counterfactuals):
            max_min_similarity = -1
            best_candidate = None
            
            for candidate in counterfactuals:
                if candidate in selected:
                    continue
                    
                # Calculate minimum similarity to already selected
                min_similarity = min(
                    cosine_similarity([candidate.embedding], 
                                    [sel.embedding])[0][0]
                    for sel in selected
                )
                
                if min_similarity > max_min_similarity:
                    max_min_similarity = min_similarity
                    best_candidate = candidate
                    
            if best_candidate is not None:
                selected.append(best_candidate)
            else:
                break
                
        return selected

# Example usage:
'''async def main():
    async with DeepSeekInterface(api_key="your_api_key_here") as llm:
        tree_generator = ReasoningTreeGenerator(llm)
        cf_generator = CounterfactualGenerator(llm, tree_generator)
        
        # Generate reasoning tree
        tree = await tree_generator.generate_tree(
            "The capital of France is Paris"
        )
        
        # Generate counterfactuals
        counterfactuals = await cf_generator.generate_counterfactuals(tree)
        
        # Get diverse subset
        diverse_cfs = cf_generator.get_most_diverse_counterfactuals(
            counterfactuals,
            n=5
        )
        
        for cf in diverse_cfs:
            print(f"\nOriginal: {cf.original_text}")
            print(f"Modified: {cf.modified_text}")
            print(f"Type: {cf.modification_type}")
            print(f"Effectiveness: {cf.effectiveness_score()}")
'''

if __name__ == "__main__":
    asyncio.run(main())
