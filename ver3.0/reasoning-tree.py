from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import numpy as np
from uuid import uuid4
import networkx as nx
from llm_interface import DeepSeekInterface, LLMResponse

@dataclass
class TreeNode:
    id: str
    text: str
    probability: float
    attention_weight: float
    parent_id: Optional[str] = None
    children_ids: Set[str] = None
    is_counterfactual: bool = False
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = set()

class ReasoningTreeGenerator:
    def __init__(
        self,
        llm: DeepSeekInterface,
        max_depth: int = 10,
        max_branches: int = 3,
        probability_threshold: float = 0.1,
        attention_threshold: float = 0.1
    ):
        self.llm = llm
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.probability_threshold = probability_threshold
        self.attention_threshold = attention_threshold
        self.graph = nx.DiGraph()
        
    async def generate_tree(self, prompt: str) -> nx.DiGraph:
        """
        Generate a reasoning tree from the given prompt.
        Returns a NetworkX DiGraph representing the tree.
        """
        # Initialize root node
        root_id = str(uuid4())
        root_node = TreeNode(
            id=root_id,
            text=prompt,
            probability=1.0,
            attention_weight=1.0
        )
        self.graph.add_node(root_id, node=root_node)
        
        # Generate tree recursively
        await self._expand_node(root_id, prompt, depth=0)
        return self.graph
    
    async def _expand_node(self, node_id: str, current_text: str, depth: int):
        """
        Recursively expand a node in the reasoning tree.
        """
        if depth >= self.max_depth:
            return
            
        # Get next token predictions
        try:
            response = await self.llm.query(current_text)
            token_probs = await self.llm.get_token_probabilities(current_text)
            attention_flow = await self.llm.get_attention_flow(current_text)
        except Exception as e:
            print(f"Error expanding node {node_id}: {e}")
            return
            
        # Sort tokens by probability
        sorted_tokens = sorted(
            token_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top branches that meet probability threshold
        valid_branches = [
            (token, prob) for token, prob in sorted_tokens[:self.max_branches]
            if prob >= self.probability_threshold
        ]
        
        # Create child nodes for each valid branch
        for token, prob in valid_branches:
            child_id = str(uuid4())
            
            # Calculate attention weight for this token
            attention_weight = self._calculate_attention_weight(
                attention_flow,
                len(current_text.split())  # Approximate position
            )
            
            # Only create branch if attention is significant enough
            if attention_weight >= self.attention_threshold:
                child_node = TreeNode(
                    id=child_id,
                    text=token,
                    probability=prob,
                    attention_weight=attention_weight,
                    parent_id=node_id
                )
                
                # Add child to graph
                self.graph.add_node(child_id, node=child_node)
                self.graph.add_edge(node_id, child_id)
                
                # Update parent's children set
                parent_node = self.graph.nodes[node_id]['node']
                parent_node.children_ids.add(child_id)
                
                # Recursively expand child
                new_text = current_text + " " + token
                await self._expand_node(child_id, new_text, depth + 1)
    
    def _calculate_attention_weight(
        self,
        attention_matrix: np.ndarray,
        position: int
    ) -> float:
        """
        Calculate attention weight for a token at given position.
        """
        if len(attention_matrix.shape) < 2:
            return 1.0
            
        # Take the attention weights for the current position
        if position < attention_matrix.shape[1]:
            weights = attention_matrix[:, position]
            return float(np.mean(weights))
        return 1.0
    
    def get_path_probabilities(self, node_id: str) -> List[float]:
        """
        Get the probabilities along the path from root to given node.
        """
        probabilities = []
        current_id = node_id
        
        while current_id in self.graph:
            node = self.graph.nodes[current_id]['node']
            probabilities.append(node.probability)
            current_id = node.parent_id
            
        return list(reversed(probabilities))
    
    def get_most_likely_path(self) -> List[str]:
        """
        Get the path with highest cumulative probability.
        """
        leaf_nodes = [n for n in self.graph.nodes() 
                     if self.graph.out_degree(n) == 0]
        
        max_prob = -1
        best_path = None
        
        for leaf in leaf_nodes:
            path = nx.shortest_path(self.graph, 
                                  source=list(self.graph.nodes())[0],
                                  target=leaf)
            path_probs = self.get_path_probabilities(leaf)
            cumulative_prob = np.prod(path_probs)
            
            if cumulative_prob > max_prob:
                max_prob = cumulative_prob
                best_path = path
                
        return best_path if best_path else []
    
    def to_dict(self) -> Dict:
        """
        Convert the tree to a dictionary format for frontend rendering.
        """
        tree_dict = {
            "nodes": [],
            "edges": []
        }
        
        for node_id in self.graph.nodes():
            node = self.graph.nodes[node_id]['node']
            tree_dict["nodes"].append({
                "id": node.id,
                "text": node.text,
                "probability": node.probability,
                "attention_weight": node.attention_weight,
                "is_counterfactual": node.is_counterfactual
            })
        
        for edge in self.graph.edges():
            tree_dict["edges"].append({
                "source": edge[0],
                "target": edge[1]
            })
            
        return tree_dict

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
