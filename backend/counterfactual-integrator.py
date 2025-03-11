from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
import networkx as nx
from uuid import uuid4
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .counterfactual import Counterfactual
    from .cgrt_tree import CGRTTree

@dataclass
class IntegrationPoint:
    original_node_id: str
    counterfactual_id: str
    similarity_score: float
    connection_strength: float
    reachable_nodes: Set[str]

class CounterfactualOverlay:
    def __init__(self):
        self.overlay_graph = nx.DiGraph()
        self.integration_points: Dict[str, IntegrationPoint] = {}
        from collections import defaultdict
        self.counterfactual_groups: Dict[str, Set[str]] = defaultdict(set)
        
    def add_counterfactual(self, cf_id: str, original_node_id: str, similarity: float, strength: float):
        self.integration_points[cf_id] = IntegrationPoint(
            original_node_id=original_node_id,
            counterfactual_id=cf_id,
            similarity_score=similarity,
            connection_strength=strength,
            reachable_nodes=set()
        )
        
    def add_to_group(self, group_id: str, cf_id: str):
        self.counterfactual_groups[group_id].add(cf_id)
        
    def get_group(self, cf_id: str) -> Optional[str]:
        for group_id, members in self.counterfactual_groups.items():
            if cf_id in members:
                return group_id
        return None

class CounterfactualIntegrator:
    def __init__(
        self,
        tree_generator: ReasoningTreeGenerator,
        cf_generator: CounterfactualGenerator,
        similarity_threshold: float = 0.7,
        max_group_size: int = 5,
        min_connection_strength: float = 0.3
    ):
        self.tree_generator = tree_generator
        self.cf_generator = cf_generator
        self.similarity_threshold = similarity_threshold
        self.max_group_size = max_group_size
        self.min_connection_strength = min_connection_strength
        self.overlay = CounterfactualOverlay()
        
    async def integrate_counterfactuals(self, base_tree: nx.DiGraph, counterfactuals: List[Counterfactual]) -> nx.DiGraph:
        # Create a copy of the base tree
        integrated_tree = base_tree.copy()
        batch_size = 10
        for i in range(0, len(counterfactuals), batch_size):
            batch = counterfactuals[i:i + batch_size]
            integration_points = await self._find_integration_points(base_tree, batch)
            groups = self._group_counterfactuals(batch)
            for group_id, group_cfs in groups.items():
                await self._integrate_group(integrated_tree, group_cfs, integration_points)
        return integrated_tree
    
    async def _find_integration_points(self, tree: nx.DiGraph, counterfactuals: List[Counterfactual]) -> Dict[str, List[Tuple[str, float]]]:
        points = {}
        for cf in counterfactuals:
            similar_nodes = []
            for node_id in tree.nodes():
                node = tree.nodes[node_id]['node']
                similarity = self._calculate_similarity(cf.embedding, self.cf_generator.get_embedding(node.text))
                if similarity > self.similarity_threshold:
                    strength = self._calculate_connection_strength(node, cf, tree)
                    if strength > self.min_connection_strength:
                        similar_nodes.append((node_id, similarity, strength))
            similar_nodes.sort(key=lambda x: x[1] * x[2], reverse=True)
            if similar_nodes:
                node_id, similarity, strength = similar_nodes[0]
                self.overlay.add_counterfactual(cf.id, node_id, similarity, strength)
                points[cf.id] = similar_nodes
        return points
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return float(cosine_similarity([embedding1], [embedding2])[0][0])
    
    def _calculate_connection_strength(self, node: TreeNode, counterfactual: Counterfactual, tree: nx.DiGraph) -> float:
        prob_factor = node.probability * counterfactual.probability
        attention_factor = node.attention_weight * counterfactual.attention_score
        depth = nx.shortest_path_length(tree, list(tree.nodes())[0], node.id)
        structure_factor = 1.0 / (1.0 + depth)
        strength = 0.4 * prob_factor + 0.4 * attention_factor + 0.2 * structure_factor
        return float(strength)
    
    def _group_counterfactuals(self, counterfactuals: List[Counterfactual]) -> Dict[str, List[Counterfactual]]:
        groups = {}
        processed = set()
        for i, cf1 in enumerate(counterfactuals):
            if cf1.id in processed:
                continue
            group_id = str(uuid4())
            current_group = [cf1]
            processed.add(cf1.id)
            for cf2 in counterfactuals[i + 1:]:
                if cf2.id in processed:
                    continue
                similarity = self._calculate_similarity(cf1.embedding, cf2.embedding)
                if similarity > self.similarity_threshold and len(current_group) < self.max_group_size:
                    current_group.append(cf2)
                    processed.add(cf2.id)
            groups[group_id] = current_group
        return groups
    
    async def _integrate_group(self, tree: nx.DiGraph, counterfactuals: List[Counterfactual], integration_points: Dict[str, List[Tuple[str, float]]]):
        for cf in counterfactuals:
            if cf.id not in self.overlay.integration_points:
                continue
            integration_point = self.overlay.integration_points[cf.id]
            original_node_id = integration_point.original_node_id
            cf_node = TreeNode(
                id=cf.id,
                text=cf.modified_text,
                probability=cf.probability,
                attention_weight=cf.attention_score,
                parent_id=original_node_id,
                is_counterfactual=True
            )
            tree.add_node(cf.id, node=cf_node)
            tree.add_edge(original_node_id, cf.id, weight=integration_point.connection_strength)
            self._update_reachable_nodes(tree, cf.id, integration_point)
    
    def _update_reachable_nodes(self, tree: nx.DiGraph, cf_node_id: str, integration_point: IntegrationPoint):
        descendants = nx.descendants(tree, cf_node_id)
        integration_point.reachable_nodes.update(descendants)
    
    def get_visualization_data(self, tree: nx.DiGraph) -> Dict:
        viz_data = {"nodes": [], "edges": [], "groups": []}
        for node_id in tree.nodes():
            node = tree.nodes[node_id]['node']
            viz_data["nodes"].append({
                "id": node.id,
                "text": node.text,
                "probability": node.probability,
                "attention_weight": node.attention_weight,
                "is_counterfactual": node.is_counterfactual
            })
        for source, target, data in tree.edges(data=True):
            viz_data["edges"].append({
                "source": source,
                "target": target,
                "weight": data.get("weight", 1.0)
            })
        for group_id, members in self.overlay.counterfactual_groups.items():
            viz_data["groups"].append({
                "id": group_id,
                "members": list(members)
            })
        return viz_data

#############################################
# MAIN EXAMPLE USAGE
#############################################

def run_vcnet_example():
    print("=== Running VCNet-based Counterfactual Generation Example ===")
    # Example input text and a dummy goal specification.
    text = "The quick brown fox jumps over the lazy dog."
    goal_spec = GoalSpecification(
        target_aspects={'content': 0.8, 'structure': 0.7, 'logic': 0.9},
        constraints={'max_change': 0.5}
    )
    ranked_counterfactuals = generate_counterfactuals(text, goal_spec)
    for idx, (cf_text, scores) in enumerate(ranked_counterfactuals):
        print(f"\nCounterfactual {idx+1}:")
        print(f"Text: {cf_text}")
        print(f"Scores: {scores}")

async def run_llm_integration_example():
    print("\n=== Running LLM-based Counterfactual Integration Example ===")
    # Initialize your LLM interface and reasoning tree generator.
    async with DeepSeekInterface(api_key="your_api_key_here") as llm:
        tree_generator = ReasoningTreeGenerator(llm)
        cf_generator = CounterfactualGenerator(llm, tree_generator)
        integrator = CounterfactualIntegrator(tree_generator, cf_generator)
        
        # Generate a base reasoning tree (the prompt is an example)
        base_tree = await tree_generator.generate_tree("The capital of France is Paris")
        
        # Generate counterfactuals using the LLM-based approach
        counterfactuals = await cf_generator.generate_counterfactuals(base_tree)
        
        # Integrate the counterfactuals into the reasoning tree
        integrated_tree = await integrator.integrate_counterfactuals(base_tree, counterfactuals)
        
        # Get visualization data for the integrated tree
        viz_data = integrator.get_visualization_data(integrated_tree)
        print("\nVisualization Data:")
        print(viz_data)

if __name__ == "__main__":
    # Run the synchronous VCNet example
    run_vcnet_example()
    # Run the asynchronous LLM integration example
    asyncio.run(run_llm_integration_example())