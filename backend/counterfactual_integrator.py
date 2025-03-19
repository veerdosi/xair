from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
import networkx as nx
from uuid import uuid4
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from backend.cgrt import CGRTGenerator
from backend.cgrt_tree import CGRTNode

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
        cgrt_generator: CGRTGenerator,
        cf_generator, # Type hint avoided to prevent circular imports
        similarity_threshold: float = 0.7,
        max_group_size: int = 5,
        min_connection_strength: float = 0.3
    ):
        self.cgrt_generator = cgrt_generator
        self.cf_generator = cf_generator
        self.similarity_threshold = similarity_threshold
        self.max_group_size = max_group_size
        self.min_connection_strength = min_connection_strength
        self.overlay = CounterfactualOverlay()
        
    async def integrate_counterfactuals(self, base_tree: nx.DiGraph, counterfactuals) -> nx.DiGraph:
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
    
    async def _find_integration_points(self, tree: nx.DiGraph, counterfactuals) -> Dict[str, List[Tuple[str, float]]]:
        points = {}
        for cf in counterfactuals:
            similar_nodes = []
            for node_id in tree.nodes():
                node_data = tree.nodes[node_id]
                if 'node' not in node_data:
                    continue
                
                node = node_data['node']
                if not hasattr(node, 'text'):
                    # Get text from tokens if available
                    if hasattr(node, 'tokens'):
                        node_text = " ".join(node.tokens)
                    else:
                        continue
                else:
                    node_text = node.text
                
                similarity = self._calculate_similarity(cf.embedding, self.cf_generator.get_embedding(node_text))
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
    
    def _calculate_connection_strength(self, node: CGRTNode, counterfactual, tree: nx.DiGraph) -> float:
        # Handle different node types (CGRTNode vs TreeNode)
        if hasattr(node, 'probability'):
            node_prob = node.probability
        elif hasattr(node, 'probabilities'):
            node_prob = np.mean(node.probabilities)
        else:
            node_prob = 0.5
            
        if hasattr(node, 'attention_weight'):
            node_attention = node.attention_weight
        elif hasattr(node, 'importance_score'):
            node_attention = node.importance_score
        else:
            node_attention = 0.5
            
        # Calculate connection strength
        prob_factor = node_prob * counterfactual.probability
        attention_factor = node_attention * counterfactual.attention_score
        
        # Get depth in tree
        try:
            start_node = list(nx.topological_sort(tree))[0]
            depth = nx.shortest_path_length(tree, start_node, node.id)
        except (nx.NetworkXError, nx.NetworkXNoPath):
            depth = 0
            
        structure_factor = 1.0 / (1.0 + depth)
        strength = 0.4 * prob_factor + 0.4 * attention_factor + 0.2 * structure_factor
        return float(strength)
    
    def _group_counterfactuals(self, counterfactuals) -> Dict[str, List]:
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
    
    async def _integrate_group(self, tree: nx.DiGraph, counterfactuals, integration_points: Dict[str, List[Tuple[str, float]]]):
        for cf in counterfactuals:
            if cf.id not in self.overlay.integration_points:
                continue
            integration_point = self.overlay.integration_points[cf.id]
            original_node_id = integration_point.original_node_id
            
            # Create a node suitable for the tree (CGRTNode)
            cf_node = CGRTNode(
                id=cf.id,
                tokens=cf.modified_text.split(),
                probabilities=[cf.probability],
                attention_pattern=np.array([[cf.attention_score]]),
                importance_score=cf.probability * cf.attention_score,
                metadata={"is_counterfactual": True, "original_text": cf.original_text}
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
            node_data = tree.nodes[node_id]
            if 'node' not in node_data:
                continue
                
            node = node_data['node']
            
            # Handle different node types
            if hasattr(node, 'text'):
                text = node.text
            elif hasattr(node, 'tokens'):
                text = " ".join(node.tokens)
            else:
                text = f"Node {node_id}"
                
            if hasattr(node, 'probability'):
                probability = node.probability
            elif hasattr(node, 'probabilities') and node.probabilities:
                probability = np.mean(node.probabilities)
            else:
                probability = 0.5
                
            if hasattr(node, 'attention_weight'):
                attention = node.attention_weight
            elif hasattr(node, 'importance_score'):
                attention = node.importance_score
            else:
                attention = 0.5
                
            is_counterfactual = False
            if hasattr(node, 'is_counterfactual'):
                is_counterfactual = node.is_counterfactual
            elif hasattr(node, 'metadata') and node.metadata and 'is_counterfactual' in node.metadata:
                is_counterfactual = node.metadata['is_counterfactual']
                
            viz_data["nodes"].append({
                "id": node.id,
                "text": text,
                "probability": probability,
                "attention_weight": attention,
                "is_counterfactual": is_counterfactual
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