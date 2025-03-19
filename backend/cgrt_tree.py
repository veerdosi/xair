from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple, Any
import numpy as np
import networkx as nx
from uuid import uuid4
from collections import defaultdict
from scipy.spatial.distance import cosine
from backend.error_handling import XAIRError, log_exception

@dataclass
class CGRTNode:
    id: str
    tokens: List[str]
    probabilities: List[float]
    attention_pattern: np.ndarray
    importance_score: float
    merge_points: Set[int] = None
    cross_links: Set[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.merge_points is None:
            self.merge_points = set()
        if self.cross_links is None:
            self.cross_links = set()
        if self.metadata is None:
            self.metadata = {}
            
    @property
    def text(self) -> str:
        return " ".join(self.tokens)
        
    @property
    def cumulative_probability(self) -> float:
        return np.prod(self.probabilities)

class CGRTTree:
    def __init__(
        self,
        merge_threshold: float = 0.8,
        importance_threshold: float = 0.3,
        max_cross_links: int = 3
    ):
        self.graph = nx.DiGraph()
        self.merge_threshold = merge_threshold
        self.importance_threshold = importance_threshold
        self.max_cross_links = max_cross_links
        self.path_cache = {}
        
    def build_from_generations(
        self,
        generations: CrossGeneration
    ) -> nx.DiGraph:
        """Construct CGRT from multiple generation paths"""
        # Start with shared prefix
        root_node = self._create_root_node(generations.shared_prefix)
        self.graph.add_node(root_node.id, node=root_node)
        
        # Process each path
        for path in generations.paths:
            self._integrate_path(path, root_node.id, generations.attention_flow)
            
        # Find and add cross-links
        self._add_cross_links(generations.attention_flow)
        
        # Prune low-importance branches
        self._prune_branches()
        
        return self.graph
    
    def _create_root_node(self, shared_prefix: List[str]) -> CGRTNode:
        """Create root node from shared prefix"""
        return CGRTNode(
            id=str(uuid4()),
            tokens=shared_prefix,
            probabilities=[1.0] * len(shared_prefix),
            attention_pattern=np.ones((len(shared_prefix), len(shared_prefix))),
            importance_score=1.0
        )
        
    def _integrate_path(
        self,
        path: GenerationPath,
        root_id: str,
        global_attention: np.ndarray
    ):
        """Integrate a generation path into the tree"""
        current_node_id = root_id
        current_tokens = []
        current_probs = []
        current_attention = []
        
        # Process tokens after shared prefix
        start_idx = len(self.graph.nodes[root_id]['node'].tokens)
        
        for idx in range(start_idx, len(path.tokens)):
            current_tokens.append(path.tokens[idx])
            current_probs.append(path.probabilities[idx])
            current_attention.append(path.attention_maps[idx])
            
            # Check if we should create a new node
            if self._should_split_node(
                current_tokens,
                current_node_id,
                path.divergence_points,
                idx
            ):
                # Create new node
                new_node = CGRTNode(
                    id=str(uuid4()),
                    tokens=current_tokens.copy(),
                    probabilities=current_probs.copy(),
                    attention_pattern=np.stack(current_attention),
                    importance_score=self._calculate_node_importance(
                        current_tokens,
                        current_probs,
                        current_attention,
                        global_attention
                    )
                )
                
                # Try to merge with existing nodes
                merge_target = self._find_merge_target(new_node)
                
                if merge_target:
                    # Update merge points and cross-links
                    self._merge_nodes(merge_target, new_node)
                    current_node_id = merge_target
                else:
                    # Add new node
                    self.graph.add_node(new_node.id, node=new_node)
                    self.graph.add_edge(
                        current_node_id,
                        new_node.id,
                        weight=new_node.cumulative_probability
                    )
                    current_node_id = new_node.id
                    
                # Reset current buffers
                current_tokens = []
                current_probs = []
                current_attention = []
    
    def _should_split_node(
        self,
        tokens: List[str],
        current_node_id: str,
        divergence_points: List[int],
        current_idx: int
    ) -> bool:
        """Determine if we should create a new node"""
        # Always split at divergence points
        if current_idx in divergence_points:
            return True
            
        # Split if sequence is getting too long
        if len(tokens) > 5:
            return True
            
        # Split if attention pattern changes significantly
        current_node = self.graph.nodes[current_node_id]['node']
        if self._attention_pattern_changed(
            current_node.attention_pattern,
            tokens
        ):
            return True
            
        return False
    
    def _find_merge_target(self, node: CGRTNode) -> Optional[str]:
        """Find existing node to merge with"""
        candidates = []
        
        for node_id in self.graph.nodes():
            existing_node = self.graph.nodes[node_id]['node']
            
            # Calculate similarity
            similarity = self._calculate_node_similarity(
                existing_node,
                node
            )
            
            if similarity >= self.merge_threshold:
                candidates.append((node_id, similarity))
                
        if candidates:
            # Return most similar candidate
            return max(candidates, key=lambda x: x[1])[0]
            
        return None
    
    def _merge_nodes(self, target_id: str, node: CGRTNode):
        """Merge two nodes"""
        target_node = self.graph.nodes[target_id]['node']
        
        # Update merge points
        merge_point = len(target_node.tokens)
        target_node.merge_points.add(merge_point)
        
        # Combine attention patterns
        combined_attention = np.mean([
            target_node.attention_pattern,
            node.attention_pattern
        ], axis=0)
        
        # Update target node
        target_node.attention_pattern = combined_attention
        target_node.importance_score = max(
            target_node.importance_score,
            node.importance_score
        )
        
        # Add metadata about merge
        target_node.metadata['merges'] = target_node.metadata.get('merges', 0) + 1
    
    def _add_cross_links(self, global_attention: np.ndarray):
        """Add cross-links between related nodes"""
        for node1_id in self.graph.nodes():
            node1 = self.graph.nodes[node1_id]['node']
            
            candidates = []
            for node2_id in self.graph.nodes():
                if node1_id != node2_id:
                    node2 = self.graph.nodes[node2_id]['node']
                    
                    # Calculate cross-node relationship
                    relationship = self._calculate_cross_relationship(
                        node1,
                        node2,
                        global_attention
                    )
                    
                    if relationship > self.importance_threshold:
                        candidates.append((node2_id, relationship))
                        
            # Add top cross-links
            for node2_id, strength in sorted(
                candidates,
                key=lambda x: x[1],
                reverse=True
            )[:self.max_cross_links]:
                node1.cross_links.add(node2_id)
                self.graph.add_edge(
                    node1_id,
                    node2_id,
                    type='cross',
                    weight=strength
                )
    
    def _calculate_node_importance(
        self,
        tokens: List[str],
        probabilities: List[float],
        attention_maps: List[np.ndarray],
        global_attention: np.ndarray
    ) -> float:
        """Calculate importance score for a node"""
        # Probability importance
        prob_importance = np.mean(probabilities)
        
        # Attention importance
        attention_importance = np.mean([
            np.mean(attention_map) for attention_map in attention_maps
        ])
        
        # Global attention alignment
        global_alignment = 1 - cosine(
            attention_maps[-1].flatten(),
            global_attention[-1].flatten()
        )
        
        return (
            0.4 * prob_importance +
            0.3 * attention_importance +
            0.3 * global_alignment
        )
    
    def _calculate_node_similarity(
        self,
        node1: CGRTNode,
        node2: CGRTNode
    ) -> float:
        """Calculate similarity between two nodes"""
        # Token sequence similarity
        token_sim = len(set(node1.tokens) & set(node2.tokens)) / len(set(node1.tokens) | set(node2.tokens))
        
        # Probability similarity
        prob_sim = 1 - abs(
            node1.cumulative_probability - node2.cumulative_probability
        )
        
        # Attention pattern similarity
        attention_sim = 1 - cosine(
            node1.attention_pattern.flatten(),
            node2.attention_pattern.flatten()
        )
        
        return 0.4 * token_sim + 0.3 * prob_sim + 0.3 * attention_sim
    
    def _calculate_cross_relationship(
        self,
        node1: CGRTNode,
        node2: CGRTNode,
        global_attention: np.ndarray
    ) -> float:
        """Calculate relationship strength between nodes"""
        # Attention flow alignment
        attention_alignment = 1 - cosine(
            node1.attention_pattern.flatten(),
            node2.attention_pattern.flatten()
        )
        
        # Global attention contribution
        global_contrib = np.mean([
            1 - cosine(
                node1.attention_pattern.flatten(),
                global_attention.flatten()
            ),
            1 - cosine(
                node2.attention_pattern.flatten(),
                global_attention.flatten()
            )
        ])
        
        # Path relationship
        path_relationship = self._calculate_path_relationship(node1, node2)
        
        return (
            0.4 * attention_alignment +
            0.3 * global_contrib +
            0.3 * path_relationship
        )
    
    def _calculate_path_relationship(
        self,
        node1: CGRTNode,
        node2: CGRTNode
    ) -> float:
        """Calculate relationship between node paths"""
        # Find paths containing nodes
        paths1 = set()
        paths2 = set()
        
        for path_id, path_nodes in self.path_cache.items():
            if node1.id in path_nodes:
                paths1.add(path_id)
            if node2.id in path_nodes:
                paths2.add(path_id)
                
        # Calculate path overlap
        if not paths1 or not paths2:
            return 0.0
            
        return len(paths1 & paths2) / len(paths1 | paths2)
    
    def _prune_branches(self):
        """Remove low-importance branches"""
        to_remove = []
        
        for node_id in self.graph.nodes():
            node = self.graph.nodes[node_id]['node']
            
            # Don't remove root
            if self.graph.in_degree(node_id) == 0:
                continue
                
            # Check importance threshold
            if node.importance_score < self.importance_threshold:
                # Only remove if no high-importance descendants
                if not self._has_important_descendants(node_id):
                    to_remove.append(node_id)
                    
        for node_id in to_remove:
            self.graph.remove_node(node_id)
            
    def _has_important_descendants(self, node_id: str) -> bool:
        """Check if node has any high-importance descendants"""
        for descendant in nx.descendants(self.graph, node_id):
            if self.graph.nodes[descendant]['node'].importance_score >= self.importance_threshold:
                return True
        return False