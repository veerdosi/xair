from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
import networkx as nx
from uuid import uuid4
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.preprocessing import MinMaxScaler
import asyncio
from functools import lru_cache
import json

from llm_interface import DeepSeekInterface, LLMResponse
from reasoning_tree import ReasoningTreeGenerator, TreeNode
from counterfactual_generator import CounterfactualGenerator, Counterfactual
from counterfactual_integrator import CounterfactualIntegrator, IntegrationPoint

@dataclass
class ImpactScore:
    counterfactual_id: str
    local_impact: float
    global_impact: float
    structural_impact: float
    plausibility: float
    composite_score: float
    confidence: float
    contributing_factors: Dict[str, float]
    affected_nodes: Set[str]

class ImpactCache:
    def __init__(self, max_size: int = 1000):
        self.scores: Dict[str, ImpactScore] = {}
        self.max_size = max_size
        self.dirty = False
        
    def add(self, cf_id: str, score: ImpactScore):
        if len(self.scores) >= self.max_size:
            # Remove lowest impact score
            min_key = min(self.scores.items(), 
                         key=lambda x: x[1].composite_score)
            del self.scores[min_key[0]]
        self.scores[cf_id] = score
        self.dirty = True
        
    def get(self, cf_id: str) -> Optional[ImpactScore]:
        return self.scores.get(cf_id)

class ImpactAnalyzer:
    def __init__(
        self,
        llm: DeepSeekInterface,
        tree_generator: ReasoningTreeGenerator,
        cf_generator: CounterfactualGenerator,
        cf_integrator: CounterfactualIntegrator,
        cache_size: int = 1000,
        min_impact_threshold: float = 0.1,
        plausibility_threshold: float = 0.3,
        batch_size: int = 10
    ):
        self.llm = llm
        self.tree_generator = tree_generator
        self.cf_generator = cf_generator
        self.cf_integrator = cf_integrator
        self.cache = ImpactCache(cache_size)
        self.min_impact_threshold = min_impact_threshold
        self.plausibility_threshold = plausibility_threshold
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        
    async def analyze_impacts(
        self,
        tree: nx.DiGraph,
        counterfactuals: List[Counterfactual]
    ) -> List[ImpactScore]:
        """Analyze the impact of counterfactuals on the reasoning tree"""
        impact_scores = []
        
        # Process counterfactuals in batches
        for i in range(0, len(counterfactuals), self.batch_size):
            batch = counterfactuals[i:i + batch_size]
            batch_scores = await asyncio.gather(*(
                self._analyze_single_impact(tree, cf)
                for cf in batch
            ))
            impact_scores.extend(batch_scores)
            
        # Normalize scores across all counterfactuals
        self._normalize_scores(impact_scores)
        
        # Cache results
        for score in impact_scores:
            self.cache.add(score.counterfactual_id, score)
            
        return impact_scores
    
    async def _analyze_single_impact(
        self,
        tree: nx.DiGraph,
        counterfactual: Counterfactual
    ) -> ImpactScore:
        """Analyze the impact of a single counterfactual"""
        # Check cache first
        cached_score = self.cache.get(counterfactual.id)
        if cached_score and not self.cache.dirty:
            return cached_score
            
        # Calculate different impact components
        local_impact = await self._calculate_local_impact(
            tree,
            counterfactual
        )
        
        global_impact = await self._calculate_global_impact(
            tree,
            counterfactual
        )
        
        structural_impact = self._calculate_structural_impact(
            tree,
            counterfactual
        )
        
        plausibility = await self._calculate_plausibility(
            counterfactual
        )
        
        # Get affected nodes
        affected_nodes = self._find_affected_nodes(
            tree,
            counterfactual
        )
        
        # Calculate composite score
        contributing_factors = {
            "local_impact": local_impact,
            "global_impact": global_impact,
            "structural_impact": structural_impact,
            "plausibility": plausibility,
        }
        
        composite_score = self._calculate_composite_score(
            contributing_factors
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            contributing_factors,
            len(affected_nodes)
        )
        
        return ImpactScore(
            counterfactual_id=counterfactual.id,
            local_impact=local_impact,
            global_impact=global_impact,
            structural_impact=structural_impact,
            plausibility=plausibility,
            composite_score=composite_score,
            confidence=confidence,
            contributing_factors=contributing_factors,
            affected_nodes=affected_nodes
        )
    
    async def _calculate_local_impact(
        self,
        tree: nx.DiGraph,
        counterfactual: Counterfactual
    ) -> float:
        """Calculate the immediate impact of the counterfactual"""
        # Get original node
        original_node_id = self.cf_integrator.overlay.integration_points[
            counterfactual.id
        ].original_node_id
        original_node = tree.nodes[original_node_id]['node']
        
        # Calculate semantic difference
        semantic_diff = 1 - self.cf_generator._calculate_similarity(
            self.cf_generator.get_embedding(original_node.text),
            self.cf_generator.get_embedding(counterfactual.modified_text)
        )
        
        # Calculate probability impact
        prob_diff = abs(original_node.probability - counterfactual.probability)
        
        # Calculate attention impact
        attention_diff = abs(
            original_node.attention_weight - counterfactual.attention_score
        )
        
        # Combine scores
        local_impact = (
            0.4 * semantic_diff +
            0.3 * prob_diff +
            0.3 * attention_diff
        )
        
        return float(local_impact)
    
    async def _calculate_global_impact(
        self,
        tree: nx.DiGraph,
        counterfactual: Counterfactual
    ) -> float:
        """Calculate the impact on final outcomes"""
        # Get original outcome
        original_path = self.tree_generator.get_most_likely_path()
        original_outcome = tree.nodes[original_path[-1]]['node'].text
        
        # Get counterfactual outcome
        cf_path = None
        for path in nx.all_simple_paths(tree, 
                                      source=list(tree.nodes())[0],
                                      target=counterfactual.id):
            if cf_path is None or len(path) < len(cf_path):
                cf_path = path
                
        if not cf_path:
            return 0.0
            
        # Calculate outcome difference
        outcome_diff = 1 - self.cf_generator._calculate_similarity(
            self.cf_generator.get_embedding(original_outcome),
            self.cf_generator.get_embedding(counterfactual.actual_outcome)
        )
        
        # Calculate path probability difference
        original_prob = np.prod([
            tree.nodes[n]['node'].probability
            for n in original_path
        ])
        
        cf_prob = np.prod([
            tree.nodes[n]['node'].probability
            for n in cf_path
        ])
        
        prob_diff = abs(original_prob - cf_prob)
        
        # Combine scores
        global_impact = 0.6 * outcome_diff + 0.4 * prob_diff
        return float(global_impact)
    
    def _calculate_structural_impact(
        self,
        tree: nx.DiGraph,
        counterfactual: Counterfactual
    ) -> float:
        """Calculate the impact on tree structure"""
        # Get original branch structure
        original_structure = self._get_branch_structure(
            tree,
            list(tree.nodes())[0]  # root
        )
        
        # Get counterfactual branch structure
        cf_structure = self._get_branch_structure(
            tree,
            counterfactual.id
        )
        
        # Calculate structural differences
        branching_diff = abs(
            len(original_structure) - len(cf_structure)
        ) / max(len(original_structure), 1)
        
        depth_diff = abs(
            max(original_structure.values()) - 
            max(cf_structure.values())
        ) / max(max(original_structure.values()), 1)
        
        # Calculate path divergence
        path_divergence = self._calculate_path_divergence(
            tree,
            counterfactual
        )
        
        # Combine scores
        structural_impact = (
            0.3 * branching_diff +
            0.3 * depth_diff +
            0.4 * path_divergence
        )
        
        return float(structural_impact)
    
    async def _calculate_plausibility(
        self,
        counterfactual: Counterfactual
    ) -> float:
        """Calculate how plausible the counterfactual is"""
        # Generate plausibility statements
        statements = [
            f"This is a realistic alternative: {counterfactual.modified_text}",
            f"This could actually happen: {counterfactual.modified_text}",
            f"This makes logical sense: {counterfactual.modified_text}"
        ]
        
        # Get LLM confidence in statements
        confidences = []
        for statement in statements:
            response = await self.llm.query(statement)
            if response.logits is not None:
                # Convert logits to probability
                conf = float(np.exp(response.logits[0]) / 
                           (1 + np.exp(response.logits[0])))
                confidences.append(conf)
                
        if not confidences:
            return 0.5  # Default if no confidences available
            
        # Average confidence scores
        plausibility = np.mean(confidences)
        return float(plausibility)
    
    def _get_branch_structure(
        self,
        tree: nx.DiGraph,
        start_node: str
    ) -> Dict[str, int]:
        """Get the branching structure starting from a node"""
        structure = {}
        for node in nx.descendants(tree, start_node):
            depth = nx.shortest_path_length(tree, start_node, node)
            structure[node] = depth
        return structure
    
    def _calculate_path_divergence(
        self,
        tree: nx.DiGraph,
        counterfactual: Counterfactual
    ) -> float:
        """Calculate how much paths diverge from original"""
        # Get original and counterfactual paths
        original_paths = list(nx.all_simple_paths(
            tree,
            source=list(tree.nodes())[0],
            target=counterfactual.id
        ))
        
        if not original_paths:
            return 1.0  # Maximum divergence if no path exists
            
        # Calculate average path difference
        path_diffs = []
        for path1 in original_paths:
            for path2 in nx.all_simple_paths(
                tree,
                source=list(tree.nodes())[0],
                target=counterfactual.id
            ):
                # Calculate path similarity using longest common subsequence
                lcs_length = self._longest_common_subsequence(path1, path2)
                diff = 1 - (lcs_length / max(len(path1), len(path2)))
                path_diffs.append(diff)
                
        return float(np.mean(path_diffs)) if path_diffs else 1.0
    def _longest_common_subsequence(
        self,
        seq1: List[str],
        seq2: List[str]
    ) -> int:
        """Calculate length of longest common subsequence between two paths"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        return dp[m][n]
    
    def _find_affected_nodes(
        self,
        tree: nx.DiGraph,
        counterfactual: Counterfactual
    ) -> Set[str]:
        """Find all nodes affected by the counterfactual"""
        affected = set()
        
        # Get integration point
        integration_point = self.cf_integrator.overlay.integration_points[
            counterfactual.id
        ]
        
        # Add all nodes reachable from counterfactual
        affected.update(integration_point.reachable_nodes)
        
        # Add nodes whose probabilities might be affected
        for node_id in nx.descendants(tree, integration_point.original_node_id):
            node = tree.nodes[node_id]['node']
            if node.probability > self.min_impact_threshold:
                affected.add(node_id)
                
        return affected
    
    def _calculate_composite_score(
        self,
        factors: Dict[str, float]
    ) -> float:
        """Calculate overall impact score from contributing factors"""
        weights = {
            "local_impact": 0.3,
            "global_impact": 0.3,
            "structural_impact": 0.2,
            "plausibility": 0.2
        }
        
        weighted_sum = sum(
            weights[factor] * score
            for factor, score in factors.items()
        )
        
        return float(weighted_sum)
    
    def _calculate_confidence(
        self,
        factors: Dict[str, float],
        num_affected_nodes: int
    ) -> float:
        """Calculate confidence in impact assessment"""
        # Consider factor variance
        factor_variance = np.var(list(factors.values()))
        
        # Consider number of affected nodes (more nodes = less confidence)
        node_factor = 1.0 / (1.0 + np.log1p(num_affected_nodes))
        
        # Consider plausibility
        plausibility = factors.get("plausibility", 0.5)
        
        confidence = (
            0.4 * (1.0 - factor_variance) +
            0.3 * node_factor +
            0.3 * plausibility
        )
        
        return float(confidence)
    
    def _normalize_scores(self, scores: List[ImpactScore]):
        """Normalize impact scores across all counterfactuals"""
        if not scores:
            return
            
        # Extract scores for normalization
        local_impacts = [s.local_impact for s in scores]
        global_impacts = [s.global_impact for s in scores]
        structural_impacts = [s.structural_impact for s in scores]
        plausibilities = [s.plausibility for s in scores]
        composite_scores = [s.composite_score for s in scores]
        
        # Create feature matrix
        features = np.column_stack([
            local_impacts,
            global_impacts,
            structural_impacts,
            plausibilities,
            composite_scores
        ])
        
        # Normalize features
        normalized = self.scaler.fit_transform(features)
        
        # Update scores with normalized values
        for i, score in enumerate(scores):
            score.local_impact = float(normalized[i, 0])
            score.global_impact = float(normalized[i, 1])
            score.structural_impact = float(normalized[i, 2])
            score.plausibility = float(normalized[i, 3])
            score.composite_score = float(normalized[i, 4])
    
    def get_ranked_impacts(
        self,
        scores: List[ImpactScore],
        min_confidence: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Get counterfactuals ranked by impact, filtered by confidence"""
        # Filter by confidence
        confident_scores = [
            score for score in scores
            if score.confidence >= min_confidence
        ]
        
        # Sort by composite score
        ranked = sorted(
            confident_scores,
            key=lambda x: x.composite_score,
            reverse=True
        )
        
        return [
            (score.counterfactual_id, score.composite_score)
            for score in ranked
        ]
    
    def get_impact_explanation(
        self,
        score: ImpactScore
    ) -> Dict[str, any]:
        """Generate human-readable explanation of impact analysis"""
        return {
            "counterfactual_id": score.counterfactual_id,
            "overall_impact": score.composite_score,
            "confidence": score.confidence,
            "contributing_factors": {
                "Local Impact": {
                    "score": score.local_impact,
                    "description": "Immediate changes to the decision point"
                },
                "Global Impact": {
                    "score": score.global_impact,
                    "description": "Effects on final outcomes"
                },
                "Structural Impact": {
                    "score": score.structural_impact,
                    "description": "Changes to reasoning structure"
                },
                "Plausibility": {
                    "score": score.plausibility,
                    "description": "How realistic the alternative is"
                }
            },
            "affected_nodes": len(score.affected_nodes),
            "key_insights": self._generate_key_insights(score)
        }
    
    def _generate_key_insights(
        self,
        score: ImpactScore
    ) -> List[str]:
        """Generate key insights about the impact"""
        insights = []
        
        # Add insights based on scores
        if score.local_impact > 0.7:
            insights.append(
                "Makes significant immediate changes to the decision point"
            )
        
        if score.global_impact > 0.7:
            insights.append(
                "Leads to substantially different final outcomes"
            )
            
        if score.structural_impact > 0.7:
            insights.append(
                "Causes major changes in reasoning structure"
            )
            
        if score.plausibility > 0.7:
            insights.append(
                "Represents a highly plausible alternative"
            )
            
        if score.confidence < 0.5:
            insights.append(
                "Impact assessment has significant uncertainty"
            )
            
        return insights

# Example usage:
async def main():
    async with DeepSeekInterface(api_key="your_api_key_here") as llm:
        # Initialize components
        tree_generator = ReasoningTreeGenerator(llm)
        cf_generator = CounterfactualGenerator(llm, tree_generator)
        cf_integrator = CounterfactualIntegrator(tree_generator, cf_generator)
        analyzer = ImpactAnalyzer(llm, tree_generator, cf_generator, cf_integrator)
        
        # Generate base reasoning tree
        base_tree = await tree_generator.generate_tree(
            "The capital of France is Paris"
        )
        
        # Generate counterfactuals
        counterfactuals = await cf_generator.generate_counterfactuals(
            base_tree
        )
        
        # Analyze impacts
        impact_scores = await analyzer.analyze_impacts(
            base_tree,
            counterfactuals
        )
        
        # Get ranked impacts
        ranked_impacts = analyzer.get_ranked_impacts(impact_scores)
        print(f"Ranked impacts: {ranked_impacts}")
        
        # Get explanation for top impact
        if impact_scores:
            explanation = analyzer.get_impact_explanation(impact_scores[0])
            print(f"Top impact explanation: {json.dumps(explanation, indent=2)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())