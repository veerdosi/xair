import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from transformers import AutoTokenizer, AutoModel
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from uuid import uuid4
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from functools import lru_cache

from llm_interface import LLMInterface
from reasoning_tree import ReasoningTreeGenerator, TreeNode

#############################################
# SECTION 1: VCNet-BASED COUNTERFACTUAL GENERATION
#############################################

# --- VCNet and Associated Classes ---

class TextEncoder(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Base transformer encoder
        self.transformer = AutoModel.from_pretrained('bert-base-uncased')
        
        # Latent projections
        self.content_proj = nn.Linear(768, hidden_dim)
        self.structure_proj = nn.Linear(768, hidden_dim)
        self.logic_proj = nn.Linear(768, hidden_dim)
        
        # Distribution parameters
        self.content_mu = nn.Linear(hidden_dim, latent_dim)
        self.content_logvar = nn.Linear(hidden_dim, latent_dim)
        self.structure_mu = nn.Linear(hidden_dim, latent_dim)
        self.structure_logvar = nn.Linear(hidden_dim, latent_dim)
        self.logic_mu = nn.Linear(hidden_dim, latent_dim)
        self.logic_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, input_ids, attention_mask):
        # Get transformer embeddings
        outputs = self.transformer(input_ids, attention_mask)
        hidden = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Project to different aspects
        content_hidden = self.content_proj(hidden)
        structure_hidden = self.structure_proj(hidden)
        logic_hidden = self.logic_proj(hidden)
        
        # Get distribution parameters
        content_mu = self.content_mu(content_hidden)
        content_logvar = self.content_logvar(content_hidden)
        structure_mu = self.structure_mu(structure_hidden)
        structure_logvar = self.structure_logvar(structure_hidden)
        logic_mu = self.logic_mu(logic_hidden)
        logic_logvar = self.logic_logvar(logic_hidden)
        
        return {
            'content': (content_mu, content_logvar),
            'structure': (structure_mu, structure_logvar),
            'logic': (logic_mu, logic_logvar)
        }

class TextDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256, vocab_size=30522):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Latent processors
        self.content_proc = nn.Linear(latent_dim, hidden_dim)
        self.structure_proc = nn.Linear(latent_dim, hidden_dim)
        self.logic_proc = nn.Linear(latent_dim, hidden_dim)
        
        # Transformer decoder layers
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim * 3,
            nhead=8
        )
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=6
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 3, vocab_size)

    def forward(self, z_content, z_structure, z_logic, tgt_mask=None):
        # Process latent vectors
        content_hidden = self.content_proc(z_content)
        structure_hidden = self.structure_proc(z_structure)
        logic_hidden = self.logic_proc(z_logic)
        
        # Combine aspects
        combined = torch.cat(
            [content_hidden, structure_hidden, logic_hidden],
            dim=-1
        )
        
        # Generate sequence (unsqueezing to add sequence dimension)
        output = self.decoder(combined.unsqueeze(0), tgt_mask=tgt_mask)
        logits = self.output_proj(output)
        
        return logits

class VCNet(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.encoder = TextEncoder(hidden_dim, latent_dim)
        self.decoder = TextDecoder(latent_dim, hidden_dim)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_ids, attention_mask, target_ids=None):
        # Encode
        distributions = self.encoder(input_ids, attention_mask)
        
        # Sample latent vectors
        z_content = self.reparameterize(*distributions['content'])
        z_structure = self.reparameterize(*distributions['structure'])
        z_logic = self.reparameterize(*distributions['logic'])
        
        # Create target mask if needed
        tgt_mask = None
        if target_ids is not None:
            tgt_mask = self.generate_square_subsequent_mask(target_ids.size(1))
            
        # Decode
        logits = self.decoder(z_content, z_structure, z_logic, tgt_mask)
        
        # Calculate losses if training
        if target_ids is not None:
            reconstruction_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            # KL divergence for each aspect
            kl_content = kl_divergence(Normal(*distributions['content'][0], torch.exp(0.5*distributions['content'][1])),
                                       Normal(torch.zeros_like(distributions['content'][0]), torch.ones_like(distributions['content'][1])))
            kl_structure = kl_divergence(Normal(*distributions['structure'][0], torch.exp(0.5*distributions['structure'][1])),
                                         Normal(torch.zeros_like(distributions['structure'][0]), torch.ones_like(distributions['structure'][1])))
            kl_logic = kl_divergence(Normal(*distributions['logic'][0], torch.exp(0.5*distributions['logic'][1])),
                                     Normal(torch.zeros_like(distributions['logic'][0]), torch.ones_like(distributions['logic'][1])))
            
            kl_loss = kl_content + kl_structure + kl_logic
            
            return {
                'loss': reconstruction_loss + kl_loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss
            }
            
        return logits

    def generate_counterfactual(self, text, aspect='all', strength=1.0):
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        
        # Get latent distributions
        with torch.no_grad():
            distributions = self.encoder(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            
        # Sample and modify latent vectors
        z_content = self.reparameterize(*distributions['content'])
        z_structure = self.reparameterize(*distributions['structure'])
        z_logic = self.reparameterize(*distributions['logic'])
        
        # Modify based on aspect
        if aspect in ['content', 'all']:
            z_content = self.perturb_latent(z_content, strength)
        if aspect in ['structure', 'all']:
            z_structure = self.perturb_latent(z_structure, strength)
        if aspect in ['logic', 'all']:
            z_logic = self.perturb_latent(z_logic, strength)
            
        # Generate counterfactual
        logits = self.decoder(z_content, z_structure, z_logic)
        tokens = torch.argmax(logits, dim=-1)
        
        return self.tokenizer.decode(tokens[0])

    def perturb_latent(self, z, strength=1.0):
        # Add controlled noise to latent vector
        noise = torch.randn_like(z) * strength
        return z + noise

# --- VCNet Counterfactual Analysis Components ---

class GoalSpecification:
    def __init__(self, target_aspects, constraints):
        self.target_aspects = target_aspects
        self.constraints = constraints
        
    def evaluate(self, original, counterfactual):
        scores = {}
        # For each target aspect, compute a dummy score (in practice, use your metric)
        for aspect, target in self.target_aspects.items():
            scores[aspect] = abs(len(counterfactual) - len(original)) / (len(original) + 1)
        for constraint, threshold in self.constraints.items():
            scores[constraint] = 1.0 if scores.get('content', 0) < threshold else 0.0
        return scores

class CounterfactualFilter:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # In practice, you might use another semantic model
        self.semantic_model = AutoModel.from_pretrained('bert-base-uncased')
        
    def filter_candidates(self, original, candidates, goal_spec):
        filtered = []
        for candidate in candidates:
            # Basic validity check (dummy implementation)
            if len(candidate.split()) < 3:
                continue
            scores = goal_spec.evaluate(original, candidate)
            # Only accept if all (dummy) constraint scores are above a threshold
            if all(score > 0.1 for score in scores.values()):
                filtered.append((candidate, scores))
        return filtered

class CounterfactualRanker:
    def __init__(self):
        self.diversity_weight = 0.3
        self.effectiveness_weight = 0.7
        
    def calculate_effectiveness(self, scores):
        return sum(score * self.effectiveness_weight for score in scores.values())

    def select_most_diverse(self, candidates, selected):
        if not selected:
            return candidates[0]
        max_diversity = -float('inf')
        most_diverse = None
        for candidate in candidates:
            # Dummy diversity: longer text is assumed more “different”
            diversity = len(candidate[0])
            if diversity > max_diversity:
                max_diversity = diversity
                most_diverse = candidate
        return most_diverse

    def rank_counterfactuals(self, filtered_candidates):
        if not filtered_candidates:
            return []
        candidates = sorted(filtered_candidates, key=lambda x: self.calculate_effectiveness(x[1]), reverse=True)
        selected = []
        while candidates and len(selected) < 5:
            best_candidate = self.select_most_diverse(candidates, selected)
            if best_candidate is None:
                break
            selected.append(best_candidate)
            candidates.remove(best_candidate)
        return selected

# --- Unified function using VCNet for counterfactual generation ---

def generate_counterfactuals(text, goal_spec):
    vcnet = VCNet()
    cf_filter = CounterfactualFilter(vcnet, vcnet.tokenizer)
    cf_ranker = CounterfactualRanker()
    
    # Generate candidates using different aspects and strengths
    candidates = []
    for aspect in ['content', 'structure', 'logic', 'all']:
        for strength in [0.5, 1.0, 1.5]:
            cf_text = vcnet.generate_counterfactual(text, aspect=aspect, strength=strength)
            candidates.append(cf_text)
            
    # Filter candidates
    filtered = cf_filter.filter_candidates(text, candidates, goal_spec)
    
    # Rank and select final counterfactuals
    ranked = cf_ranker.rank_counterfactuals(filtered)
    
    return ranked

#############################################
# SECTION 2: LLM-BASED COUNTERFACTUAL GENERATOR & INTEGRATOR
#############################################

# --- Data Structures for LLM Counterfactuals ---

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
            min_key = min(self.cache.items(), key=lambda x: x[1].effectiveness_score())
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

# --- LLM-based Counterfactual Generator ---

class CounterfactualGenerator:
    def __init__(
        self,
        llm: LLMInterface,
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
    
    async def generate_counterfactuals(self, reasoning_tree: nx.DiGraph) -> List[Counterfactual]:
        """Generate counterfactuals for all significant nodes in the tree"""
        counterfactuals = []
        # Process nodes in breadth-first order
        for node_id in nx.bfs_tree(reasoning_tree, list(reasoning_tree.nodes())[0]):
            node = reasoning_tree.nodes[node_id]['node']
            # Skip nodes with low attention or probability
            if (node.attention_weight < self.min_probability or 
                node.probability < self.min_probability):
                continue
            new_counterfactuals = await self._generate_node_counterfactuals(node, reasoning_tree)
            counterfactuals.extend(new_counterfactuals)
        return counterfactuals
    
    async def _generate_node_counterfactuals(self, node: TreeNode, tree: nx.DiGraph) -> List[Counterfactual]:
        counterfactuals = []
        context = self._get_node_context(node, tree)
        modifications = await asyncio.gather(
            self._generate_token_substitutions(context),
            self._generate_context_modifications(context),
            self._generate_semantic_alternatives(context)
        )
        for mod_list in modifications:
            for mod in mod_list:
                cf = await self._create_counterfactual(
                    original_text=context,
                    modified_text=mod['text'],
                    modification_type=mod['type'],
                    node=node
                )
                if cf and not self._is_too_similar(cf, counterfactuals):
                    counterfactuals.append(cf)
                if len(counterfactuals) >= self.max_counterfactuals_per_node:
                    break
        return counterfactuals
    
    def _get_node_context(self, node: TreeNode, tree: nx.DiGraph) -> str:
        path = []
        current = node
        while current.parent_id is not None:
            path.append(current.text)
            current = tree.nodes[current.parent_id]['node']
        path.append(current.text)  # Add root node
        return " ".join(reversed(path))
    
    async def _generate_token_substitutions(self, context: str) -> List[Dict[str, str]]:
        try:
            token_probs = await self.llm.get_token_probabilities(context)
            significant_tokens = {token: prob for token, prob in token_probs.items() if prob >= self.min_probability}
            modifications = []
            for token in significant_tokens:
                prompt = f"Replace '{token}' in this context with a different word that changes the meaning: {context}"
                response = await self.llm.query(prompt)
                modifications.append({'text': response.text, 'type': 'token_substitution'})
            return modifications
        except Exception as e:
            print(f"Error generating token substitutions: {e}")
            return []
    
    async def _generate_context_modifications(self, context: str) -> List[Dict[str, str]]:
        try:
            prompts = [
                f"Modify this text to lead to a different conclusion: {context}",
                f"What's an alternative version of this that changes the outcome: {context}",
                f"Rewrite this text to explore a different possibility: {context}"
            ]
            responses = await asyncio.gather(*(self.llm.query(prompt) for prompt in prompts))
            return [{'text': resp.text, 'type': 'context_modification'} for resp in responses]
        except Exception as e:
            print(f"Error generating context modifications: {e}")
            return []
    
    async def _generate_semantic_alternatives(self, context: str) -> List[Dict[str, str]]:
        try:
            prompts = [
                f"Express this idea in a completely different way: {context}",
                f"What's a contrasting perspective on this: {context}"
            ]
            responses = await asyncio.gather(*(self.llm.query(prompt) for prompt in prompts))
            return [{'text': resp.text, 'type': 'semantic_alternative'} for resp in responses]
        except Exception as e:
            print(f"Error generating semantic alternatives: {e}")
            return []
    
    async def _create_counterfactual(self, original_text: str, modified_text: str, modification_type: str, node: TreeNode) -> Optional[Counterfactual]:
        try:
            response = await self.llm.query(modified_text)
            attention_flow = await self.llm.get_attention_flow(modified_text)
            attention_score = float(np.mean(attention_flow))
            cf = Counterfactual(
                id=str(uuid4()),
                original_text=original_text,
                modified_text=modified_text,
                modification_type=modification_type,
                target_outcome="alternative_path",
                actual_outcome=response.text,
                probability=response.logits[0] if response.logits is not None else 0.0,
                attention_score=attention_score,
                parent_node_id=node.id,
                embedding=self.get_embedding(modified_text)
            )
            self.cache.add(cf)
            return cf
        except Exception as e:
            print(f"Error creating counterfactual: {e}")
            return None
    
    def _is_too_similar(self, new_cf: Counterfactual, existing_cfs: List[Counterfactual]) -> bool:
        if not existing_cfs:
            return False
        for cf in existing_cfs:
            if cf.embedding is not None and new_cf.embedding is not None:
                similarity = cosine_similarity([cf.embedding], [new_cf.embedding])[0][0]
                if similarity > self.similarity_threshold:
                    return True
        return False
