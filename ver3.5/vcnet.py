import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from transformers import AutoTokenizer, AutoModel
import numpy as np

# VCNet Implementation
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
        
        # Generate sequence
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
        z_content = self.reparameterize(
            *distributions['content']
        )
        z_structure = self.reparameterize(
            *distributions['structure']
        )
        z_logic = self.reparameterize(
            *distributions['logic']
        )
        
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
            kl_content = self.kl_divergence(*distributions['content'])
            kl_structure = self.kl_divergence(*distributions['structure'])
            kl_logic = self.kl_divergence(*distributions['logic'])
            
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

# GYC Framework Implementation
class GoalSpecification:
    def __init__(self, target_aspects, constraints):
        self.target_aspects = target_aspects
        self.constraints = constraints
        
    def evaluate(self, original, counterfactual):
        scores = {}
        
        for aspect, target in self.target_aspects.items():
            scores[aspect] = self.evaluate_aspect(
                original,
                counterfactual,
                aspect,
                target
            )
            
        for constraint, threshold in self.constraints.items():
            scores[constraint] = self.evaluate_constraint(
                original,
                counterfactual,
                constraint,
                threshold
            )
            
        return scores

class CounterfactualFilter:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.semantic_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        
    def filter_candidates(self, original, candidates, goal_spec):
        filtered = []
        
        for candidate in candidates:
            # Check basic validity
            if not self.is_valid(candidate):
                continue
                
            # Evaluate against goals
            scores = goal_spec.evaluate(original, candidate)
            
            # Check if meets all constraints
            if self.meets_constraints(scores, goal_spec.constraints):
                filtered.append((candidate, scores))
                
        return filtered

    def is_valid(self, text):
        # Check basic validity (grammar, structure, etc.)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Check perplexity
        perplexity = torch.exp(
            F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                inputs['input_ids'].view(-1)
            )
        )
        
        return perplexity.item() < 100  # Threshold for validity

class CounterfactualRanker:
    def __init__(self):
        self.diversity_weight = 0.3
        self.effectiveness_weight = 0.7
        
    def rank_counterfactuals(self, filtered_candidates):
        if not filtered_candidates:
            return []
            
        ranked = []
        selected = []
        
        # Sort by effectiveness
        candidates = sorted(
            filtered_candidates,
            key=lambda x: self.calculate_effectiveness(x[1]),
            reverse=True
        )
        
        # Select diverse subset
        while candidates and len(selected) < 5:  # Top-5 diverse
            best_candidate = self.select_most_diverse(
                candidates,
                selected
            )
            if best_candidate is None:
                break
                
            selected.append(best_candidate)
            candidates.remove(best_candidate)
            
        return selected

    def calculate_effectiveness(self, scores):
        # Combine different aspects of effectiveness
        return sum(
            score * weight
            for score, weight in scores.items()
        )

    def select_most_diverse(self, candidates, selected):
        if not selected:
            return candidates[0]
            
        max_diversity = -float('inf')
        most_diverse = None
        
        for candidate in candidates:
            diversity = self.calculate_diversity(candidate, selected)
            
            if diversity > max_diversity:
                max_diversity = diversity
                most_diverse = candidate
                
        return most_diverse

# Example usage
def generate_counterfactuals(text, goal_spec):
    # Initialize components
    vcnet = VCNet()
    filter = CounterfactualFilter(vcnet, vcnet.tokenizer)
    ranker = CounterfactualRanker()
    
    # Generate candidates using different aspects and strengths
    candidates = []
    for aspect in ['content', 'structure', 'logic', 'all']:
        for strength in [0.5, 1.0, 1.5]:
            counterfactual = vcnet.generate_counterfactual(
                text,
                aspect=aspect,
                strength=strength
            )
            candidates.append(counterfactual)
            
    # Filter candidates
    filtered = filter.filter_candidates(text, candidates, goal_spec)
    
    # Rank and select final counterfactuals
    ranked = ranker.rank_counterfactuals(filtered)
    
    return ranked