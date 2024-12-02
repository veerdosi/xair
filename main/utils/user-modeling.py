# utils/user_modeling.py

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

@dataclass
class UserInteraction:
    timestamp: datetime
    query_type: str
    complexity_level: float
    interaction_duration: float
    feedback_score: Optional[float]
    expertise_signals: Dict[str, float]

class UserProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.interactions: List[UserInteraction] = []
        self.expertise_scores = {
            'technical': 0.0,
            'domain': 0.0,
            'interaction': 0.0
        }
        self.preferences = {
            'visual_style': 'detailed',
            'explanation_depth': 'intermediate',
            'interaction_mode': 'guided'
        }
        self.last_update = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'expertise_scores': self.expertise_scores,
            'preferences': self.preferences,
            'last_update': self.last_update.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        profile = cls(data['user_id'])
        profile.expertise_scores = data['expertise_scores']
        profile.preferences = data['preferences']
        profile.last_update = datetime.fromisoformat(data['last_update'])
        return profile

class UserModelingSystem:
    def __init__(self, storage_path: Path = Path('data/user_profiles')):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.active_profiles: Dict[str, UserProfile] = {}
        
        # Expertise assessment weights
        self.expertise_weights = {
            'query_complexity': 0.3,
            'interaction_time': 0.2,
            'feedback_quality': 0.3,
            'historical_performance': 0.2
        }
    
    def get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.active_profiles:
            profile_path = self.storage_path / f"{user_id}.json"
            if profile_path.exists():
                with open(profile_path, 'r') as f:
                    profile_data = json.load(f)
                self.active_profiles[user_id] = UserProfile.from_dict(profile_data)
            else:
                self.active_profiles[user_id] = UserProfile(user_id)
        return self.active_profiles[user_id]
    
    def update_profile(self, user_id: str, interaction: UserInteraction):
        """Update user profile based on new interaction"""
        profile = self.get_user_profile(user_id)
        profile.interactions.append(interaction)
        
        # Update expertise scores
        self._update_expertise_scores(profile, interaction)
        
        # Update preferences
        self._update_preferences(profile, interaction)
        
        # Save updated profile
        self._save_profile(profile)
    
    def get_explanation_level(self, user_id: str) -> int:
        """Determine appropriate explanation level for user"""
        profile = self.get_user_profile(user_id)
        
        # Calculate weighted expertise score
        expertise_score = sum(profile.expertise_scores.values()) / len(profile.expertise_scores)
        
        # Map to discrete levels
        if expertise_score < 0.3:
            return 0  # Beginner
        elif expertise_score < 0.7:
            return 1  # Intermediate
        else:
            return 2  # Expert
    
    def _update_expertise_scores(self, profile: UserProfile, interaction: UserInteraction):
        """Update expertise scores based on interaction"""
        # Update technical expertise
        profile.expertise_scores['technical'] = self._calculate_technical_expertise(
            profile.expertise_scores['technical'],
            interaction
        )
        
        # Update domain expertise
        profile.expertise_scores['domain'] = self._calculate_domain_expertise(
            profile.expertise_scores['domain'],
            interaction
        )
        
        # Update interaction expertise
        profile.expertise_scores['interaction'] = self._calculate_interaction_expertise(
            profile.expertise_scores['interaction'],
            interaction
        )
    
    def _calculate_technical_expertise(self, current_score: float, interaction: UserInteraction) -> float:
        """Calculate updated technical expertise score"""
        weights = self.expertise_weights
        
        # Calculate new score components
        complexity_score = interaction.complexity_level * weights['query_complexity']
        time_score = min(1.0, interaction.interaction_duration / 300.0) * weights['interaction_time']
        feedback_score = (interaction.feedback_score or 0.5) * weights['feedback_quality']
        
        # Combine with current score
        new_score = (current_score + complexity_score + time_score + feedback_score) / 2
        return min(1.0, max(0.0, new_score))
    
    def _calculate_domain_expertise(self, current_score: float, interaction: UserInteraction) -> float:
        """Calculate updated domain expertise score"""
        domain_signals = interaction.expertise_signals.get('domain', 0.5)
        
        # Combine with current score using exponential moving average
        alpha = 0.3  # Learning rate
        new_score = (1 - alpha) * current_score + alpha * domain_signals
        return min(1.0, max(0.0, new_score))
    
    def _calculate_interaction_expertise(self, current_score: float, interaction: UserInteraction) -> float:
        """Calculate updated interaction expertise score"""
        interaction_quality = interaction.expertise_signals.get('interaction_quality', 0.5)
        
        # Update score based on interaction quality
        alpha = 0.2  # Learning rate
        new_score = (1 - alpha) * current_score + alpha * interaction_quality
        return min(1.0, max(0.0, new_score))
    
    def _update_preferences(self, profile: UserProfile, interaction: UserInteraction):
        """Update user preferences based on interaction"""
        # Update visual style preference
        if interaction.expertise_signals.get('visual_preference'):
            profile.preferences['visual_style'] = 'detailed' if interaction.expertise_signals['visual_preference'] > 0.5 else 'simple'
        
        # Update explanation depth preference
        expertise_level = self.get_explanation_level(profile.user_id)
        profile.preferences['explanation_depth'] = ['basic', 'intermediate', 'advanced'][expertise_level]
        
        # Update interaction mode preference
        if interaction.expertise_signals.get('guidance_needed', 0.5) > 0.7:
            profile.preferences['interaction_mode'] = 'guided'
        else:
            profile.preferences['interaction_mode'] = 'advanced'
    
    def _save_profile(self, profile: UserProfile):
        """Save user profile to storage"""
        profile_path = self.storage_path / f"{profile.user_id}.json"
        with open(profile_path, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)

# Integration utilities
class ExplanationAdapter:
    def __init__(self, user_modeling_system: UserModelingSystem):
        self.user_modeling = user_modeling_system
    
    def adapt_explanation(
        self,
        base_explanation: Dict[str, torch.Tensor],
        user_id: str
    ) -> Dict[str, torch.Tensor]:
        """Adapt explanation based on user profile"""
        profile = self.user_modeling.get_user_profile(user_id)
        expertise_level = self.user_modeling.get_explanation_level(user_id)
        
        # Adjust explanation components based on user preferences
        adapted_explanation = {}
        
        # Adapt base explanation
        adapted_explanation['base_explanation'] = self._adapt_base_explanation(
            base_explanation['base_explanation'],
            profile.preferences
        )
        
        # Add expertise-specific components
        if expertise_level >= 1:
            adapted_explanation.update(self._add_intermediate_components(base_explanation))
        
        if expertise_level >= 2:
            adapted_explanation.update(self._add_advanced_components(base_explanation))
        
        return adapted_explanation
    
    def _adapt_base_explanation(
        self,
        base_explanation: torch.Tensor,
        preferences: Dict[str, str]
    ) -> torch.Tensor:
        """Adapt base explanation based on user preferences"""
        # Implementation depends on specific explanation format
        return base_explanation
    
    def _add_intermediate_components(
        self,
        base_explanation: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add intermediate level explanation components"""
        return {
            'attention_patterns': base_explanation.get('attention_patterns', None),
            'uncertainty': base_explanation.get('uncertainty', None)
        }
    
    def _add_advanced_components(
        self,
        base_explanation: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add advanced level explanation components"""
        return {
            'detailed_attention': base_explanation.get('detailed_attention', None),
            'counterfactuals': base_explanation.get('counterfactuals', None)
        }
