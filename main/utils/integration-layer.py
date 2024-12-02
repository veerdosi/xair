# utils/integration.py

import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from utils.user_modeling import UserModelingSystem, ExplanationAdapter
from utils.time_series_analyzer import TimeSeriesAnalyzer, TemporalExplanationGenerator
from visualization.attention_vis import AttentionVisualizer
from visualization.explanation_vis import ExplanationVisualizer

class IntegrationLayer:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize components
        self.user_modeling = UserModelingSystem()
        self.explanation_adapter = ExplanationAdapter(self.user_modeling)
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.temporal_explanation_generator = TemporalExplanationGenerator(self.time_series_analyzer)
        self.attention_visualizer = AttentionVisualizer()
        self.explanation_visualizer = ExplanationVisualizer()
    
    def process_query(
        self,
        query: str,
        user_id: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """Process a query and generate appropriate explanations"""
        # Tokenize input
        input_ids = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
        
        # Get user profile and expertise level
        profile = self.user_modeling.get_user_profile(user_id)
        expertise_level = self.user_modeling.get_explanation_level(user_id)
        
        # Generate model output and base explanation
        with torch.no_grad():
            output, explanation, attention_weights, attention_patterns = self.model.generate(input_ids)
        
        # Generate temporal analysis if needed
        temporal_analysis = None
        if context and context.get('temporal_data') is not None:
            temporal_analysis = self.time_series_analyzer.calculate_temporal_importance(
                context['temporal_data'],
                attention_weights
            )
        
        # Adapt explanation based on user profile
        adapted_explanation = self.explanation_adapter.adapt_explanation(
            {
                'base_explanation': explanation,
                'attention_weights': attention_weights,
                'attention_patterns': attention_patterns
            },
            user_id
        )
        
        # Generate visualizations
        visualizations = self._generate_visualizations(
            adapted_explanation,
            expertise_level,
            temporal_analysis
        )
        
        # Record interaction
        self._record_interaction(user_id, query, output, adapted_explanation)
        
        return {
            'output': self.tokenizer.decode(output[0], skip_special_tokens=True),
            'explanation': adapted_explanation,
            'visualizations': visualizations,
            'expertise_level': expertise_level,
            'temporal_analysis': temporal_analysis
        }
    
    def _generate_visualizations(
        self,
        explanation: Dict[str, torch.Tensor],
        expertise_level: int,
        temporal_analysis: Optional[Dict] = None
    ) -> Dict:
        """Generate appropriate visualizations based on expertise level"""
        visualizations = {}
        
        # Basic visualizations for all levels
        visualizations['attention'] = self.attention_visualizer.create_attention_heatmap(
            explanation['attention_weights'][-1].mean(dim=1),
            self.tokenizer.tokenize(explanation['base_explanation'])
        )
        
        if expertise_level >= 1:
            # Add intermediate visualizations
            visualizations['prototype'] = self.explanation_visualizer.create_prototype_visualization(
                explanation.get('prototypes', torch.tensor([])),
                explanation.get('query_embedding', torch.tensor([])),
                explanation.get('similarities', torch.tensor([])),
                []  # feature names
            )
        
        if expertise_level >= 2:
            # Add advanced visualizations
            visualizations['hierarchy'] = self.explanation_visualizer.create_hierarchical_view(
                explanation.get('attention_patterns', {})
            )
            
            if temporal_analysis:
                visualizations['temporal'] = self.temporal_explanation_generator.generate_temporal_explanation(
                    temporal_analysis['sequence_data'],
                    temporal_analysis['attention_weights'],
                    []  # feature names
                )
        
        return visualizations
    
    def _record_interaction(
        self,
        user_id: str,
        query: str,
        output: torch.Tensor,
        explanation: Dict
    ):
        """Record user interaction for profile updating"""
        # Implement interaction recording logic
        pass

# Usage example:
"""
integration = IntegrationLayer(model, tokenizer)
result = integration.process_query(
    query="What causes climate change?",
    user_id="user123",
    context={"temporal_data": climate_data}  # Optional
)

# Access results
output = result['output']
explanation = result['explanation']
visualizations = result['visualizations']
"""
