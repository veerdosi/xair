# interface/dashboard.py

import gradio as gr
import torch
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ExplanationDashboard:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.expertise_levels = {
            0: "Beginner",
            1: "Intermediate",
            2: "Expert"
        }
        self.current_attention_weights = None
        self.current_explanation = None
    
    def create_attention_heatmap(self, attention_weights: torch.Tensor, tokens: List[str]) -> go.Figure:
        """Create an interactive attention heatmap using plotly"""
        attention_weights = attention_weights.cpu().numpy()
        
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Attention Weights Visualization',
            xaxis_title='Target Tokens',
            yaxis_title='Source Tokens',
            width=800,
            height=600
        )
        
        return fig
    
    def create_uncertainty_plot(self, uncertainty: torch.Tensor) -> go.Figure:
        """Create an uncertainty visualization"""
        uncertainty = uncertainty.cpu().numpy()
        
        fig = go.Figure(data=go.Bar(
            y=uncertainty,
            error_y=dict(type='data', array=uncertainty*0.1),
            name='Prediction Uncertainty'
        ))
        
        fig.update_layout(
            title='Model Uncertainty',
            xaxis_title='Token Position',
            yaxis_title='Uncertainty Score',
            width=600,
            height=400
        )
        
        return fig
    
    def format_explanation(self, explanation_components: Dict[str, torch.Tensor], expertise_level: int) -> str:
        """Format explanation based on expertise level"""
        formatted_text = []
        
        # Basic explanation for all levels
        formatted_text.append("Basic Explanation:")
        formatted_text.append(self.tokenizer.decode(explanation_components['base_explanation'].argmax(dim=-1)))
        
        if expertise_level >= 1:
            # Add intermediate details
            formatted_text.append("\nConfidence Analysis:")
            uncertainty = explanation_components['uncertainty'].mean().item()
            formatted_text.append(f"Model Confidence: {(1 - uncertainty) * 100:.2f}%")
            
            formatted_text.append("\nSimilar Examples:")
            for i, prototype in enumerate(explanation_components['similar_prototypes'].indices[:3]):
                formatted_text.append(f"Example {i+1}: {self.tokenizer.decode(prototype)}")
        
        if expertise_level >= 2:
            # Add expert-level details
            formatted_text.append("\nDetailed Analysis:")
            formatted_text.append("Attention Analysis:")
            attn_patterns = explanation_components['detailed_attention']
            formatted_text.append(f"- Number of significant attention heads: {(attn_patterns['head_importance'] > 0.1).sum().item()}")
            formatted_text.append(f"- Cross-layer interaction strength: {attn_patterns['cross_layer_patterns'].mean().item():.3f}")
            
            formatted_text.append("\nCounterfactual Analysis:")
            counterfactual = explanation_components['counterfactuals']
            formatted_text.append(f"Alternative prediction: {self.tokenizer.decode(counterfactual.argmax(dim=-1))}")
        
        return "\n".join(formatted_text)
    
    def process_input(self, question: str, expertise_level: int) -> Tuple[str, str, go.Figure, go.Figure]:
        """Process input and generate visualizations"""
        # Tokenize input
        input_ids = self.tokenizer.encode(question, return_tensors='pt').to(self.device)
        
        # Generate answer and explanation
        with torch.no_grad():
            output, explanation, attention_weights, attention_patterns = self.model.generate(input_ids)
            
        # Generate adaptive explanation
        explanation_components = self.model.adaptive_explanation_module.generate_explanation(
            output,
            attention_weights,
            expertise_level
        )
        
        # Format outputs
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        formatted_explanation = self.format_explanation(explanation_components, expertise_level)
        
        # Create visualizations
        tokens = self.tokenizer.tokenize(question + " " + answer)
        attention_viz = self.create_attention_heatmap(attention_weights[-1].mean(dim=1), tokens)
        uncertainty_viz = self.create_uncertainty_plot(explanation_components['uncertainty'])
        
        return answer, formatted_explanation, attention_viz, uncertainty_viz
    
    def launch(self, share=False):
        """Launch the Gradio interface"""
        interface = gr.Interface(
            fn=self.process_input,
            inputs=[
                gr.Textbox(label="Enter your question", placeholder="Ask a question..."),
                gr.Slider(minimum=0, maximum=2, step=1, value=0, label="Expertise Level", 
                         info="0: Beginner, 1: Intermediate, 2: Expert")
            ],
            outputs=[
                gr.Textbox(label="Answer"),
                gr.Textbox(label="Explanation", lines=10),
                gr.Plot(label="Attention Visualization"),
                gr.Plot(label="Uncertainty Visualization")
            ],
            title="Explainable LLM Dashboard",
            description="Interactive dashboard for exploring model explanations at different expertise levels.",
            theme="default",
            css="footer {display: none !important;}"
        )
        
        interface.launch(share=share)

# Usage example:
"""
from models.explainable_llm import ExplainableLLM
from transformers import AutoTokenizer

model = ExplainableLLM(...)  # Initialize with appropriate parameters
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Or your chosen tokenizer

dashboard = ExplanationDashboard(model, tokenizer)
dashboard.launch()
"""
