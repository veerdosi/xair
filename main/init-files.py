# core/__init__.py
"""Core components for the explainable LLM system."""
from .explainable_llm import ExplainableLLM
from .explainable_transformer_layer import ExplainableTransformerLayer
from .explainable_multihead_attention import ExplainableMultiheadAttention
from .explanation_decoder import ExplanationDecoder

__all__ = [
    'ExplainableLLM',
    'ExplainableTransformerLayer',
    'ExplainableMultiheadAttention',
    'ExplanationDecoder'
]

# data/__init__.py
"""Data handling components for the explainable LLM system."""
from .data_generator import DataGenerator
from .dataset import QAExplanationDataset

__all__ = [
    'DataGenerator',
    'QAExplanationDataset'
]

# eval/__init__.py
"""Evaluation components for the explainable LLM system."""
from .evaluation_module import Evaluator, ExplanationMetrics
from .human_evaluation_protocol import HumanEvaluationProtocol
from .comprehensive_testing_script import comprehensive_testing

__all__ = [
    'Evaluator',
    'ExplanationMetrics',
    'HumanEvaluationProtocol',
    'comprehensive_testing'
]

# training/__init__.py
"""Training components for the explainable LLM system."""
from .trainer import Trainer
from .visualization_module import Visualizer

__all__ = [
    'Trainer',
    'Visualizer'
]

# interface/__init__.py
"""Interface components for the explainable LLM system."""
from .dashboard import ExplanationDashboard

__all__ = [
    'ExplanationDashboard'
]

# visualization/__init__.py
"""Visualization components for the explainable LLM system."""
from .attention_vis import AttentionVisualizer
from .explanation_vis import ExplanationVisualizer

__all__ = [
    'AttentionVisualizer',
    'ExplanationVisualizer'
]

# utils/__init__.py
"""Utility components for the explainable LLM system."""
from .explanation_utils import ExplanationProcessor
from .user_modeling import UserExpertiseTracker

__all__ = [
    'ExplanationProcessor',
    'UserExpertiseTracker'
]

# Root level __init__.py (in project root directory)
"""
Explainable LLM System
======================

A system for training and deploying explainable language models with
adaptive explanation capabilities and interactive visualization tools.
"""

from . import core
from . import data
from . import eval
from . import training
from . import interface
from . import visualization
from . import utils

__version__ = '0.1.0'
__author__ = 'Your Name'

__all__ = [
    'core',
    'data',
    'eval',
    'training',
    'interface',
    'visualization',
    'utils'
]
