"""
Counterfactual package for the XAIR system.
Handles generation and evaluation of counterfactual alternatives.
"""

from backend.counterfactual.generator import CounterfactualGenerator, CounterfactualCandidate
from backend.counterfactual.evaluator import CounterfactualEvaluator, EvaluationMetrics
from backend.counterfactual.ranker import CounterfactualRanker, RankingResult
from backend.counterfactual.counterfactual_main import Counterfactual

__all__ = [
    'CounterfactualGenerator',
    'CounterfactualCandidate',
    'CounterfactualEvaluator',
    'EvaluationMetrics',
    'CounterfactualRanker',
    'RankingResult',
    'Counterfactual'
]