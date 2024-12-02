# utils/time_series_analyzer.py

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy
from statsmodels.tsa.stattools import adfuller
import pandas as pd

class TimeSeriesAnalyzer:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
    
    def calculate_temporal_importance(
        self,
        sequence_data: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Calculate feature importance across time steps"""
        
        # Calculate attention-based importance
        temporal_importance = self._compute_temporal_attention(attention_weights)
        
        # Calculate feature dynamics
        feature_dynamics = self._analyze_feature_dynamics(sequence_data)
        
        # Detect temporal patterns
        patterns = self._detect_temporal_patterns(sequence_data)
        
        return {
            'temporal_importance': temporal_importance,
            'feature_dynamics': feature_dynamics,
            'temporal_patterns': patterns
        }
    
    def _compute_temporal_attention(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute importance based on attention patterns over time"""
        # Average attention across heads
        mean_attention = attention_weights.mean(dim=1)
        
        # Calculate temporal importance scores
        importance_scores = torch.zeros(mean_attention.size(0))
        
        for t in range(mean_attention.size(0)):
            # Calculate information flow at each time step
            forward_flow = mean_attention[t, t:].sum()
            backward_flow = mean_attention[t, :t].sum()
            importance_scores[t] = forward_flow + backward_flow
            
        return importance_scores
    
    def _analyze_feature_dynamics(self, sequence_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze how features change over time"""
        dynamics = {}
        
        # Calculate rate of change
        derivatives = torch.diff(sequence_data, dim=0)
        
        # Detect trend
        trend = self._calculate_trend(sequence_data)
        
        # Calculate volatility
        volatility = torch.std(sequence_data, dim=0)
        
        # Detect seasonality
        seasonality = self._detect_seasonality(sequence_data)
        
        dynamics['rate_of_change'] = derivatives
        dynamics['trend'] = trend
        dynamics['volatility'] = volatility
        dynamics['seasonality'] = seasonality
        
        return dynamics
    
    def _detect_temporal_patterns(self, sequence_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect significant temporal patterns in the data"""
        patterns = {}
        
        # Sliding window analysis
        for i in range(len(sequence_data) - self.window_size + 1):
            window = sequence_data[i:i+self.window_size]
            
            # Pattern significance score
            significance = self._calculate_pattern_significance(window)
            
            if significance > 0.5:  # Threshold for significant patterns
                patterns[f'pattern_{i}'] = {
                    'window': window,
                    'significance': significance
                }
        
        return patterns
    
    def _calculate_trend(self, sequence_data: torch.Tensor) -> torch.Tensor:
        """Calculate trend component using linear regression"""
        x = torch.arange(sequence_data.size(0), dtype=torch.float32)
        X = torch.stack([x, torch.ones_like(x)], dim=1)
        
        # Solve normal equations
        beta = torch.linalg.lstsq(X, sequence_data).solution
        
        return beta[0]  # Return slope as trend indicator
    
    def _detect_seasonality(self, sequence_data: torch.Tensor) -> torch.Tensor:
        """Detect seasonal patterns using autocorrelation"""
        n_features = sequence_data.size(1)
        seasonality_scores = torch.zeros(n_features)
        
        for i in range(n_features):
            feature_data = sequence_data[:, i]
            
            # Calculate autocorrelation
            autocorr = self._autocorrelation(feature_data)
            
            # Find peaks in autocorrelation
            peaks = self._find_peaks(autocorr)
            
            if len(peaks) > 0:
                seasonality_scores[i] = peaks[0]  # Use first peak as seasonality indicator
                
        return seasonality_scores
    
    @staticmethod
    def _autocorrelation(sequence: torch.Tensor) -> torch.Tensor:
        """Calculate autocorrelation of a sequence"""
        n = len(sequence)
        mean = sequence.mean()
        var = sequence.var()
        
        r = torch.zeros(n//2)
        for lag in range(n//2):
            r[lag] = ((sequence[lag:] - mean) * (sequence[:-lag if lag > 0 else n] - mean)).mean() / var
            
        return r
    
    @staticmethod
    def _find_peaks(sequence: torch.Tensor) -> List[int]:
        """Find peaks in a sequence"""
        peaks = []
        for i in range(1, len(sequence)-1):
            if sequence[i-1] < sequence[i] > sequence[i+1]:
                peaks.append(i)
                
        return peaks
    
    @staticmethod
    def _calculate_pattern_significance(window: torch.Tensor) -> float:
        """Calculate significance score for a temporal pattern"""
        # Calculate entropy of the pattern
        normalized = (window - window.mean()) / window.std()
        hist = torch.histc(normalized, bins=10)
        probs = hist / hist.sum()
        pattern_entropy = entropy(probs[probs > 0].numpy())
        
        # Convert entropy to significance score (lower entropy = higher significance)
        significance = 1 / (1 + pattern_entropy)
        
        return significance

class TemporalExplanationGenerator:
    def __init__(self, analyzer: TimeSeriesAnalyzer):
        self.analyzer = analyzer
        
    def generate_temporal_explanation(
        self,
        sequence_data: torch.Tensor,
        attention_weights: torch.Tensor,
        feature_names: List[str]
    ) -> Dict[str, str]:
        """Generate natural language explanations for temporal patterns"""
        
        # Get temporal analysis
        analysis = self.analyzer.calculate_temporal_importance(sequence_data, attention_weights)
        
        # Generate explanations
        explanations = {}
        
        # Explain important time steps
        time_importance = analysis['temporal_importance']
        important_times = torch.topk(time_importance, k=3)
        explanations['important_times'] = self._explain_important_times(
            important_times.indices,
            important_times.values
        )
        
        # Explain feature dynamics
        dynamics = analysis['feature_dynamics']
        explanations['feature_dynamics'] = self._explain_feature_dynamics(
            dynamics,
            feature_names
        )
        
        # Explain patterns
        patterns = analysis['temporal_patterns']
        explanations['patterns'] = self._explain_patterns(patterns)
        
        return explanations
    
    def _explain_important_times(
        self,
        indices: torch.Tensor,
        values: torch.Tensor
    ) -> str:
        """Generate explanation for important time steps"""
        explanation = "The most significant time steps are:\n"
        
        for idx, val in zip(indices, values):
            explanation += f"- Time step {idx.item()}: Importance score {val.item():.3f}\n"
            
        return explanation
    
    def _explain_feature_dynamics(
        self,
        dynamics: Dict[str, torch.Tensor],
        feature_names: List[str]
    ) -> str:
        """Generate explanation for feature dynamics"""
        explanation = "Feature dynamics analysis:\n"
        
        for i, name in enumerate(feature_names):
            trend = dynamics['trend'][i].item()
            volatility = dynamics['volatility'][i].item()
            seasonality = dynamics['seasonality'][i].item()
            
            explanation += f"\n{name}:\n"
            explanation += f"- {'Increasing' if trend > 0 else 'Decreasing'} trend (slope: {trend:.3f})\n"
            explanation += f"- {'High' if volatility > 0.5 else 'Low'} volatility ({volatility:.3f})\n"
            explanation += f"- {'Seasonal' if seasonality > 0.5 else 'Non-seasonal'} pattern "
            explanation += f"(seasonality score: {seasonality:.3f})\n"
            
        return explanation
    
    def _explain_patterns(self, patterns: Dict[str, Dict]) -> str:
        """Generate explanation for detected patterns"""
        if not patterns:
            return "No significant temporal patterns detected."
            
        explanation = "Detected temporal patterns:\n"
        
        for pattern_id, pattern_info in patterns.items():
            significance = pattern_info['significance']
            explanation += f"\n- Pattern at {pattern_id}:\n"
            explanation += f"  Significance: {significance:.3f}\n"
            explanation += "  Characteristics: "
            explanation += self._characterize_pattern(pattern_info['window'])
            
        return explanation
    
    def _characterize_pattern(self, window: torch.Tensor) -> str:
        """Characterize a temporal pattern"""
        # Calculate basic statistics
        mean = window.mean().item()
        std = window.std().item()
        slope = (window[-1] - window[0]).item()
        
        # Generate description
        description = []
        
        if abs(slope) > std:
            description.append("trending" if slope > 0 else "declining")
        if std > abs(mean) * 0.5:
            description.append("volatile")
        if len(self.analyzer._find_peaks(window)) > 1:
            description.append("oscillating")
            
        return ", ".join(description) if description else "stable"
