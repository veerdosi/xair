import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import entropy
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
from pytorch_forecasting import TemporalFusionTransformer
import torch.nn.functional as F

@dataclass
class TemporalPattern:
    pattern_type: str  # 'trend', 'seasonal', 'cyclic', 'anomaly'
    start_idx: int
    end_idx: int
    confidence: float
    magnitude: float
    description: str

@dataclass
class TemporalAnalysisResult:
    patterns: List[TemporalPattern]
    importance_scores: Dict[str, np.ndarray]
    seasonal_components: Dict[str, np.ndarray]
    trend_components: Dict[str, np.ndarray]
    anomalies: Dict[str, List[int]]
    feature_interactions: Dict[str, Dict[str, float]]
    temporal_attribution: Dict[str, np.ndarray]

class EnhancedTimeSeriesAnalyzer:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: List[str],
        window_size: int = 10,
        seasonality_period: Optional[int] = None
    ):
        self.model = model
        self.feature_names = feature_names
        self.window_size = window_size
        self.seasonality_period = seasonality_period
        
        # Initialize temporal importance analyzer
        self.temporal_importance = TemporalImportanceAnalyzer(
            model=model,
            window_size=window_size
        )
        
        # Initialize pattern detector
        self.pattern_detector = TemporalPatternDetector(
            seasonality_period=seasonality_period
        )
        
        # Initialize anomaly detector
        self.anomaly_detector = TemporalAnomalyDetector(
            window_size=window_size
        )
        
        # Initialize temporal attribution
        self.attribution_analyzer = TemporalAttributionAnalyzer(
            model=model,
            feature_names=feature_names
        )
    
    def analyze_temporal_data(
        self,
        sequence_data: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> TemporalAnalysisResult:
        """Perform comprehensive temporal analysis"""
        # Calculate temporal importance
        importance_scores = self.temporal_importance.calculate_importance(
            sequence_data,
            attention_weights
        )
        
        # Detect temporal patterns
        patterns = self.pattern_detector.detect_patterns(sequence_data)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(sequence_data)
        
        # Calculate temporal attribution
        attribution = self.attribution_analyzer.calculate_attribution(
            sequence_data
        )
        
        # Decompose temporal components
        seasonal_components = {}
        trend_components = {}
        feature_interactions = {}
        
        for i, feature in enumerate(self.feature_names):
            # Decompose time series
            decomposition = self._decompose_time_series(
                sequence_data[:, i].cpu().numpy()
            )
            
            seasonal_components[feature] = decomposition.seasonal
            trend_components[feature] = decomposition.trend
            
            # Calculate feature interactions
            feature_interactions[feature] = self._calculate_temporal_interactions(
                sequence_data,
                i,
                importance_scores
            )
        
        return TemporalAnalysisResult(
            patterns=patterns,
            importance_scores=importance_scores,
            seasonal_components=seasonal_components,
            trend_components=trend_components,
            anomalies=anomalies,
            feature_interactions=feature_interactions,
            temporal_attribution=attribution
        )
    
    def _decompose_time_series(
        self,
        data: np.ndarray
    ) -> seasonal_decompose:
        """Decompose time series into trend, seasonal, and residual components"""
        if self.seasonality_period:
            period = self.seasonality_period
        else:
            # Estimate period using autocorrelation
            acf_values = acf(data, nlags=len(data)//2)
            period = self._estimate_period(acf_values)
        
        return seasonal_decompose(
            data,
            period=period,
            extrapolate_trend='freq'
        )
    
    def _calculate_temporal_interactions(
        self,
        sequence_data: torch.Tensor,
        feature_idx: int,
        importance_scores: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate temporal interactions between features"""
        interactions = {}
        feature_data = sequence_data[:, feature_idx]
        feature_importance = importance_scores[self.feature_names[feature_idx]]
        
        for i, other_feature in enumerate(self.feature_names):
            if i != feature_idx:
                other_data = sequence_data[:, i]
                other_importance = importance_scores[other_feature]
                
                # Calculate interaction strength
                interaction = self._compute_interaction_strength(
                    feature_data,
                    other_data,
                    feature_importance,
                    other_importance
                )
                
                interactions[other_feature] = float(interaction)
        
        return interactions
    
    @staticmethod
    def _estimate_period(acf_values: np.ndarray) -> int:
        """Estimate seasonality period from autocorrelation function"""
        # Find peaks in ACF
        peaks = []
        for i in range(1, len(acf_values)-1):
            if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                peaks.append((i, acf_values[i]))
        
        # Sort peaks by correlation value
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        if peaks:
            return peaks[0][0]  # Return lag of highest peak
        return 1  # Default if no clear periodicity

class TemporalImportanceAnalyzer:
    def __init__(self, model: torch.nn.Module, window_size: int):
        self.model = model
        self.window_size = window_size
    
    def calculate_importance(
        self,
        sequence_data: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, np.ndarray]:
        """Calculate temporal importance scores"""
        importance_scores = {}
        
        # Use attention weights if available
        if attention_weights is not None:
            importance_scores.update(
                self._attention_based_importance(attention_weights)
            )
        
        # Add gradient-based importance
        importance_scores.update(
            self._gradient_based_importance(sequence_data)
        )
        
        # Add prediction-based importance
        importance_scores.update(
            self._prediction_based_importance(sequence_data)
        )
        
        return importance_scores

class TemporalPatternDetector:
    def __init__(self, seasonality_period: Optional[int] = None):
        self.seasonality_period = seasonality_period
    
    def detect_patterns(
        self,
        sequence_data: torch.Tensor
    ) -> List[TemporalPattern]:
        """Detect various temporal patterns in the data"""
        patterns = []
        
        # Detect trends
        trends = self._detect_trends(sequence_data)
        patterns.extend(trends)
        
        # Detect seasonality
        seasonal = self._detect_seasonality(sequence_data)
        patterns.extend(seasonal)
        
        # Detect cycles
        cycles = self._detect_cycles(sequence_data)
        patterns.extend(cycles)
        
        return patterns

class TemporalAnomalyDetector:
    def __init__(self, window_size: int):
        self.window_size = window_size
    
    def detect_anomalies(
        self,
        sequence_data: torch.Tensor
    ) -> Dict[str, List[int]]:
        """Detect anomalies in temporal data"""
        anomalies = {}
        
        # Statistical anomaly detection
        stat_anomalies = self._statistical_anomaly_detection(sequence_data)
        anomalies['statistical'] = stat_anomalies
        
        # Model-based anomaly detection
        model_anomalies = self._model_based_anomaly_detection(sequence_data)
        anomalies['model_based'] = model_anomalies
        
        # Pattern-based anomaly detection
        pattern_anomalies = self._pattern_based_anomaly_detection(sequence_data)
        anomalies['pattern_based'] = pattern_anomalies
        
        return anomalies

class TemporalAttributionAnalyzer:
    def __init__(self, model: torch.nn.Module, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
    
    def calculate_attribution(
        self,
        sequence_data: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Calculate temporal attribution scores"""
        attribution = {}
        
        # Integrated gradients over time
        ig_attribution = self._integrated_gradients(sequence_data)
        attribution['integrated_gradients'] = ig_attribution
        
        # Attention-based attribution
        attn_attribution = self._attention_attribution(sequence_data)
        attribution['attention'] = attn_attribution
        
        return attribution
