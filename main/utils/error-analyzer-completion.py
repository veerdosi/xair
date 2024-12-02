# eval/error_analyzer.py (continuation)

def _find_significant_patterns(
    self,
    attention: torch.Tensor,
    threshold: float = 0.1
) -> List[str]:
    """Find significant attention patterns in errors"""
    patterns = []
    
    # Analyze per layer
    for layer_idx, layer_attention in enumerate(attention):
        # Find strong attention connections
        strong_connections = torch.where(layer_attention > threshold)
        
        for head_idx in range(layer_attention.size(0)):
            head_patterns = self._analyze_head_patterns(
                layer_attention[head_idx],
                strong_connections
            )
            if head_patterns:
                patterns.extend([
                    f"L{layer_idx}H{head_idx}: {pattern}"
                    for pattern in head_patterns
                ])
    
    return patterns

def _analyze_head_patterns(
    self,
    head_attention: torch.Tensor,
    strong_connections: Tuple[torch.Tensor, ...]
) -> List[str]:
    """Analyze attention patterns in a specific head"""
    patterns = []
    
    # Find attention focus patterns
    attention_stats = {
        'local': self._check_local_attention(head_attention),
        'global': self._check_global_attention(head_attention),
        'sequential': self._check_sequential_attention(head_attention)
    }
    
    # Add identified patterns
    for pattern_type, is_present in attention_stats.items():
        if is_present:
            patterns.append(f"{pattern_type}_attention")
    
    return patterns

def _analyze_error_attention(
    self,
    error_indices: List[int],
    attention_weights: List[torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Analyze attention patterns in errors"""
    error_attention = defaultdict(list)
    
    for idx in error_indices:
        attention = attention_weights[idx]
        
        # Analyze attention statistics
        error_attention['mean_attention'].append(
            torch.mean(attention, dim=(0, 1))
        )
        error_attention['attention_variance'].append(
            torch.var(attention, dim=(0, 1))
        )
        error_attention['attention_entropy'].append(
            self._calculate_attention_entropy(attention)
        )
    
    # Aggregate statistics
    return {
        key: torch.stack(tensors).mean(dim=0)
        for key, tensors in error_attention.items()
    }

def _analyze_feature_importance(
    self,
    error_indices: List[int],
    attention_weights: List[torch.Tensor]
) -> Dict[str, float]:
    """Analyze feature importance in errors"""
    feature_importance = defaultdict(float)
    
    for idx in error_indices:
        attention = attention_weights[idx]
        
        # Calculate feature importance based on attention
        importance = self._calculate_feature_importance(attention)
        
        for feature, score in importance.items():
            feature_importance[feature] += score
    
    # Normalize importance scores
    total = sum(feature_importance.values())
    return {
        feature: score/total
        for feature, score in feature_importance.items()
    }

def _analyze_error_distribution(
    self,
    predictions: List[int],
    targets: List[int]
) -> pd.DataFrame:
    """Analyze the distribution of errors"""
    # Create confusion matrix
    conf_matrix = confusion_matrix(targets, predictions)
    
    # Calculate error statistics
    report = classification_report(
        targets,
        predictions,
        output_dict=True
    )
    
    # Create DataFrame with error statistics
    df = pd.DataFrame(report).transpose()
    df['error_rate'] = 1 - df['precision']
    
    return df

def visualize_error_analysis(
    self,
    result: ErrorAnalysisResult
) -> Dict[str, go.Figure]:
    """Create visualizations for error analysis"""
    visualizations = {}
    
    # Error type distribution
    visualizations['error_types'] = self._create_error_type_viz(
        result.error_types
    )
    
    # Attention pattern analysis
    visualizations['attention_patterns'] = self._create_attention_pattern_viz(
        result.attention_analysis
    )
    
    # Feature importance
    visualizations['feature_importance'] = self._create_feature_importance_viz(
        result.feature_importance
    )
    
    # Error distribution
    visualizations['error_distribution'] = self._create_error_distribution_viz(
        result.error_distribution
    )
    
    return visualizations

def _create_error_type_viz(
    self,
    error_types: Dict[str, int]
) -> go.Figure:
    """Create visualization for error types"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(error_types.keys()),
            y=list(error_types.values()),
            marker_color='red'
        )
    ])
    
    fig.update_layout(
        title="Distribution of Error Types",
        xaxis_title="Error Type",
        yaxis_title="Count",
        height=400
    )
    
    return fig

def _create_attention_pattern_viz(
    self,
    attention_analysis: Dict[str, torch.Tensor]
) -> go.Figure:
    """Create visualization for attention patterns"""
    fig = go.Figure()
    
    for name, pattern in attention_analysis.items():
        fig.add_trace(go.Heatmap(
            z=pattern.cpu().numpy(),
            name=name,
            colorscale='Viridis'
        ))
    
    fig.update_layout(
        title="Attention Patterns in Errors",
        height=600,
        updatemenus=[{
            'buttons': [
                {'label': name,
                    'method': 'update',
                    'args': [{'visible': [i == j for i in range(len(attention_analysis))]}]}
                for j, name in enumerate(attention_analysis.keys())
            ],
            'direction': 'down',
            'showactive': True
        }]
    )
    
    return fig

def _create_feature_importance_viz(
    self,
    feature_importance: Dict[str, float]
) -> go.Figure:
    """Create visualization for feature importance"""
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f[0] for f in sorted_features],
            y=[f[1] for f in sorted_features],
            marker_color='blue'
        )
    ])
    
    fig.update_layout(
        title="Feature Importance in Errors",
        xaxis_title="Feature",
        yaxis_title="Importance Score",
        height=400
    )
    
    return fig

def _create_error_distribution_viz(
    self,
    error_distribution: pd.DataFrame
) -> go.Figure:
    """Create visualization for error distribution"""
    fig = go.Figure()
    
    # Add error rate trace
    fig.add_trace(go.Scatter(
        x=error_distribution.index,
        y=error_distribution['error_rate'],
        mode='lines+markers',
        name='Error Rate'
    ))
    
    # Add support trace
    fig.add_trace(go.Bar(
        x=error_distribution.index,
        y=error_distribution['support'],
        name='Support',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Error Distribution Analysis",
        xaxis_title="Class",
        yaxis_title="Error Rate",
        yaxis2=dict(
            title="Support",
            overlaying='y',
            side='right'
        ),
        height=500
    )
    
    return fig

def _calculate_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
    """Calculate entropy of attention weights"""
    # Add small epsilon to avoid log(0)
    attention = attention + 1e-10
    attention = attention / attention.sum(dim=-1, keepdim=True)
    entropy = -(attention * torch.log(attention)).sum(dim=-1)
    return entropy

def _check_local_attention(self, attention: torch.Tensor, window: int = 3) -> bool:
    """Check if attention pattern is primarily local"""
    diagonal_sum = torch.sum(torch.diagonal(attention, offset=k) 
                            for k in range(-window, window+1))
    return diagonal_sum > 0.5 * attention.sum()

def _check_global_attention(self, attention: torch.Tensor, threshold: float = 0.1) -> bool:
    """Check if attention pattern is primarily global"""
    return torch.mean(attention > threshold) > 0.5

def _check_sequential_attention(self, attention: torch.Tensor) -> bool:
    """Check if attention pattern is sequential"""
    upper_triangular = torch.triu(attention)
    return torch.sum(upper_triangular) > 0.7 * attention.sum()
