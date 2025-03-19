import { useState, useEffect } from 'react';
import { useAppContext } from '../contexts/AppContext';

export function useCounterfactuals(treeId) {
  const { state, dispatch } = useAppContext();
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [selectedCounterfactual, setSelectedCounterfactual] = useState(null);

  const counterfactuals = state.counterfactuals[treeId] || [];
  const impacts = state.impacts[treeId] || [];

  // Flag to track if counterfactuals have been loaded
  const hasCounterfactuals = counterfactuals.length > 0;
  // Flag to track if impacts have been loaded
  const hasImpacts = impacts.length > 0;

  // Generate counterfactuals for the tree
  const generateCounterfactuals = async () => {
    if (!treeId || loading) return;
    
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/counterfactuals', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ tree_id: treeId }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to generate counterfactuals');
      }

      const data = await response.json();
      
      // Update the global state
      dispatch({
        type: 'ADD_COUNTERFACTUALS',
        payload: {
          treeId,
          counterfactuals: data.counterfactuals,
        },
      });

      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Analyze the impact of counterfactuals
  const analyzeImpacts = async (counterfactualIds = null) => {
    if (!treeId || analyzing || !hasCounterfactuals) return;
    
    setAnalyzing(true);
    setError(null);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tree_id: treeId,
          counterfactual_ids: counterfactualIds,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to analyze impacts');
      }

      const data = await response.json();
      
      // Update the global state
      dispatch({
        type: 'ADD_IMPACTS',
        payload: {
          treeId,
          impacts: data.impact_scores,
        },
      });

      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setAnalyzing(false);
    }
  };

  // Get counterfactual with impact data
  const getCounterfactualsWithImpact = () => {
    return counterfactuals.map(cf => {
      const impact = impacts.find(imp => imp.counterfactual_id === cf.id);
      return { ...cf, impact };
    });
  };

  // Get sorted counterfactuals by impact score
  const getSortedCounterfactuals = () => {
    const cfWithImpacts = getCounterfactualsWithImpact();
    return [...cfWithImpacts].sort((a, b) => {
      const impactA = a.impact?.composite_score || 0;
      const impactB = b.impact?.composite_score || 0;
      return impactB - impactA;
    });
  };

  // Select a counterfactual by ID
  const selectCounterfactual = (id) => {
    const cf = counterfactuals.find(c => c.id === id);
    setSelectedCounterfactual(cf || null);
    return cf;
  };

  return {
    loading,
    analyzing,
    error,
    counterfactuals,
    impacts,
    hasCounterfactuals,
    hasImpacts,
    selectedCounterfactual,
    generateCounterfactuals,
    analyzeImpacts,
    getCounterfactualsWithImpact,
    getSortedCounterfactuals,
    selectCounterfactual,
  };
}