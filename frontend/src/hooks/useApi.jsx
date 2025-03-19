import { useState, useCallback } from 'react';
import { useAppContext } from '../contexts/AppContext';

const API_BASE_URL = '/api';

export function useApi() {
  const { dispatch } = useAppContext();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const request = useCallback(
    async (endpoint, options = {}) => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
          headers: {
            'Content-Type': 'application/json',
          },
          ...options,
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || 'API request failed');
        }

        const data = await response.json();
        return data;
      } catch (err) {
        setError(err.message);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [dispatch]
  );

  const generateTree = useCallback(
    async (prompt, settings) => {
      dispatch({ type: 'SET_LOADING', payload: true });

      try {
        const data = await request('/generate', {
          method: 'POST',
          body: JSON.stringify({
            prompt,
            temperature: settings.temperature,
            max_tokens: settings.maxTokens,
            max_depth: settings.maxDepth,
            min_probability: settings.minProbability,
          }),
        });

        dispatch({ type: 'ADD_TREE', payload: data });
        return data;
      } catch (err) {
        dispatch({ type: 'SET_ERROR', payload: err.message });
        throw err;
      }
    },
    [dispatch, request]
  );

  const generateCounterfactuals = useCallback(
    async (treeId) => {
      dispatch({ type: 'SET_LOADING', payload: true });

      try {
        const data = await request('/counterfactuals', {
          method: 'POST',
          body: JSON.stringify({ tree_id: treeId }),
        });

        dispatch({
          type: 'ADD_COUNTERFACTUALS',
          payload: {
            treeId,
            counterfactuals: data.counterfactuals,
          },
        });

        return data;
      } catch (err) {
        dispatch({ type: 'SET_ERROR', payload: err.message });
        throw err;
      }
    },
    [dispatch, request]
  );

  const analyzeImpacts = useCallback(
    async (treeId, counterfactualIds = null) => {
      dispatch({ type: 'SET_LOADING', payload: true });

      try {
        const data = await request('/analyze', {
          method: 'POST',
          body: JSON.stringify({
            tree_id: treeId,
            counterfactual_ids: counterfactualIds,
          }),
        });

        dispatch({
          type: 'ADD_IMPACTS',
          payload: {
            treeId,
            impacts: data.impact_scores,
          },
        });

        return data;
      } catch (err) {
        dispatch({ type: 'SET_ERROR', payload: err.message });
        throw err;
      }
    },
    [dispatch, request]
  );

  return {
    loading,
    error,
    request,
    generateTree,
    generateCounterfactuals,
    analyzeImpacts,
  };
}
