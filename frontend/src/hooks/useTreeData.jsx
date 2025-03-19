import { useState, useEffect } from 'react';
import { useAppContext } from '../contexts/AppContext';

export function useTreeData(treeId) {
  const { state, dispatch } = useAppContext();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [treeData, setTreeData] = useState(null);

  const tree = state.trees[treeId];

  useEffect(() => {
    if (tree) {
      setTreeData(tree);
    } else if (treeId) {
      fetchTree();
    }
  }, [treeId, tree]);

  const fetchTree = async () => {
    if (!treeId || loading) return;
    
    setLoading(true);
    setError(null);

    try {
      // Check if we already have the tree in state
      if (state.trees[treeId]) {
        setTreeData(state.trees[treeId]);
        return;
      }

      // Fetch the tree data from the API
      const response = await fetch(`/api/tree/${treeId}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to fetch tree data');
      }

      const data = await response.json();
      
      // Update the global state
      dispatch({
        type: 'ADD_TREE',
        payload: data,
      });

      setTreeData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const regenerateTree = async (prompt, settings) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
          max_depth: settings.maxDepth,
          min_probability: settings.minProbability,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to regenerate tree');
      }

      const data = await response.json();
      
      // Update the global state
      dispatch({
        type: 'ADD_TREE',
        payload: data,
      });

      setTreeData(data);
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Process tree data for visualization
  const processedData = treeData ? {
    nodes: treeData.visualization?.nodes || [],
    edges: treeData.visualization?.edges || [],
    stats: treeData.stats || {},
    prompt: treeData.prompt || '',
  } : null;

  return {
    loading,
    error,
    treeData: processedData,
    regenerateTree,
    fetchTree,
    rawData: treeData,
  };
}