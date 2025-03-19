import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useAppContext } from '../contexts/AppContext';
import ReasoningTree from '../components/visualizations/ReasoningTree';
import CounterfactualPanel from '../components/visualizations/CounterfactualPanel';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Loading } from '../components/common/Loading';
import { Error } from '../components/common/Error';

const TreeExplorer = () => {
  const { treeId } = useParams();
  const { state, dispatch } = useAppContext();
  const { trees, counterfactuals, impacts } = state;
  const [selectedNode, setSelectedNode] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState({
    counterfactuals: false,
    impacts: false,
  });

  const tree = trees[treeId];
  const hasCounterfactuals = counterfactuals[treeId] && counterfactuals[treeId].length > 0;
  const hasImpacts = impacts[treeId] && impacts[treeId].length > 0;

  useEffect(() => {
    if (tree) {
      dispatch({ type: 'SET_CURRENT_TREE', payload: treeId });
    }
  }, [tree, treeId, dispatch]);

  const generateCounterfactuals = async () => {
    setLoading({ ...loading, counterfactuals: true });
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
      
      dispatch({
        type: 'ADD_COUNTERFACTUALS',
        payload: {
          treeId,
          counterfactuals: data.counterfactuals,
        },
      });
      
      // Automatically analyze the impacts
      analyzeImpacts();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading({ ...loading, counterfactuals: false });
    }
  };

  const analyzeImpacts = async () => {
    setLoading({ ...loading, impacts: true });
    setError(null);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ tree_id: treeId }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to analyze impacts');
      }

      const data = await response.json();
      
      dispatch({
        type: 'ADD_IMPACTS',
        payload: {
          treeId,
          impacts: data.impact_scores,
        },
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading({ ...loading, impacts: false });
    }
  };

  if (!tree) {
    return <Error message={`Tree ${treeId} not found`} />;
  }

  return (
    <div className="container mx-auto max-w-7xl py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-2">Tree Explorer</h1>
        <p className="text-gray-600 mb-4">Explore the reasoning tree and generate counterfactuals</p>
        
        <div className="flex flex-wrap items-center gap-3 mb-3">
          <Card className="p-3">
            <div className="text-xs text-gray-500">Prompt</div>
            <div className="text-sm font-medium">{tree.prompt}</div>
          </Card>
          
          <Card className="p-3">
            <div className="text-xs text-gray-500">Nodes</div>
            <div className="text-sm font-medium">{tree.stats?.node_count || 0}</div>
          </Card>
          
          <Card className="p-3">
            <div className="text-xs text-gray-500">Edges</div>
            <div className="text-sm font-medium">{tree.stats?.edge_count || 0}</div>
          </Card>
          
          <Card className="p-3">
            <div className="text-xs text-gray-500">Divergence Points</div>
            <div className="text-sm font-medium">{tree.stats?.divergence_points || 0}</div>
          </Card>
        </div>
        
        {!hasCounterfactuals && (
          <Button 
            onClick={generateCounterfactuals}
            disabled={loading.counterfactuals}
          >
            {loading.counterfactuals ? 'Generating...' : 'Generate Counterfactuals'}
          </Button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ReasoningTree 
            treeId={treeId} 
            onNodeSelect={setSelectedNode} 
          />
        </div>
        
        <div>
          {loading.counterfactuals ? (
            <Loading message="Generating counterfactuals..." />
          ) : loading.impacts ? (
            <Loading message="Analyzing impacts..." />
          ) : hasCounterfactuals ? (
            <CounterfactualPanel 
              treeId={treeId} 
              onSelect={(cf) => {
                // You could highlight the related nodes in the tree here
                console.log('Selected counterfactual:', cf);
              }} 
            />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Counterfactual Analysis</CardTitle>
                <CardDescription>
                  Generate counterfactuals to see alternative reasoning paths
                </CardDescription>
              </CardHeader>
              <CardContent className="text-center py-6">
                <p className="text-gray-500 mb-4">
                  Counterfactuals help you understand how the model's reasoning would change with different inputs
                </p>
                <Button onClick={generateCounterfactuals}>
                  Generate Counterfactuals
                </Button>
              </CardContent>
            </Card>
          )}
          
          {error && <Error message={error} className="mt-4" />}
        </div>
      </div>
    </div>
  );
};

export default TreeExplorer;