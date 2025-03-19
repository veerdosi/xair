import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAppContext } from '../contexts/AppContext';
import ReasoningTree from '../components/visualizations/ReasoningTree';
import CounterfactualPanel from '../components/visualizations/CounterfactualPanel';
import ImpactAnalysis from '../components/visualizations/ImpactAnalysis';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/Tabs';
import { Loading } from '../components/common/Loading';
import { Error } from '../components/common/Error';

const CounterfactualAnalysis = () => {
  const { treeId } = useParams();
  const navigate = useNavigate();
  const { state, dispatch } = useAppContext();
  const { trees, counterfactuals, impacts } = state;
  const [activeTab, setActiveTab] = useState('counterfactuals');
  const [selectedCounterfactual, setSelectedCounterfactual] = useState(null);
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

  useEffect(() => {
    // If we don't have counterfactuals yet but we do have a tree, generate them
    if (tree && !hasCounterfactuals && !loading.counterfactuals) {
      generateCounterfactuals();
    }
  }, [tree, hasCounterfactuals]);

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
      
      // Once we have counterfactuals, analyze their impacts
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

  const handleSelectCounterfactual = (cf) => {
    setSelectedCounterfactual(cf);
    // Could highlight the affected nodes in the tree here
  };

  if (!tree) {
    return <Error message={`Tree ${treeId} not found`} />;
  }

  return (
    <div className="container mx-auto max-w-7xl py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-2">Counterfactual Analysis</h1>
        <p className="text-gray-600 mb-4">
          Explore alternative reasoning paths and their impacts on the decision process
        </p>
        
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <Card className="p-3">
            <div className="text-xs text-gray-500">Prompt</div>
            <div className="text-sm font-medium">{tree.prompt}</div>
          </Card>
          
          {hasCounterfactuals && (
            <Card className="p-3">
              <div className="text-xs text-gray-500">Counterfactuals</div>
              <div className="text-sm font-medium">{counterfactuals[treeId].length}</div>
            </Card>
          )}
        </div>
        
        <div className="flex gap-3">
          <Button
            variant="outline"
            onClick={() => navigate(`/tree/${treeId}`)}
          >
            Back to Tree Explorer
          </Button>
          
          {hasCounterfactuals && !hasImpacts && !loading.impacts && (
            <Button onClick={analyzeImpacts}>
              Analyze Impacts
            </Button>
          )}
        </div>
      </div>

      {(loading.counterfactuals || loading.impacts) && (
        <Loading 
          message={loading.counterfactuals 
            ? "Generating counterfactuals..." 
            : "Analyzing impacts..."
          }
        />
      )}

      {error && <Error message={error} className="mb-6" />}

      {!loading.counterfactuals && hasCounterfactuals && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <div className="flex justify-between items-center mb-4">
                <TabsList>
                  <TabsTrigger value="counterfactuals">Counterfactuals</TabsTrigger>
                  <TabsTrigger value="analysis" disabled={!hasImpacts}>Impact Analysis</TabsTrigger>
                </TabsList>
              </div>
              
              <TabsContent value="counterfactuals">
                <div className="grid grid-cols-1 gap-6">
                  <ReasoningTree 
                    treeId={treeId} 
                    onNodeSelect={() => {}} 
                  />
                </div>
              </TabsContent>
              
              <TabsContent value="analysis">
                <ImpactAnalysis treeId={treeId} />
              </TabsContent>
            </Tabs>
          </div>
          
          <div>
            <CounterfactualPanel 
              treeId={treeId} 
              onSelect={handleSelectCounterfactual} 
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default CounterfactualAnalysis;