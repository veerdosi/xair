import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../contexts/AppContext';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { EmptyState } from '../components/common/EmptyState';

const Dashboard = () => {
  const { state, dispatch } = useAppContext();
  const { trees, settings } = state;
  const [prompt, setPrompt] = React.useState('');
  const navigate = useNavigate();

  const handleGenerateTree = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    dispatch({ type: 'SET_LOADING', payload: true });

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
        const error = await response.json();
        throw new Error(error.message || 'Failed to generate tree');
      }

      const data = await response.json();
      dispatch({ type: 'ADD_TREE', payload: data });
      
      // Navigate to the new tree
      navigate(`/tree/${data.id}`);
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: error.message });
    }
  };

  return (
    <div className="container mx-auto max-w-6xl py-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Explainable AI Reasoning</h1>
        <p className="text-gray-600">
          Generate and explore reasoning trees with counterfactual explanations for LLM decisions.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Generate a New Reasoning Tree</CardTitle>
            <CardDescription>
              Provide a prompt to analyze the LLM's reasoning process
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleGenerateTree} className="space-y-4">
              <div>
                <textarea
                  className="w-full min-h-[120px] p-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter your prompt here, e.g., 'Explain the logic behind climate change models' or 'Should I invest in renewable energy?'"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
              </div>
              <Button type="submit" className="w-full">
                Generate Reasoning Tree
              </Button>
            </form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Current Settings</CardTitle>
            <CardDescription>
              Generation parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="text-sm font-medium mb-1">Temperature</div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500">{settings.temperature}</span>
              </div>
            </div>
            <div>
              <div className="text-sm font-medium mb-1">Max Tokens</div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500">{settings.maxTokens}</span>
              </div>
            </div>
            <div>
              <div className="text-sm font-medium mb-1">Max Tree Depth</div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500">{settings.maxDepth}</span>
              </div>
            </div>
            <Button 
              variant="outline" 
              size="sm" 
              className="w-full"
              onClick={() => navigate('/settings')}
            >
              Adjust Settings
            </Button>
          </CardContent>
        </Card>

        <Card className="md:col-span-3">
          <CardHeader>
            <CardTitle>Your Reasoning Trees</CardTitle>
            <CardDescription>
              Previously generated trees and their analyses
            </CardDescription>
          </CardHeader>
          <CardContent>
            {Object.keys(trees).length === 0 ? (
              <EmptyState
                title="No trees generated yet"
                description="Generate a new reasoning tree to get started"
                action={
                  <Button onClick={() => document.querySelector('textarea').focus()}>
                    Create your first tree
                  </Button>
                }
              />
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.values(trees).map((tree) => (
                  <Card key={tree.id} className="overflow-hidden">
                    <div className="p-4">
                      <div className="font-medium mb-1">Tree #{tree.id}</div>
                      <div className="text-sm text-gray-500 line-clamp-2 mb-3">
                        {tree.prompt}
                      </div>
                      <div className="flex items-center justify-between mb-3 text-xs text-gray-600">
                        <div>{tree.stats?.node_count || 0} nodes</div>
                        <div>{tree.stats?.edge_count || 0} edges</div>
                        <div>{tree.stats?.divergence_points || 0} divergence points</div>
                      </div>
                      <div className="flex space-x-2">
                        <Button
                          variant="outline"
                          size="sm"
                          className="flex-1"
                          onClick={() => navigate(`/tree/${tree.id}`)}
                        >
                          Explore
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="flex-1"
                          onClick={() => navigate(`/counterfactuals/${tree.id}`)}
                        >
                          Counterfactuals
                        </Button>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;