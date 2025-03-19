import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAppContext } from '../../contexts/AppContext';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';
import { Input } from '../ui/Input';

const Sidebar = () => {
  const { state, dispatch } = useAppContext();
  const { trees, currentTree, settings } = state;
  const [prompt, setPrompt] = useState('');
  const navigate = useNavigate();

  const handleGenerateTree = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    dispatch({ type: 'SET_LOADING', payload: true });

    try {
      const response = await window.fetch('/api/generate', {
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
    <aside className="w-64 border-r border-gray-200 bg-white p-4 flex flex-col h-full">
      <div className="mb-6">
        <h2 className="text-lg font-medium mb-3">Generate New Tree</h2>
        <form onSubmit={handleGenerateTree}>
          <div className="space-y-3">
            <Input
              type="text"
              placeholder="Enter prompt..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full"
            />
            <Button type="submit" className="w-full">
              Generate
            </Button>
          </div>
        </form>
      </div>

      <div className="flex-1 overflow-auto">
        <h2 className="text-lg font-medium mb-3">Reasoning Trees</h2>
        {Object.keys(trees).length === 0 ? (
          <div className="text-sm text-gray-500 py-2">No trees generated yet</div>
        ) : (
          <div className="space-y-2">
            {Object.values(trees).map((tree) => (
              <Card 
                key={tree.id} 
                className={`p-3 hover:shadow-md transition-shadow cursor-pointer ${
                  currentTree === tree.id ? 'bg-blue-50 border-blue-200' : ''
                }`}
                onClick={() => {
                  dispatch({ type: 'SET_CURRENT_TREE', payload: tree.id });
                  navigate(`/tree/${tree.id}`);
                }}
              >
                <div className="font-medium mb-1 truncate">Tree #{tree.id}</div>
                <div className="text-xs text-gray-500 truncate">{tree.prompt}</div>
                <div className="flex items-center justify-between mt-2 text-xs text-gray-600">
                  <span>{tree.stats?.node_count || 0} nodes</span>
                  <div className="flex space-x-2">
                    <Link 
                      to={`/tree/${tree.id}`}
                      className="text-blue-600 hover:text-blue-800"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Explore
                    </Link>
                    <Link 
                      to={`/counterfactuals/${tree.id}`}
                      className="text-orange-600 hover:text-orange-800"
                      onClick={(e) => e.stopPropagation()}
                    >
                      CF
                    </Link>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>

      <div className="mt-4 pt-4 border-t border-gray-200">
        <Link to="/settings" className="text-sm text-gray-600 hover:text-gray-900 flex items-center">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-4 w-4 mr-2"
          >
            <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path>
            <circle cx="12" cy="12" r="3"></circle>
          </svg>
          Settings
        </Link>
      </div>
    </aside>
  );
};

export default Sidebar;