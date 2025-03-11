import React, { useState } from 'react';
import { Brain, ZoomIn, ZoomOut } from 'lucide-react';
import { ReasoningTree } from './components/ReasoningTree';
import { CounterfactualPanel } from './components/CounterfactualPanel';
import { ExplanationPanel } from './components/ExplanationPanel';
import { TreeNode, Counterfactual } from './types';

// Sample data
const sampleTree: TreeNode = {
  id: 'root',
  label: 'Initial Query',
  type: 'actual',
  children: [
    {
      id: '1',
      label: 'Analysis Step 1',
      type: 'actual',
      children: [
        {
          id: '1.1',
          label: 'Conclusion A',
          type: 'actual',
          children: []
        },
        {
          id: '1.2',
          label: 'Alternative Path',
          type: 'counterfactual',
          children: []
        }
      ]
    },
    {
      id: '2',
      label: 'Analysis Step 2',
      type: 'actual',
      children: []
    }
  ]
};

const sampleCounterfactuals: Counterfactual[] = [
  {
    id: 'cf1',
    summary: 'If input contained more specific details, confidence would increase by 25%',
    impact: 0.8,
    applied: false
  },
  {
    id: 'cf2',
    summary: 'Changing context to technical domain would alter classification',
    impact: 0.6,
    applied: false
  }
];

function App() {
  const [zoom, setZoom] = useState(100);
  const [counterfactuals, setCounterfactuals] = useState(sampleCounterfactuals);

  const handleZoom = (direction: 'in' | 'out') => {
    setZoom(prev => direction === 'in' ? Math.min(prev + 10, 200) : Math.max(prev - 10, 50));
  };

  const handleApplyCounterfactual = (id: string) => {
    setCounterfactuals(prev =>
      prev.map(cf => ({
        ...cf,
        applied: cf.id === id ? !cf.applied : cf.applied
      }))
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-blue-500" />
              <h1 className="ml-3 text-2xl font-bold text-gray-900">XAI Visualization System</h1>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => handleZoom('out')}
                className="p-2 rounded-full hover:bg-gray-100"
              >
                <ZoomOut className="h-5 w-5 text-gray-600" />
              </button>
              <span className="text-sm text-gray-600">{zoom}%</span>
              <button
                onClick={() => handleZoom('in')}
                className="p-2 rounded-full hover:bg-gray-100"
              >
                <ZoomIn className="h-5 w-5 text-gray-600" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid grid-cols-3 gap-6">
          {/* Main Visualization */}
          <div className="col-span-2 bg-white rounded-lg shadow-lg p-4" style={{ height: 'calc(100vh - 200px)' }}>
            <ReasoningTree data={sampleTree} />
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <CounterfactualPanel
              counterfactuals={counterfactuals}
              onApply={handleApplyCounterfactual}
            />
            <ExplanationPanel
              explanation="The model analyzed the input using a multi-step reasoning process, considering both direct implications and potential alternative outcomes. The primary decision path shows high confidence in the final classification."
              insights={[
                "High confidence in initial classification",
                "Alternative paths show potential for different outcomes",
                "Context sensitivity identified as key factor"
              ]}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;