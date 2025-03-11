import React from 'react';
import { Counterfactual } from '../types';
import { ArrowUpRight, Lightbulb } from 'lucide-react';

interface CounterfactualPanelProps {
  counterfactuals: Counterfactual[];
  onApply: (id: string) => void;
}

export const CounterfactualPanel: React.FC<CounterfactualPanelProps> = ({ counterfactuals, onApply }) => {
  return (
    <div className="bg-white rounded-lg shadow-lg p-4">
      <div className="flex items-center gap-2 mb-4">
        <Lightbulb className="w-5 h-5 text-yellow-500" />
        <h2 className="text-lg font-semibold">Counterfactuals</h2>
      </div>
      <div className="space-y-3">
        {counterfactuals.sort((a, b) => b.impact - a.impact).map((cf) => (
          <div
            key={cf.id}
            className={`p-3 rounded-lg border ${
              cf.applied ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
            }`}
          >
            <div className="flex justify-between items-start">
              <p className="text-sm text-gray-700">{cf.summary}</p>
              <button
                onClick={() => onApply(cf.id)}
                className={`ml-2 p-2 rounded-full ${
                  cf.applied
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                <ArrowUpRight className="w-4 h-4" />
              </button>
            </div>
            <div className="mt-2 flex items-center gap-2">
              <div className="text-xs text-gray-500">Impact Score:</div>
              <div className="h-2 w-24 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500"
                  style={{ width: `${cf.impact * 100}%` }}
                />
              </div>
              <div className="text-xs text-gray-500">{(cf.impact * 100).toFixed(0)}%</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};