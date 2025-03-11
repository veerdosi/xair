import React from 'react';
import { MessageSquare } from 'lucide-react';

interface ExplanationPanelProps {
  explanation: string;
  insights: string[];
}

export const ExplanationPanel: React.FC<ExplanationPanelProps> = ({ explanation, insights }) => {
  return (
    <div className="bg-white rounded-lg shadow-lg p-4">
      <div className="flex items-center gap-2 mb-4">
        <MessageSquare className="w-5 h-5 text-blue-500" />
        <h2 className="text-lg font-semibold">Explanation</h2>
      </div>
      
      <div className="mb-4">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Key Insights</h3>
        <ul className="space-y-2">
          {insights.map((insight, index) => (
            <li key={index} className="flex items-start gap-2">
              <span className="inline-block w-2 h-2 mt-1.5 rounded-full bg-blue-500" />
              <span className="text-sm text-gray-600">{insight}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="prose prose-sm max-w-none">
        <p className="text-gray-600">{explanation}</p>
      </div>
    </div>
  );
};