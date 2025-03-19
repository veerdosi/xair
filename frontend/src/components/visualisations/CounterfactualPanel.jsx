import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/Card';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { useAppContext } from '../../contexts/AppContext';

const CounterfactualPanel = ({ treeId, onSelect }) => {
  const { state } = useAppContext();
  const [selectedCf, setSelectedCf] = useState(null);
  
  const counterfactuals = state.counterfactuals[treeId] || [];
  const impacts = state.impacts[treeId] || [];
  
  // Join counterfactuals with impact data
  const cfWithImpacts = counterfactuals.map(cf => {
    const impact = impacts.find(i => i.counterfactual_id === cf.id);
    return { ...cf, impact };
  });
  
  // Sort by impact score (descending)
  const sortedCfs = [...cfWithImpacts].sort((a, b) => {
    return (b.impact?.composite_score || 0) - (a.impact?.composite_score || 0);
  });
  
  const handleSelect = (cf) => {
    setSelectedCf(cf.id === selectedCf ? null : cf.id);
    onSelect?.(cf);
  };
  
  const getImpactColor = (score) => {
    if (!score && score !== 0) return 'bg-gray-200';
    if (score > 0.7) return 'bg-green-100 text-green-800';
    if (score > 0.4) return 'bg-yellow-100 text-yellow-800';
    return 'bg-blue-100 text-blue-800';
  };
  
  const formatModificationType = (type) => {
    return type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Counterfactual Analysis</CardTitle>
        <CardDescription>
          Explore alternative reasoning paths and their impacts
        </CardDescription>
      </CardHeader>
      <CardContent>
        {sortedCfs.length === 0 ? (
          <div className="text-center py-6 text-gray-500">
            No counterfactuals generated yet
          </div>
        ) : (
          <div className="space-y-4">
            {sortedCfs.map(cf => (
              <Card 
                key={cf.id}
                className={`overflow-hidden transition-all cursor-pointer hover:shadow-md ${
                  selectedCf === cf.id ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => handleSelect(cf)}
              >
                <CardContent className="p-4">
                  <div className="flex justify-between items-start mb-2">
                    <Badge variant="outline" className="mr-2">
                      {formatModificationType(cf.modification_type)}
                    </Badge>
                    {cf.impact && (
                      <Badge className={getImpactColor(cf.impact.composite_score)}>
                        Impact: {(cf.impact.composite_score * 100).toFixed(0)}%
                      </Badge>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                    <div className="p-3 bg-gray-50 rounded text-sm">
                      <div className="text-xs font-medium text-gray-500 mb-1">Original</div>
                      {cf.original_text}
                    </div>
                    <div className="p-3 bg-orange-50 rounded text-sm">
                      <div className="text-xs font-medium text-orange-500 mb-1">Counterfactual</div>
                      {cf.modified_text}
                    </div>
                  </div>
                  
                  {cf.impact && selectedCf === cf.id && (
                    <div className="mt-4 pt-4 border-t border-gray-100">
                      <div className="text-sm font-medium mb-2">Impact Analysis</div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="flex justify-between p-2 bg-gray-50 rounded">
                          <span>Local Impact:</span>
                          <span className="font-medium">{(cf.impact.local_impact * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between p-2 bg-gray-50 rounded">
                          <span>Global Impact:</span>
                          <span className="font-medium">{(cf.impact.global_impact * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between p-2 bg-gray-50 rounded">
                          <span>Structural Impact:</span>
                          <span className="font-medium">{(cf.impact.structural_impact * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between p-2 bg-gray-50 rounded">
                          <span>Plausibility:</span>
                          <span className="font-medium">{(cf.impact.plausibility * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                      
                      <div className="mt-3 text-xs text-gray-500">
                        Affects {cf.impact.affected_nodes?.length || 0} nodes in the reasoning tree
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default CounterfactualPanel;