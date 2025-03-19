import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../ui/Tabs';
import { useAppContext } from '../../contexts/AppContext';

const ImpactAnalysis = ({ treeId }) => {
  const { state } = useAppContext();
  const [activeTab, setActiveTab] = useState('overview');
  const [sortedImpacts, setSortedImpacts] = useState([]);
  
  const impacts = state.impacts[treeId] || [];
  const counterfactuals = state.counterfactuals[treeId] || [];
  
  useEffect(() => {
    // Sort impacts by composite score (descending)
    const sorted = [...impacts].sort((a, b) => b.composite_score - a.composite_score);
    setSortedImpacts(sorted);
  }, [impacts]);

  // Get counterfactual details by ID
  const getCounterfactualById = (id) => {
    return counterfactuals.find(cf => cf.id === id) || {};
  };
  
  // Format impact percentage
  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(0)}%`;
  };
  
  // Get color class for impact score
  const getImpactColorClass = (score) => {
    if (score > 0.7) return 'bg-green-100 text-green-800';
    if (score > 0.4) return 'bg-yellow-100 text-yellow-800';
    return 'bg-blue-100 text-blue-800';
  };
  
  // Calculate averages
  const calculateAverages = () => {
    if (impacts.length === 0) return { local: 0, global: 0, structural: 0, plausibility: 0 };
    
    return {
      local: impacts.reduce((sum, i) => sum + i.local_impact, 0) / impacts.length,
      global: impacts.reduce((sum, i) => sum + i.global_impact, 0) / impacts.length,
      structural: impacts.reduce((sum, i) => sum + i.structural_impact, 0) / impacts.length,
      plausibility: impacts.reduce((sum, i) => sum + i.plausibility, 0) / impacts.length,
      composite: impacts.reduce((sum, i) => sum + i.composite_score, 0) / impacts.length
    };
  };
  
  const averages = calculateAverages();
  
  // Generate insights based on impact data
  const generateInsights = () => {
    if (impacts.length === 0) return [];
    
    const insights = [];
    
    // Analyze local impact
    if (averages.local > 0.6) {
      insights.push("Counterfactuals have significant immediate impact on decision points, suggesting the model's reasoning is sensitive to small changes in inputs.");
    } else if (averages.local < 0.3) {
      insights.push("Counterfactuals have limited immediate impact, suggesting the model's initial decision points are relatively stable.");
    }
    
    // Analyze global impact
    if (averages.global > 0.6) {
      insights.push("High global impact indicates that small changes significantly affect the final outcomes, highlighting potential instability in the reasoning chain.");
    } else if (averages.global < 0.3) {
      insights.push("Low global impact suggests the model reaches similar conclusions despite alternative reasoning paths, indicating robustness in the overall decision process.");
    }
    
    // Analyze structural impact
    if (averages.structural > 0.6) {
      insights.push("Counterfactuals cause major changes in reasoning structure, revealing branching paths that lead to significantly different decision trees.");
    }
    
    // Analyze plausibility
    if (averages.plausibility > 0.7) {
      insights.push("Generated counterfactuals are highly plausible, representing realistic alternative reasoning paths rather than unlikely scenarios.");
    } else if (averages.plausibility < 0.4) {
      insights.push("Counterfactuals have low plausibility, suggesting the model's reasoning paths are firmly established and alternative scenarios are less realistic.");
    }
    
    // Analyze specific patterns
    const highImpactCfs = impacts.filter(i => i.composite_score > 0.7);
    if (highImpactCfs.length > 0) {
      const percent = Math.round((highImpactCfs.length / impacts.length) * 100);
      insights.push(`${percent}% of counterfactuals have high overall impact, indicating significant decision points where alternative reasoning could lead to different outcomes.`);
    }
    
    // Add most affected nodes insight
    const mostAffectedCount = Math.max(...impacts.map(i => i.affected_nodes?.length || 0));
    if (mostAffectedCount > 0) {
      insights.push(`The most impactful counterfactual affects ${mostAffectedCount} nodes in the reasoning tree, highlighting the cascading nature of alternative decisions.`);
    }
    
    // Add any missing insights if we don't have enough
    if (insights.length < 3) {
      insights.push("The model's reasoning appears to follow consistent patterns across multiple generation paths, with predictable divergence points.");
    }
    
    return insights;
  };
  
  const insights = generateInsights();
  
  // Find counterfactual with highest impact in each category
  const findHighestImpactCf = (category) => {
    if (impacts.length === 0) return null;
    
    let highest = impacts[0];
    
    impacts.forEach(impact => {
      if (impact[category] > highest[category]) {
        highest = impact;
      }
    });
    
    return highest;
  };
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Impact Analysis</CardTitle>
        <CardDescription>
          Understand how counterfactuals affect the reasoning process
        </CardDescription>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="counterfactuals">Counterfactuals</TabsTrigger>
            <TabsTrigger value="insights">Key Insights</TabsTrigger>
          </TabsList>
        </Tabs>
      </CardHeader>
      <CardContent>
        {impacts.length === 0 ? (
          <div className="text-center py-6 text-gray-500">
            No impact analysis available yet
          </div>
        ) : (
          <>
            <TabsContent value="overview" className="mt-0">
              <div className="space-y-6">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <Card className="bg-blue-50 border-blue-100">
                    <CardContent className="p-4">
                      <div className="text-sm font-medium text-blue-700 mb-1">
                        Average Local Impact
                      </div>
                      <div className="text-2xl font-bold">
                        {formatPercentage(averages.local)}
                      </div>
                      <div className="text-xs text-blue-600 mt-1">
                        Immediate changes to decision points
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-purple-50 border-purple-100">
                    <CardContent className="p-4">
                      <div className="text-sm font-medium text-purple-700 mb-1">
                        Average Global Impact
                      </div>
                      <div className="text-2xl font-bold">
                        {formatPercentage(averages.global)}
                      </div>
                      <div className="text-xs text-purple-600 mt-1">
                        Effects on final outcomes
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-indigo-50 border-indigo-100">
                    <CardContent className="p-4">
                      <div className="text-sm font-medium text-indigo-700 mb-1">
                        Average Structural Impact
                      </div>
                      <div className="text-2xl font-bold">
                        {formatPercentage(averages.structural)}
                      </div>
                      <div className="text-xs text-indigo-600 mt-1">
                        Changes to reasoning structure
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="bg-emerald-50 border-emerald-100">
                    <CardContent className="p-4">
                      <div className="text-sm font-medium text-emerald-700 mb-1">
                        Average Plausibility
                      </div>
                      <div className="text-2xl font-bold">
                        {formatPercentage(averages.plausibility)}
                      </div>
                      <div className="text-xs text-emerald-600 mt-1">
                        How realistic the alternatives are
                      </div>
                    </CardContent>
                  </Card>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-3">Impact Distribution</h3>
                  <div className="bg-white p-4 rounded-lg border border-gray-200">
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Local Impact</span>
                          <span>{formatPercentage(averages.local)}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-blue-600 h-2.5 rounded-full impact-chart-bar" 
                            style={{ width: formatPercentage(averages.local) }}
                          ></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Global Impact</span>
                          <span>{formatPercentage(averages.global)}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-purple-600 h-2.5 rounded-full impact-chart-bar" 
                            style={{ width: formatPercentage(averages.global) }}
                          ></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Structural Impact</span>
                          <span>{formatPercentage(averages.structural)}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-indigo-600 h-2.5 rounded-full impact-chart-bar" 
                            style={{ width: formatPercentage(averages.structural) }}
                          ></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Plausibility</span>
                          <span>{formatPercentage(averages.plausibility)}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-emerald-600 h-2.5 rounded-full impact-chart-bar" 
                            style={{ width: formatPercentage(averages.plausibility) }}
                          ></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="font-medium">Overall Impact</span>
                          <span className="font-medium">{formatPercentage(averages.composite)}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3">
                          <div 
                            className="bg-gradient-to-r from-blue-500 to-purple-600 h-3 rounded-full impact-chart-bar" 
                            style={{ width: formatPercentage(averages.composite) }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-3">Highest Impact Categories</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {['local_impact', 'global_impact'].map(category => {
                      const highestImpact = findHighestImpactCf(category);
                      if (!highestImpact) return null;
                      
                      const cf = getCounterfactualById(highestImpact.counterfactual_id);
                      const categoryName = category.split('_').map(word => 
                        word.charAt(0).toUpperCase() + word.slice(1)
                      ).join(' ');
                      
                      return (
                        <Card key={category} className="overflow-hidden">
                          <CardHeader className="p-4 bg-gray-50">
                            <CardTitle className="text-base">Highest {categoryName}</CardTitle>
                            <div className="flex items-center">
                              <Badge className={getImpactColorClass(highestImpact[category])}>
                                {formatPercentage(highestImpact[category])}
                              </Badge>
                            </div>
                          </CardHeader>
                          <CardContent className="p-4">
                            <div className="text-sm">
                              <div className="font-medium mb-1">Original Text:</div>
                              <div className="text-gray-600 mb-3">{cf.original_text}</div>
                              <div className="font-medium mb-1">Counterfactual:</div>
                              <div className="text-gray-600">{cf.modified_text}</div>
                            </div>
                          </CardContent>
                        </Card>
                      );
                    })}
                  </div>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="counterfactuals" className="mt-0">
              <div className="space-y-4">
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 mb-4">
                  <div className="text-sm text-gray-600 mb-2">
                    Showing all counterfactuals, sorted by overall impact score
                  </div>
                  <div className="flex items-center text-xs text-gray-500">
                    <div className="flex items-center mr-4">
                      <div className="w-3 h-3 rounded-full bg-green-500 mr-1"></div>
                      <span>High Impact (70-100%)</span>
                    </div>
                    <div className="flex items-center mr-4">
                      <div className="w-3 h-3 rounded-full bg-yellow-500 mr-1"></div>
                      <span>Medium Impact (40-70%)</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 rounded-full bg-blue-500 mr-1"></div>
                      <span>Low Impact (0-40%)</span>
                    </div>
                  </div>
                </div>
                
                {sortedImpacts.map((impact) => {
                  const cf = getCounterfactualById(impact.counterfactual_id);
                  
                  return (
                    <Card key={impact.counterfactual_id} className="overflow-hidden">
                      <div className="p-4">
                        <div className="flex justify-between items-start mb-3">
                          <Badge variant="outline">
                            {cf.modification_type?.replace(/_/g, ' ')}
                          </Badge>
                          <Badge className={getImpactColorClass(impact.composite_score)}>
                            Overall Impact: {formatPercentage(impact.composite_score)}
                          </Badge>
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                          <div className="p-3 bg-gray-50 rounded text-sm">
                            <div className="text-xs font-medium text-gray-500 mb-1">Original</div>
                            {cf.original_text}
                          </div>
                          <div className="p-3 bg-orange-50 rounded text-sm">
                            <div className="text-xs font-medium text-orange-500 mb-1">Counterfactual</div>
                            {cf.modified_text}
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                          <div className="flex justify-between p-2 bg-blue-50 rounded">
                            <span>Local:</span>
                            <span className="font-medium">{formatPercentage(impact.local_impact)}</span>
                          </div>
                          <div className="flex justify-between p-2 bg-purple-50 rounded">
                            <span>Global:</span>
                            <span className="font-medium">{formatPercentage(impact.global_impact)}</span>
                          </div>
                          <div className="flex justify-between p-2 bg-indigo-50 rounded">
                            <span>Structural:</span>
                            <span className="font-medium">{formatPercentage(impact.structural_impact)}</span>
                          </div>
                          <div className="flex justify-between p-2 bg-emerald-50 rounded">
                            <span>Plausibility:</span>
                            <span className="font-medium">{formatPercentage(impact.plausibility)}</span>
                          </div>
                        </div>
                        
                        {impact.affected_nodes?.length > 0 && (
                          <div className="mt-3 text-xs text-gray-500">
                            Affects {impact.affected_nodes.length} node{impact.affected_nodes.length !== 1 ? 's' : ''} in the reasoning tree
                          </div>
                        )}
                      </div>
                    </Card>
                  );
                })}
              </div>
            </TabsContent>
            
            <TabsContent value="insights" className="mt-0">
              <div className="space-y-6">
                <div className="bg-white p-4 rounded-lg border border-gray-200">
                  <h3 className="text-lg font-medium mb-3">Key Insights</h3>
                  <div className="space-y-3">
                    {insights.map((insight, idx) => (
                      <div key={idx} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                        <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center text-blue-700">
                          <span className="text-xs">{idx + 1}</span>
                        </div>
                        <div className="text-sm">{insight}</div>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-3">Counterfactual Effectiveness</h3>
                  <div className="bg-white p-4 rounded-lg border border-gray-200">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <div className="text-sm font-medium mb-2">Impact vs. Plausibility</div>
                        <div className="aspect-square bg-gray-50 rounded-lg flex items-center justify-center">
                          <div className="text-center text-gray-500 text-sm p-4">
                            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-2 text-gray-400">
                              <circle cx="12" cy="12" r="10"/>
                              <line x1="12" y1="8" x2="12" y2="12"/>
                              <line x1="12" y1="16" x2="12.01" y2="16"/>
                            </svg>
                            Scatter plot visualization would appear here in the full implementation
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="text-sm font-medium mb-2">Most Effective Modification Types</div>
                        <div className="aspect-square bg-gray-50 rounded-lg flex items-center justify-center">
                          <div className="text-center text-gray-500 text-sm p-4">
                            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-2 text-gray-400">
                              <path d="M3 3v18h18"/>
                              <path d="M18 17V9"/>
                              <path d="M13 17V5"/>
                              <path d="M8 17v-3"/>
                            </svg>
                            Bar chart visualization would appear here in the full implementation
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-3">Recommendations</h3>
                  <div className="space-y-3">
                    <Card className="p-4">
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-700">
                          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M12 2v8"></path>
                            <path d="M12 18v4"></path>
                            <circle cx="12" cy="12" r="8"></circle>
                            <path d="M12 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"></path>
                          </svg>
                        </div>
                        <div>
                          <div className="font-medium mb-1">Focus on high-impact counterfactuals</div>
                          <div className="text-sm text-gray-600">
                            {impacts.filter(i => i.composite_score > 0.7).length > 0 
                              ? `The ${impacts.filter(i => i.composite_score > 0.7).length} highest-impact counterfactuals reveal critical decision points in the model's reasoning process.`
                              : "Focus on generating more diverse counterfactuals to better understand the model's reasoning process."}
                          </div>
                        </div>
                      </div>
                    </Card>
                    
                    <Card className="p-4">
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-700">
                          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 11V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h6"></path>
                            <path d="m12 12 4 10 1.7-4.3L22 16Z"></path>
                          </svg>
                        </div>
                        <div>
                          <div className="font-medium mb-1">Improve model robustness</div>
                          <div className="text-sm text-gray-600">
                            {averages.global > 0.6 
                              ? "The high global impact scores suggest the model could benefit from training on more diverse examples to improve stability in reasoning paths."
                              : "The model demonstrates relatively stable reasoning paths, but there's still room for improvement in handling edge cases."}
                          </div>
                        </div>
                      </div>
                    </Card>
                  </div>
                </div>
              </div>
            </TabsContent>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default ImpactAnalysis;