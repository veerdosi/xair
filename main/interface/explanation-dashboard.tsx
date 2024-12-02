import React, { useState, useEffect } from 'react';
import { LineChart, XAxis, YAxis, Tooltip, Line, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Info, AlertTriangle, TrendingUp, Activity } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

const ExplanationDashboard = () => {
  const [expertiseLevel, setExpertiseLevel] = useState(0);
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [timeRange, setTimeRange] = useState([0, 100]);
  const [explanationType, setExplanationType] = useState('basic');
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(false);
  const [modelOutput, setModelOutput] = useState(null);

  // Simulated data - in real implementation, this would come from the backend
  const sampleData = [
    { timestamp: '2024-01', value: 45, prediction: 48 },
    { timestamp: '2024-02', value: 52, prediction: 50 },
    { timestamp: '2024-03', value: 49, prediction: 51 },
  ];

  return (
    <div className="w-full h-full p-4 bg-gray-50">
      <div className="mb-6">
        <CardHeader>
          <CardTitle className="text-2xl font-bold">Model Explanation Dashboard</CardTitle>
        </CardHeader>
      </div>

      <div className="grid grid-cols-12 gap-4">
        {/* Control Panel */}
        <div className="col-span-3">
          <Card>
            <CardHeader>
              <CardTitle>Controls</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Expertise Level</label>
                  <Slider
                    value={[expertiseLevel]}
                    onValueChange={([value]) => setExpertiseLevel(value)}
                    max={2}
                    step={1}
                    className="mt-2"
                  />
                  <div className="text-xs text-gray-500 mt-1">
                    {expertiseLevel === 0 ? 'Basic' : expertiseLevel === 1 ? 'Intermediate' : 'Expert'}
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium">Explanation Type</label>
                  <div className="flex flex-col space-y-2 mt-2">
                    {['basic', 'counterfactual', 'rule-based', 'temporal'].map(type => (
                      <Button
                        key={type}
                        variant={explanationType === type ? 'default' : 'outline'}
                        onClick={() => setExplanationType(type)}
                        className="justify-start"
                      >
                        {type.charAt(0).toUpperCase() + type.slice(1)}
                      </Button>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Area */}
        <div className="col-span-9">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="temporal">Temporal Analysis</TabsTrigger>
              <TabsTrigger value="counterfactual">Counterfactuals</TabsTrigger>
              <TabsTrigger value="rules">Rules</TabsTrigger>
            </TabsList>

            <TabsContent value="overview">
              <Card>
                <CardHeader>
                  <CardTitle>Model Output Overview</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={sampleData}>
                        <XAxis dataKey="timestamp" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="value" stroke="#8884d8" name="Actual" />
                        <Line type="monotone" dataKey="prediction" stroke="#82ca9d" name="Predicted" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {expertiseLevel >= 1 && (
                    <Alert className="mt-4">
                      <Info className="h-4 w-4" />
                      <AlertDescription>
                        Model confidence: 87% - Key features: Temperature, Humidity
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="temporal">
              <Card>
                <CardHeader>
                  <CardTitle>Temporal Pattern Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="h-60">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={sampleData}>
                          <XAxis dataKey="timestamp" />
                          <YAxis />
                          <Tooltip />
                          <Line type="monotone" dataKey="value" stroke="#8884d8" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    <div>
                      <Card>
                        <CardContent className="p-4">
                          <div className="space-y-2">
                            <div className="flex items-center">
                              <TrendingUp className="h-4 w-4 mr-2" />
                              <span>Upward trend detected</span>
                            </div>
                            <div className="flex items-center">
                              <Activity className="h-4 w-4 mr-2" />
                              <span>Seasonal pattern (period: 12)</span>
                            </div>
                            {expertiseLevel >= 2 && (
                              <div className="flex items-center">
                                <AlertTriangle className="h-4 w-4 mr-2" />
                                <span>Anomaly detected at 2024-02</span>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="counterfactual">
              <Card>
                <CardHeader>
                  <CardTitle>Counterfactual Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    {/* Original vs Counterfactual visualization would go here */}
                    <div className="h-60 bg-gray-100 rounded-lg p-4">
                      <div className="text-center">Original Scenario</div>
                    </div>
                    <div className="h-60 bg-gray-100 rounded-lg p-4">
                      <div className="text-center">Counterfactual Scenario</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="rules">
              <Card>
                <CardHeader>
                  <CardTitle>Rule-based Explanation</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {/* Rule visualization would go here */}
                    <Alert>
                      <Info className="h-4 w-4" />
                      <AlertDescription>
                        IF temperature > 25°C AND humidity > 80% THEN prediction increases by 15%
                      </AlertDescription>
                    </Alert>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
};

export default ExplanationDashboard;
