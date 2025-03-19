import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../contexts/AppContext';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Slider } from '../components/ui/Slider';

const Settings = () => {
  const { state, dispatch } = useAppContext();
  const { settings } = state;
  const [formValues, setFormValues] = React.useState(settings);
  const navigate = useNavigate();

  const handleChange = (name, value) => {
    setFormValues({
      ...formValues,
      [name]: value,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    dispatch({
      type: 'UPDATE_SETTINGS',
      payload: formValues,
    });
    
    navigate(-1);
  };

  return (
    <div className="container mx-auto max-w-2xl py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-2">Settings</h1>
        <p className="text-gray-600">
          Configure the parameters for tree generation and counterfactual analysis
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Generation Settings</CardTitle>
          <CardDescription>
            Parameters for the LLM and reasoning tree generation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Temperature
                </label>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[formValues.temperature]}
                    min={0.1}
                    max={2.0}
                    step={0.1}
                    onValueChange={([value]) => handleChange('temperature', value)}
                    className="flex-1"
                  />
                  <span className="text-sm text-gray-500 w-12 text-right">
                    {formValues.temperature.toFixed(1)}
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Higher values (0.7-2.0) increase randomness, lower values (0.1-0.6) make output more deterministic
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Max Tokens
                </label>
                <Input
                  type="number"
                  min={100}
                  max={4000}
                  value={formValues.maxTokens}
                  onChange={(e) => handleChange('maxTokens', parseInt(e.target.value))}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Maximum number of tokens to generate (100-4000)
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Max Tree Depth
                </label>
                <Input
                  type="number"
                  min={1}
                  max={10}
                  value={formValues.maxDepth}
                  onChange={(e) => handleChange('maxDepth', parseInt(e.target.value))}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Maximum depth of the reasoning tree (1-10)
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Minimum Probability Threshold
                </label>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[formValues.minProbability]}
                    min={0.01}
                    max={0.5}
                    step={0.01}
                    onValueChange={([value]) => handleChange('minProbability', value)}
                    className="flex-1"
                  />
                  <span className="text-sm text-gray-500 w-12 text-right">
                    {formValues.minProbability.toFixed(2)}
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Minimum probability threshold for including nodes (0.01-0.5)
                </p>
              </div>
            </div>

            <div className="flex gap-3">
              <Button type="submit">
                Save Settings
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => navigate(-1)}
              >
                Cancel
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default Settings;