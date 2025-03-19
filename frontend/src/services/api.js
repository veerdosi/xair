const API_BASE_URL = '/api';

async function fetchData(endpoint, options = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.message || 'API request failed');
  }

  return response.json();
}

export async function generateTree(prompt, settings) {
  return fetchData('/generate', {
    method: 'POST',
    body: JSON.stringify({
      prompt,
      temperature: settings.temperature,
      max_tokens: settings.maxTokens,
      max_depth: settings.maxDepth,
      min_probability: settings.minProbability
    })
  });
}

export async function generateCounterfactuals(treeId) {
  return fetchData('/counterfactuals', {
    method: 'POST',
    body: JSON.stringify({ tree_id: treeId })
  });
}

export async function analyzeImpacts(treeId, counterfactualIds = null) {
  return fetchData('/analyze', {
    method: 'POST',
    body: JSON.stringify({
      tree_id: treeId,
      counterfactual_ids: counterfactualIds
    })
  });
}