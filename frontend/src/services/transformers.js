/**
 * Utility functions to transform data between API and UI formats
 */

/**
 * Transform API tree data to D3 compatible format
 * @param {Object} treeData - Tree data from the API
 * @returns {Object} - D3 compatible tree data
 */
export function transformTreeToD3(treeData) {
  if (!treeData || !treeData.visualization) {
    return { nodes: [], links: [] };
  }

  const { nodes, edges } = treeData.visualization;
  
  // Create a map of nodes by ID for quick lookup
  const nodeMap = {};
  nodes.forEach(node => {
    nodeMap[node.id] = {
      ...node,
      children: [],
    };
  });
  
  // Connect nodes based on edges
  edges.forEach(edge => {
    if (nodeMap[edge.source] && nodeMap[edge.target]) {
      // Only add as child if it's a regular edge (not cross-link)
      if (!edge.type || edge.type !== 'cross') {
        nodeMap[edge.source].children.push(nodeMap[edge.target]);
      }
    }
  });
  
  // Find the root node (node without parents)
  let rootNode = null;
  const childIds = new Set(edges.map(e => e.target));
  
  for (const nodeId in nodeMap) {
    if (!childIds.has(nodeId)) {
      rootNode = nodeMap[nodeId];
      break;
    }
  }
  
  return rootNode || { nodes: [], links: [] };
}

/**
 * Transform API impact data to visualization format
 * @param {Array} impacts - Impact data from the API
 * @returns {Object} - Processed impact data for visualizations
 */
export function transformImpactsForVisualization(impacts) {
  if (!impacts || !impacts.length) {
    return {
      averages: {
        local: 0,
        global: 0,
        structural: 0,
        plausibility: 0,
        composite: 0,
      },
      distribution: {
        high: 0,
        medium: 0,
        low: 0,
      },
      topImpacts: [],
    };
  }
  
  // Calculate averages
  const averages = {
    local: impacts.reduce((sum, i) => sum + i.local_impact, 0) / impacts.length,
    global: impacts.reduce((sum, i) => sum + i.global_impact, 0) / impacts.length,
    structural: impacts.reduce((sum, i) => sum + i.structural_impact, 0) / impacts.length,
    plausibility: impacts.reduce((sum, i) => sum + i.plausibility, 0) / impacts.length,
    composite: impacts.reduce((sum, i) => sum + i.composite_score, 0) / impacts.length,
  };
  
  // Calculate impact distribution
  const distribution = {
    high: impacts.filter(i => i.composite_score > 0.7).length,
    medium: impacts.filter(i => i.composite_score > 0.4 && i.composite_score <= 0.7).length,
    low: impacts.filter(i => i.composite_score <= 0.4).length,
  };
  
  // Get top impacts by composite score
  const topImpacts = [...impacts]
    .sort((a, b) => b.composite_score - a.composite_score)
    .slice(0, 3);
  
  return {
    averages,
    distribution,
    topImpacts,
  };
}

/**
 * Transform API counterfactual data with impacts
 * @param {Array} counterfactuals - Counterfactual data from the API
 * @param {Array} impacts - Impact data from the API
 * @returns {Array} - Counterfactuals with impact data
 */
export function mergeCounterfactualsWithImpacts(counterfactuals, impacts) {
  if (!counterfactuals || !counterfactuals.length) {
    return [];
  }
  
  if (!impacts || !impacts.length) {
    return counterfactuals;
  }
  
  // Create a map of impacts by counterfactual ID
  const impactMap = {};
  impacts.forEach(impact => {
    impactMap[impact.counterfactual_id] = impact;
  });
  
  // Merge counterfactuals with impacts
  return counterfactuals.map(cf => ({
    ...cf,
    impact: impactMap[cf.id] || null,
  }));
}

/**
 * Group counterfactuals by modification type
 * @param {Array} counterfactuals - Array of counterfactuals
 * @returns {Object} - Grouped counterfactuals
 */
export function groupCounterfactualsByType(counterfactuals) {
  if (!counterfactuals || !counterfactuals.length) {
    return {};
  }
  
  const groups = {};
  
  counterfactuals.forEach(cf => {
    const type = cf.modification_type || 'unknown';
    if (!groups[type]) {
      groups[type] = [];
    }
    groups[type].push(cf);
  });
  
  return groups;
}

/**
 * Format a modification type to display format
 * @param {string} type - Modification type
 * @returns {string} - Formatted type
 */
export function formatModificationType(type) {
  if (!type) return 'Unknown';
  
  return type
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Format impact score as percentage
 * @param {number} score - Impact score (0-1)
 * @returns {string} - Formatted percentage
 */
export function formatImpactPercentage(score) {
  if (score === undefined || score === null) return '0%';
  return `${Math.round(score * 100)}%`;
}

/**
 * Get CSS class for impact score
 * @param {number} score - Impact score (0-1)
 * @returns {string} - CSS class
 */
export function getImpactScoreClass(score) {
  if (score === undefined || score === null) return 'bg-gray-100 text-gray-800';
  if (score > 0.7) return 'bg-green-100 text-green-800';
  if (score > 0.4) return 'bg-yellow-100 text-yellow-800';
  return 'bg-blue-100 text-blue-800';
}

/**
 * Generate insights based on impact data
 * @param {Array} impacts - Impact data
 * @returns {Array} - Generated insights
 */
export function generateInsightsFromImpacts(impacts) {
  if (!impacts || !impacts.length) {
    return [];
  }
  
  const insights = [];
  
  // Calculate averages
  const avgLocalImpact = impacts.reduce((sum, i) => sum + i.local_impact, 0) / impacts.length;
  const avgGlobalImpact = impacts.reduce((sum, i) => sum + i.global_impact, 0) / impacts.length;
  const avgStructuralImpact = impacts.reduce((sum, i) => sum + i.structural_impact, 0) / impacts.length;
  const avgPlausibility = impacts.reduce((sum, i) => sum + i.plausibility, 0) / impacts.length;
  
  // Generate insights based on data patterns
  if (avgLocalImpact > 0.6) {
    insights.push("Counterfactuals have significant immediate impact on decision points, suggesting the model's reasoning is sensitive to small changes in inputs.");
  } else if (avgLocalImpact < 0.3) {
    insights.push("Counterfactuals have limited immediate impact, suggesting the model's initial decision points are relatively stable.");
  }
  
  if (avgGlobalImpact > 0.6) {
    insights.push("High global impact indicates that small changes significantly affect the final outcomes, highlighting potential instability in the reasoning chain.");
  } else if (avgGlobalImpact < 0.3) {
    insights.push("Low global impact suggests the model reaches similar conclusions despite alternative reasoning paths, indicating robustness in the overall decision process.");
  }
  
  if (avgStructuralImpact > 0.6) {
    insights.push("Counterfactuals cause major changes in reasoning structure, revealing branching paths that lead to significantly different decision trees.");
  }
  
  if (avgPlausibility > 0.7) {
    insights.push("Generated counterfactuals are highly plausible, representing realistic alternative reasoning paths rather than unlikely scenarios.");
  } else if (avgPlausibility < 0.4) {
    insights.push("Counterfactuals have low plausibility, suggesting the model's reasoning paths are firmly established and alternative scenarios are less realistic.");
  }
  
  // Add any missing insights if we don't have enough
  if (insights.length < 3) {
    insights.push("The model's reasoning appears to follow consistent patterns across multiple generation paths, with predictable divergence points.");
  }
  
  return insights;
}