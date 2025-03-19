import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/Tabs';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Button } from '../ui/Button';
import { Slider } from '../ui/Slider';
import { useAppContext } from '../../contexts/AppContext';

const COLORS = {
  mainPath: '#2563eb',  // Blue
  counterfactual: '#f97316',  // Orange
  highlight: '#10b981',  // Green
  lowImpact: '#94a3b8',  // Gray
};

const ReasoningTree = ({ treeId, onNodeSelect }) => {
  const { state } = useAppContext();
  const svgRef = useRef(null);
  const [viewMode, setViewMode] = useState('tree');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [selectedNode, setSelectedNode] = useState(null);
  const [filters, setFilters] = useState({
    showCounterfactuals: true,
    minImpact: 0,
    nodeTypes: ['all']
  });

  const tree = state.trees[treeId];
  const counterfactuals = state.counterfactuals[treeId] || [];
  const impacts = state.impacts[treeId] || [];

  useEffect(() => {
    if (!tree || !svgRef.current) return;

    // Clear previous visualization
    d3.select(svgRef.current).selectAll('*').remove();

    // Set up dimensions
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const margin = { top: 40, right: 120, bottom: 40, left: 120 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Create container for the tree
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Process tree data
    const root = d3.hierarchy(processTreeData(tree.visualization));

    // Create tree layout
    const treeLayout = d3.tree()
      .size([innerHeight, innerWidth]);

    // Apply layout
    treeLayout(root);

    // Draw links
    g.selectAll('.link')
      .data(root.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', d3.linkHorizontal()
        .x(d => d.y)
        .y(d => d.x))
      .attr('fill', 'none')
      .attr('stroke', d => getLinkColor(d))
      .attr('stroke-width', d => getStrokeWidth(d))
      .attr('opacity', 0.6);

    // Draw nodes
    const nodes = g.selectAll('.node')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.y},${d.x})`)
      .on('click', (event, d) => {
        setSelectedNode(d.data);
        onNodeSelect?.(d.data);
      });

    // Add node circles
    nodes.append('circle')
      .attr('r', 8)
      .attr('fill', d => getNodeColor(d.data))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    // Add node labels
    nodes.append('text')
      .attr('dy', '.31em')
      .attr('x', d => d.children ? -12 : 12)
      .attr('text-anchor', d => d.children ? 'end' : 'start')
      .text(d => truncateText(d.data.text))
      .style('font-size', '12px')
      .style('font-family', 'sans-serif');

    // Set up zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 3])
      .on('zoom', (event) => {
        setZoomLevel(event.transform.k);
        g.attr('transform', event.transform);
      });

    svg.call(zoom)
      .on('dblclick.zoom', null);

    // Apply initial zoom to fit the tree
    const initialZoom = 0.8;
    svg.call(zoom.transform, d3.zoomIdentity
      .translate(width / 4, height / 2)
      .scale(initialZoom));

    // Apply filters
    applyFilters(filters);

  }, [tree, counterfactuals, impacts]);

  // Helper functions
  const processTreeData = (visualization) => {
    const { nodes, edges } = visualization;
    
    // Create adjacency list
    const adjList = {};
    nodes.forEach(node => {
      adjList[node.id] = { ...node, children: [] };
    });
    
    // Connect nodes
    edges.forEach(edge => {
      if (adjList[edge.source] && adjList[edge.target]) {
        adjList[edge.source].children.push(adjList[edge.target]);
      }
    });
    
    // Return root node
    return adjList[nodes[0]?.id] || {};
  };

  const getNodeColor = (node) => {
    if (node.is_counterfactual) {
      return COLORS.counterfactual;
    }
    
    // Get impact score for this node
    const nodeImpacts = impacts.filter(impact => 
      impact.affected_nodes.includes(node.id)
    );
    
    if (nodeImpacts.length > 0) {
      const maxImpact = Math.max(...nodeImpacts.map(i => i.composite_score));
      if (maxImpact > 0.7) {
        return COLORS.highlight;
      }
    }
    
    return node.probability > 0.5 ? COLORS.mainPath : COLORS.lowImpact;
  };

  const getLinkColor = (link) => {
    const source = link.source.data;
    const target = link.target.data;
    
    if (source.is_counterfactual || target.is_counterfactual) {
      return COLORS.counterfactual;
    }
    
    return COLORS.mainPath;
  };

  const getStrokeWidth = (link) => {
    const target = link.target.data;
    return 1 + (target.probability * 2);
  };

  const truncateText = (text) => {
    return text.length > 20 ? text.substring(0, 20) + '...' : text;
  };

  const applyFilters = (filters) => {
    if (!svgRef.current) return;
    
    const { showCounterfactuals, minImpact, nodeTypes } = filters;
    
    // Apply node filters
    d3.select(svgRef.current).selectAll('.node')
      .style('display', function() {
        const nodeData = d3.select(this).datum().data;
        
        if (!showCounterfactuals && nodeData.is_counterfactual) {
          return 'none';
        }
        
        // Filter by impact
        const nodeImpacts = impacts.filter(impact => 
          impact.affected_nodes.includes(nodeData.id)
        );
        
        const maxImpact = nodeImpacts.length > 0 
          ? Math.max(...nodeImpacts.map(i => i.composite_score))
          : 0;
          
        if (maxImpact < minImpact) {
          return 'none';
        }
        
        // Filter by node type
        if (nodeTypes[0] !== 'all') {
          if (nodeData.is_counterfactual && !nodeTypes.includes('counterfactual')) {
            return 'none';
          }
          
          if (!nodeData.is_counterfactual && !nodeTypes.includes('main')) {
            return 'none';
          }
        }
        
        return null;
      });
      
    // Apply link filters
    d3.select(svgRef.current).selectAll('.link')
      .style('display', function() {
        const linkData = d3.select(this).datum();
        const source = linkData.source.data;
        const target = linkData.target.data;
        
        if (!showCounterfactuals && (source.is_counterfactual || target.is_counterfactual)) {
          return 'none';
        }
        
        return null;
      });
  };

  const handleFilterChange = (newFilters) => {
    const updatedFilters = { ...filters, ...newFilters };
    setFilters(updatedFilters);
    applyFilters(updatedFilters);
  };

  return (
    <Card className="w-full h-full">
      <CardHeader>
        <CardTitle>Reasoning Tree Visualization</CardTitle>
        <div className="flex flex-wrap gap-4">
          <Tabs defaultValue="tree" value={viewMode} onValueChange={setViewMode}>
            <TabsList>
              <TabsTrigger value="tree">Tree View</TabsTrigger>
              <TabsTrigger value="list">List View</TabsTrigger>
            </TabsList>
          </Tabs>
          
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                const newZoom = zoomLevel * 1.2;
                setZoomLevel(newZoom);
                applyZoom(newZoom);
              }}
            >
              Zoom In
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                const newZoom = zoomLevel / 1.2;
                setZoomLevel(newZoom);
                applyZoom(newZoom);
              }}
            >
              Zoom Out
            </Button>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm">Show Counterfactuals:</span>
            <input
              type="checkbox"
              checked={filters.showCounterfactuals}
              onChange={(e) => handleFilterChange({ showCounterfactuals: e.target.checked })}
            />
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm">Min Impact:</span>
            <Slider
              defaultValue={[0]}
              max={1}
              step={0.1}
              value={[filters.minImpact]}
              onValueChange={([value]) => handleFilterChange({ minImpact: value })}
              className="w-32"
            />
            <span className="text-sm">{filters.minImpact.toFixed(1)}</span>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <div className="relative w-full h-[600px] bg-gray-50 rounded-lg overflow-hidden">
          <svg
            ref={svgRef}
            className="w-full h-full"
          />
          
          {selectedNode && (
            <div className="absolute bottom-4 right-4 p-4 bg-white rounded-lg shadow-lg max-w-xs">
              <h3 className="font-medium text-lg mb-2 truncate">{selectedNode.text}</h3>
              <p className="text-sm text-gray-600 mb-1">
                Probability: {(selectedNode.probability * 100).toFixed(1)}%
              </p>
              
              {selectedNode.importance_score !== undefined && (
                <p className="text-sm text-gray-600 mb-1">
                  Importance: {(selectedNode.importance_score * 100).toFixed(1)}%
                </p>
              )}
              
              {selectedNode.is_counterfactual && (
                <div className="mt-2">
                  <p className="text-xs text-amber-600 font-medium mb-1">Counterfactual</p>
                  {/* Add more counterfactual-specific info here */}
                </div>
              )}
              
              <Button
                variant="outline"
                size="sm"
                className="mt-2 w-full"
                onClick={() => setSelectedNode(null)}
              >
                Close
              </Button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// Function to apply zoom programmatically
const applyZoom = (zoomLevel) => {
  if (!svgRef.current) return;
  
  const svg = d3.select(svgRef.current);
  const width = svgRef.current.clientWidth;
  const height = svgRef.current.clientHeight;
  
  svg.call(d3.zoom().transform, d3.zoomIdentity
    .translate(width / 2, height / 2)
    .scale(zoomLevel)
    .translate(-width / 2, -height / 2));
};

export default ReasoningTree;