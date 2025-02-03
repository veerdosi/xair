import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { TreeChart } from 'dependentree';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Select } from '@/components/ui/select';
import { Tabs, TabList, Tab, TabPanel } from '@/components/ui/tabs';

const COLORS = {
  mainPath: '#2563eb',  // Blue
  counterfactual: '#f97316',  // Orange
  highlight: '#10b981',  // Green
  lowImpact: '#94a3b8',  // Gray
};

const TreeVisualizer = ({
  reasoningTree,
  counterfactuals,
  impactScores,
  onNodeSelect,
  onCounterfactualToggle
}) => {
  const svgRef = useRef(null);
  const [viewMode, setViewMode] = useState('tree');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [selectedNode, setSelectedNode] = useState(null);
  const [filters, setFilters] = useState({
    showCounterfactuals: true,
    minImpact: 0,
    nodeTypes: ['all']
  });

  useEffect(() => {
    if (!reasoningTree || !svgRef.current) return;

    // Transform data for DependenTree
    const treeData = transformData(reasoningTree, counterfactuals, impactScores);
    
    // Initialize tree visualization
    const chart = new TreeChart(svgRef.current, {
      data: treeData,
      nodeRadius: 8,
      linkWidth: 2,
      duration: 750,
      layout: 'vertical',
      nodeColor: getNodeColor,
      linkColor: getLinkColor,
      allowCycles: true,
      responsive: true
    });

    // Set up zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 3])
      .on('zoom', (event) => {
        setZoomLevel(event.transform.k);
        chart.container.attr('transform', event.transform);
      });

    d3.select(svgRef.current)
      .call(zoom)
      .on('dblclick.zoom', null);

    // Set up node interactions
    chart.onNodeClick((node) => {
      setSelectedNode(node);
      onNodeSelect?.(node);
    });

    // Set up node expansion
    chart.onNodeExpand((node) => {
      if (node.data.hasChildren && !node.children) {
        node.children = node.data._children;
        chart.update();
      }
    });

    // Apply filters
    applyFilters(chart, filters);

    return () => {
      // Cleanup
      chart.destroy();
    };
  }, [reasoningTree, counterfactuals, impactScores]);

  const transformData = (tree, counterfactuals, impactScores) => {
    const nodes = [];
    const links = [];
    const processNode = (node, parent = null) => {
      const nodeData = {
        id: node.id,
        text: node.text,
        probability: node.probability,
        attention: node.attention_weight,
        isCounterfactual: node.is_counterfactual,
        impact: impactScores[node.id]?.composite_score || 0,
        hasChildren: node.children_ids?.size > 0,
        _children: [], // For lazy loading
        parent: parent?.id
      };

      nodes.push(nodeData);
      if (parent) {
        links.push({
          source: parent.id,
          target: node.id,
          weight: node.probability
        });
      }

      node.children_ids?.forEach(childId => {
        const childNode = tree.nodes[childId].node;
        processNode(childNode, nodeData);
      });
    };

    processNode(tree.nodes[Object.keys(tree.nodes)[0]].node);
    
    return { nodes, links };
  };

  const getNodeColor = (node) => {
    if (node.data.isCounterfactual) {
      return COLORS.counterfactual;
    }
    if (node.data.impact > 0.7) {
      return COLORS.highlight;
    }
    return node.data.probability > 0.5 ? COLORS.mainPath : COLORS.lowImpact;
  };

  const getLinkColor = (link) => {
    const source = link.source.data;
    const target = link.target.data;
    if (source.isCounterfactual || target.isCounterfactual) {
      return COLORS.counterfactual;
    }
    return COLORS.mainPath;
  };

  const applyFilters = (chart, filters) => {
    const { showCounterfactuals, minImpact, nodeTypes } = filters;
    
    chart.nodes.style('display', node => {
      if (!showCounterfactuals && node.data.isCounterfactual) return 'none';
      if (node.data.impact < minImpact) return 'none';
      if (nodeTypes[0] !== 'all' && !nodeTypes.includes(node.data.type)) {
        return 'none';
      }
      return null;
    });

    chart.links.style('display', link => {
      const source = link.source.data;
      const target = link.target.data;
      if (!showCounterfactuals && (source.isCounterfactual || target.isCounterfactual)) {
        return 'none';
      }
      return null;
    });
  };

  return (
    <Card className="w-full h-full">
      <CardHeader>
        <CardTitle>Reasoning Tree Visualization</CardTitle>
        <div className="flex space-x-4">
          <Tabs value={viewMode} onValueChange={setViewMode}>
            <TabList>
              <Tab value="tree">Tree View</Tab>
              <Tab value="list">List View</Tab>
            </TabList>
          </Tabs>
          
          <div className="flex items-center space-x-2">
            <Button
              size="sm"
              onClick={() => setZoomLevel(prev => prev * 1.2)}
            >
              Zoom In
            </Button>
            <Button
              size="sm"
              onClick={() => setZoomLevel(prev => prev / 1.2)}
            >
              Zoom Out
            </Button>
          </div>

          <Select
            value={filters.nodeTypes}
            onChange={(value) => setFilters(prev => ({
              ...prev,
              nodeTypes: value
            }))}
            multiple
          >
            <option value="all">All Types</option>
            <option value="main">Main Path</option>
            <option value="counterfactual">Counterfactuals</option>
          </Select>

          <div className="flex items-center space-x-2">
            <span>Min Impact:</span>
            <Slider
              value={[filters.minImpact]}
              min={0}
              max={1}
              step={0.1}
              onValueChange={([value]) => setFilters(prev => ({
                ...prev,
                minImpact: value
              }))}
            />
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <div className="relative w-full h-[600px]">
          <svg
            ref={svgRef}
            className="w-full h-full"
            style={{
              backgroundColor: '#f8fafc',
              borderRadius: '0.5rem'
            }}
          />
          
          {selectedNode && (
            <div className="absolute bottom-4 right-4 p-4 bg-white rounded-lg shadow-lg">
              <h3 className="font-medium">{selectedNode.data.text}</h3>
              <p>Probability: {(selectedNode.data.probability * 100).toFixed(1)}%</p>
              <p>Impact: {(selectedNode.data.impact * 100).toFixed(1)}%</p>
              {selectedNode.data.isCounterfactual && (
                <Button
                  size="sm"
                  onClick={() => onCounterfactualToggle?.(selectedNode.data.id)}
                >
                  Toggle Counterfactual
                </Button>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default TreeVisualizer;