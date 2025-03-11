import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { TreeNode } from '../types';

interface ReasoningTreeProps {
  data: TreeNode;
}

export const ReasoningTree: React.FC<ReasoningTreeProps> = ({ data }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data) return;

    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 90, bottom: 30, left: 90 };

    // Clear previous content
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const tree = d3.tree<TreeNode>()
      .size([height - margin.top - margin.bottom, width - margin.left - margin.right]);

    const root = d3.hierarchy(data);
    const treeData = tree(root);

    // Add links
    g.selectAll(".link")
      .data(treeData.links())
      .enter()
      .append("path")
      .attr("class", "link")
      .attr("d", d3.linkHorizontal()
        .x(d => d.y)
        .y(d => d.x))
      .style("fill", "none")
      .style("stroke", "#ccc")
      .style("stroke-width", "2px");

    // Add nodes
    const node = g.selectAll(".node")
      .data(treeData.descendants())
      .enter()
      .append("g")
      .attr("class", "node")
      .attr("transform", d => `translate(${d.y},${d.x})`);

    // Add circles for nodes
    node.append("circle")
      .attr("r", 10)
      .style("fill", d => (d.data as TreeNode).type === 'actual' ? '#4299e1' : '#ed64a6')
      .style("stroke", "#fff")
      .style("stroke-width", "2px");

    // Add labels
    node.append("text")
      .attr("dy", ".35em")
      .attr("x", d => d.children ? -13 : 13)
      .style("text-anchor", d => d.children ? "end" : "start")
      .text(d => (d.data as TreeNode).label);
  }, [data]);

  return (
    <div className="w-full h-full overflow-auto bg-white rounded-lg shadow-lg">
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
};