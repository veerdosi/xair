/**
 * Visualization module for the XAI Chat Interface
 * Handles rendering of different visualization types using D3.js
 */

class XAIVisualization {
    constructor(containerId, config) {
        this.container = document.getElementById(containerId);
        this.config = config;
        this.currentVizType = config.visualization.defaultType;
        this.currentData = null;
        this.svg = null;
        this.tooltip = null;
        this.selectedNode = null;
        this.importanceThreshold = config.visualization.importanceThreshold;
        
        // Initialize tooltip
        this.createTooltip();
    }
    
    /**
     * Create a tooltip element for interactive visualizations
     */
    createTooltip() {
        this.tooltip = document.createElement('div');
        this.tooltip.classList.add('tooltip');
        this.tooltip.style.display = 'none';
        document.body.appendChild(this.tooltip);
    }
    
    /**
     * Show the tooltip with the given content at the specified position
     * @param {string} content - HTML content for the tooltip
     * @param {Event} event - Mouse event to position the tooltip
     */
    showTooltip(content, event) {
        this.tooltip.innerHTML = content;
        this.tooltip.style.display = 'block';
        
        // Position the tooltip near the mouse pointer
        const x = event.pageX + 10;
        const y = event.pageY + 10;
        
        // Adjust position to keep tooltip within viewport
        const tooltipRect = this.tooltip.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        let adjustedX = x;
        let adjustedY = y;
        
        if (x + tooltipRect.width > viewportWidth) {
            adjustedX = x - tooltipRect.width - 20;
        }
        
        if (y + tooltipRect.height > viewportHeight) {
            adjustedY = y - tooltipRect.height - 20;
        }
        
        this.tooltip.style.left = `${adjustedX}px`;
        this.tooltip.style.top = `${adjustedY}px`;
    }
    
    /**
     * Hide the tooltip
     */
    hideTooltip() {
        this.tooltip.style.display = 'none';
    }
    
    /**
     * Clear the visualization container
     */
    clear() {
        this.container.innerHTML = '';
        this.svg = null;
        this.selectedNode = null;
    }
    
    /**
     * Set the visualization data and render
     * @param {object} data - Visualization data
     * @param {string} type - Visualization type
     */
    setData(data, type = null) {
        this.currentData = data;
        
        if (type) {
            this.currentVizType = type;
        }
        
        this.render();
    }
    
    /**
     * Set the importance threshold for filtering nodes
     * @param {number} threshold - Threshold value (0.0 to 1.0)
     */
    setImportanceThreshold(threshold) {
        this.importanceThreshold = threshold;
        
        // Re-render if we have data
        if (this.currentData) {
            this.render();
        }
    }
    
    /**
     * Show detailed information about a node in the details panel
     * @param {object} node - Node data
     */
    showNodeDetails(node) {
        const detailsPanel = document.getElementById('node-details');
        const nodeContent = document.getElementById('node-content');
        
        if (!detailsPanel || !nodeContent) return;
        
        // Set content based on node data
        nodeContent.innerHTML = `
            <div class="space-y-3">
                <div>
                    <span class="text-xs text-gray-500">ID:</span>
                    <span class="text-sm font-mono">${node.id}</span>
                </div>
                <div>
                    <span class="text-xs text-gray-500">Text:</span>
                    <div class="text-sm p-2 bg-gray-100 rounded">${node.text || node.token || ''}</div>
                </div>
                <div class="flex space-x-4">
                    <div>
                        <span class="text-xs text-gray-500">Importance:</span>
                        <span class="text-sm font-semibold">${(node.importance || 0).toFixed(2)}</span>
                    </div>
                    <div>
                        <span class="text-xs text-gray-500">Attention:</span>
                        <span class="text-sm">${(node.attention_score || 0).toFixed(2)}</span>
                    </div>
                    <div>
                        <span class="text-xs text-gray-500">Position:</span>
                        <span class="text-sm">${node.position || 'N/A'}</span>
                    </div>
                </div>
                ${node.is_divergence_point ? 
                    `<div class="text-xs bg-red-100 text-red-700 p-1 rounded">Divergence Point</div>` : ''}
                ${node.entity_links ? 
                    `<div>
                        <span class="text-xs text-gray-500">Knowledge Graph:</span>
                        <div class="text-sm">
                            ${this.formatEntityLinks(node.entity_links)}
                        </div>
                    </div>` : ''}
                ${node.counterfactuals ? 
                    `<div>
                        <span class="text-xs text-gray-500">Counterfactuals:</span>
                        <div class="text-sm">
                            ${this.formatCounterfactuals(node.counterfactuals)}
                        </div>
                    </div>` : ''}
            </div>
        `;
        
        // Display the panel
        detailsPanel.classList.remove('hidden');
        
        // Set the selected node and update visualization
        this.selectedNode = node;
        this.updateSelectedNodeInVisualization();
    }
    
    /**
     * Format entity links for display
     * @param {Array} entityLinks - Array of entity links
     * @returns {string} - Formatted HTML
     */
    formatEntityLinks(entityLinks) {
        if (!entityLinks || entityLinks.length === 0) {
            return '<span class="text-gray-500">No entity links available</span>';
        }
        
        return entityLinks.map(entity => `
            <div class="bg-indigo-50 p-1 rounded mb-1">
                <a href="${entity.url}" target="_blank" class="text-indigo-600 hover:underline">
                    ${entity.label} (${entity.similarity.toFixed(2)})
                </a>
            </div>
        `).join('');
    }
    
    /**
     * Format counterfactuals for display
     * @param {Array} counterfactuals - Array of counterfactuals
     * @returns {string} - Formatted HTML
     */
    formatCounterfactuals(counterfactuals) {
        if (!counterfactuals || counterfactuals.length === 0) {
            return '<span class="text-gray-500">No counterfactuals available</span>';
        }
        
        return counterfactuals.map(cf => `
            <div class="bg-orange-50 p-1 rounded mb-1 flex justify-between">
                <span class="font-mono">'${cf.original}' → '${cf.alternative}'</span>
                <span class="text-orange-700">${cf.impact.toFixed(2)}</span>
            </div>
        `).join('');
    }
    
    /**
     * Update the visualization to highlight the selected node
     */
    updateSelectedNodeInVisualization() {
        if (!this.svg || !this.selectedNode) return;
        
        // Remove previous selection
        this.svg.selectAll('.reasoning-node').classed('selected', false);
        this.svg.selectAll('.reasoning-edge').classed('highlighted', false);
        
        // Highlight the selected node
        this.svg.select(`.reasoning-node[data-id="${this.selectedNode.id}"]`).classed('selected', true);
        
        // Highlight edges connected to the selected node
        this.svg.selectAll('.reasoning-edge')
            .filter(d => d.source.id === this.selectedNode.id || d.target.id === this.selectedNode.id)
            .classed('highlighted', true);
    }
    
    /**
     * Close the node details panel
     */
    closeNodeDetails() {
        const detailsPanel = document.getElementById('node-details');
        
        if (detailsPanel) {
            detailsPanel.classList.add('hidden');
        }
        
        // Clear selection
        this.selectedNode = null;
        if (this.svg) {
            this.svg.selectAll('.reasoning-node').classed('selected', false);
            this.svg.selectAll('.reasoning-edge').classed('highlighted', false);
        }
    }
    
    /**
     * Render the current visualization based on type and data
     */
    render() {
        if (!this.currentData) {
            this.container.innerHTML = '<p class="text-gray-500 text-center">No data available for visualization</p>';
            return;
        }
        
        this.clear();
        
        // Create SVG container
        const containerRect = this.container.getBoundingClientRect();
        const width = containerRect.width;
        const height = Math.max(400, containerRect.height - 20);  // Ensure minimum height
        
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('class', 'viz-svg')
            .append('g')
            .attr('transform', 'translate(5,5)');  // Small margin
        
        // Render based on visualization type
        switch (this.currentVizType) {
            case 'reasoning-tree':
                this.renderReasoningTree(width, height);
                break;
            case 'token-importance':
                this.renderTokenImportance(width, height);
                break;
            case 'counterfactual':
                this.renderCounterfactual(width, height);
                break;
            case 'knowledge-graph':
                this.renderKnowledgeGraph(width, height);
                break;
            case 'divergence-points':
                this.renderDivergencePoints(width, height);
                break;
            default:
                this.container.innerHTML = `<p class="text-gray-500 text-center">Unsupported visualization type: ${this.currentVizType}</p>`;
        }
    }
    
    /**
     * Render a reasoning tree visualization
     * @param {number} width - Container width
     * @param {number} height - Container height
     */
    renderReasoningTree(width, height) {
        const data = this.currentData;
        
        if (!data || !data.nodes || !data.edges || data.nodes.length === 0) {
            this.container.innerHTML = '<p class="text-gray-500 text-center">No reasoning tree data available</p>';
            return;
        }
        
        // Filter nodes based on importance threshold
        const filteredNodes = data.nodes.filter(node => 
            (node.importance || 0) >= this.importanceThreshold
        );
        
        // Filter edges to include only connections between filtered nodes
        const filteredNodeIds = new Set(filteredNodes.map(node => node.id));
        const filteredEdges = data.edges.filter(edge => 
            filteredNodeIds.has(edge.from) && filteredNodeIds.has(edge.to)
        );
        
        // Set up force directed layout
        const simulation = d3.forceSimulation(filteredNodes)
            .force('link', d3.forceLink(filteredEdges)
                .id(d => d.id)
                .distance(100)
            )
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(50));
        
        // Create links
        const linkElements = this.svg.append('g')
            .selectAll('line')
            .data(filteredEdges)
            .enter()
            .append('line')
            .attr('class', 'reasoning-edge')
            .attr('stroke-width', d => 1 + (d.weight || 0.5) * 3)
            .attr('marker-end', 'url(#arrow)');
        
        // Add arrow marker for directed edges
        this.svg.append('defs').append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#d1d5db');
        
        // Create nodes
        const nodeElements = this.svg.append('g')
            .selectAll('circle')
            .data(filteredNodes)
            .enter()
            .append('g')
            .attr('class', 'reasoning-node')
            .attr('data-id', d => d.id)
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended)
            );
        
        // Add circles for nodes with size and color based on importance
        nodeElements.append('circle')
            .attr('r', d => 5 + (d.importance || 0.5) * 20)
            .attr('class', d => {
                const importance = d.importance || 0;
                if (importance >= 0.7) return 'node-importance-high';
                if (importance >= 0.4) return 'node-importance-medium';
                return 'node-importance-low';
            })
            .classed('divergence-point', d => d.is_divergence_point);
        
        // Add text labels
        nodeElements.append('text')
            .attr('dx', 12)
            .attr('dy', 4)
            .text(d => {
                const text = d.text || d.token || '';
                return text.length > 15 ? text.substring(0, 12) + '...' : text;
            })
            .attr('font-size', '12px')
            .attr('fill', '#4b5563');
        
        // Add events for node interaction
        nodeElements
            .on('mouseover', (event, d) => {
                const text = d.text || d.token || '';
                const importance = (d.importance || 0).toFixed(2);
                const attention = (d.attention_score || 0).toFixed(2);
                const isDivergence = d.is_divergence_point ? 
                    '<span class="text-red-600 text-xs">Divergence Point</span>' : '';
                
                const tooltipContent = `
                    <div>
                        <div class="font-bold">${text}</div>
                        <div class="text-xs">Importance: ${importance}</div>
                        <div class="text-xs">Attention: ${attention}</div>
                        ${isDivergence}
                    </div>
                `;
                
                this.showTooltip(tooltipContent, event);
            })
            .on('mousemove', (event) => {
                // Update tooltip position on mouse move
                const tooltipRect = this.tooltip.getBoundingClientRect();
                this.tooltip.style.left = `${event.pageX + 10}px`;
                this.tooltip.style.top = `${event.pageY - tooltipRect.height / 2}px`;
            })
            .on('mouseout', () => {
                this.hideTooltip();
            })
            .on('click', (event, d) => {
                this.showNodeDetails(d);
                event.stopPropagation();
            });
        
        // Add click handler to close node details when clicking outside
        this.svg.on('click', () => {
            this.closeNodeDetails();
        });
        
        // Update positions on each simulation tick
        simulation.on('tick', () => {
            // Constrain nodes to visualization area
            filteredNodes.forEach(node => {
                node.x = Math.max(50, Math.min(width - 50, node.x));
                node.y = Math.max(50, Math.min(height - 50, node.y));
            });
            
            linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            nodeElements
                .attr('transform', d => `translate(${d.x}, ${d.y})`);
        });
        
        // Drag functions
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    }
    
    /**
     * Render a token importance visualization
     * @param {number} width - Container width
     * @param {number} height - Container height
     */
    renderTokenImportance(width, height) {
        const data = this.currentData;
        
        if (!data || !data.tokens || data.tokens.length === 0) {
            this.container.innerHTML = '<p class="text-gray-500 text-center">No token importance data available</p>';
            return;
        }
        
        const tokens = data.tokens;
        const importanceScores = tokens.map(t => t.importance || 0);
        const attentionScores = tokens.map(t => t.attention_score || 0);
        
        // Set up scales
        const xScale = d3.scaleBand()
            .domain(tokens.map((_, i) => i))
            .range([50, width - 50])
            .padding(0.1);
        
        const yScale = d3.scaleLinear()
            .domain([0, 1])
            .range([height - 50, 50]);
        
        // Add y-axis
        this.svg.append('g')
            .attr('transform', `translate(50, 0)`)
            .call(d3.axisLeft(yScale).ticks(5));
        
        // Add x-axis
        this.svg.append('g')
            .attr('transform', `translate(0, ${height - 50})`)
            .call(d3.axisBottom(xScale).tickFormat(i => {
                const token = tokens[i]?.text || tokens[i]?.token || '';
                return token.length > 8 ? token.substring(0, 5) + '...' : token;
            }))
            .selectAll('text')
            .attr('transform', 'rotate(-45)')
            .style('text-anchor', 'end');
        
        // Y-axis label
        this.svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', 15)
            .attr('x', -height / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#6b7280')
            .attr('font-size', '12px')
            .text('Score');
        
        // X-axis label
        this.svg.append('text')
            .attr('y', height - 10)
            .attr('x', width / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#6b7280')
            .attr('font-size', '12px')
            .text('Tokens');
        
        // Draw importance bars
        this.svg.selectAll('.token-bar')
            .data(tokens)
            .enter()
            .append('rect')
            .attr('class', 'token-bar')
            .attr('x', (d, i) => xScale(i))
            .attr('y', d => yScale(d.importance || 0))
            .attr('width', xScale.bandwidth())
            .attr('height', d => height - 50 - yScale(d.importance || 0))
            .attr('fill', d => {
                const importance = d.importance || 0;
                // Use color scale from indigo-100 to indigo-900
                return d3.interpolate('#e0e7ff', '#312e81')(importance);
            })
            .attr('stroke', '#c7d2fe')
            .attr('stroke-width', 1)
            .on('mouseover', (event, d) => {
                const token = d.text || d.token || '[Empty]';
                const importance = (d.importance || 0).toFixed(2);
                const attention = (d.attention_score || 0).toFixed(2);
                
                const tooltipContent = `
                    <div>
                        <div class="font-bold">${token}</div>
                        <div class="text-xs">Importance: ${importance}</div>
                        <div class="text-xs">Attention: ${attention}</div>
                    </div>
                `;
                
                this.showTooltip(tooltipContent, event);
            })
            .on('mousemove', (event) => {
                const tooltipRect = this.tooltip.getBoundingClientRect();
                this.tooltip.style.left = `${event.pageX + 10}px`;
                this.tooltip.style.top = `${event.pageY - tooltipRect.height / 2}px`;
            })
            .on('mouseout', () => {
                this.hideTooltip();
            })
            .on('click', (event, d) => {
                this.showNodeDetails(d);
                event.stopPropagation();
            });
        
        // Add attention score line
        if (attentionScores.some(score => score > 0)) {
            const line = d3.line()
                .x((d, i) => xScale(i) + xScale.bandwidth() / 2)
                .y(d => yScale(d.attention_score || 0))
                .curve(d3.curveMonotoneX);
            
            this.svg.append('path')
                .datum(tokens)
                .attr('fill', 'none')
                .attr('stroke', '#ef4444')
                .attr('stroke-width', 2)
                .attr('d', line);
            
            // Add dots for attention scores
            this.svg.selectAll('.attention-dot')
                .data(tokens)
                .enter()
                .append('circle')
                .attr('class', 'attention-dot')
                .attr('cx', (d, i) => xScale(i) + xScale.bandwidth() / 2)
                .attr('cy', d => yScale(d.attention_score || 0))
                .attr('r', 4)
                .attr('fill', '#ef4444');
        }
        
        // Add legend
        const legend = this.svg.append('g')
            .attr('transform', `translate(${width - 150}, 20)`);
        
        legend.append('rect')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', 15)
            .attr('height', 15)
            .attr('fill', '#4f46e5');
        
        legend.append('text')
            .attr('x', 20)
            .attr('y', 12)
            .attr('font-size', '12px')
            .attr('fill', '#6b7280')
            .text('Importance');
        
        legend.append('line')
            .attr('x1', 0)
            .attr('y1', 30)
            .attr('x2', 15)
            .attr('y2', 30)
            .attr('stroke', '#ef4444')
            .attr('stroke-width', 2);
        
        legend.append('circle')
            .attr('cx', 7.5)
            .attr('cy', 30)
            .attr('r', 4)
            .attr('fill', '#ef4444');
        
        legend.append('text')
            .attr('x', 20)
            .attr('y', 34)
            .attr('font-size', '12px')
            .attr('fill', '#6b7280')
            .text('Attention');
    }
    
    /**
     * Render a counterfactual visualization
     * @param {number} width - Container width
     * @param {number} height - Container height
     */
    renderCounterfactual(width, height) {
        const data = this.currentData;
        
        if (!data || !data.counterfactuals || data.counterfactuals.length === 0) {
            this.container.innerHTML = '<p class="text-gray-500 text-center">No counterfactual data available</p>';
            return;
        }
        
        // Sort counterfactuals by impact score
        const counterfactuals = [...data.counterfactuals]
            .sort((a, b) => (b.impact || 0) - (a.impact || 0))
            .slice(0, 10); // Show top 10 for readability
        
        // Add title
        this.svg.append('text')
            .attr('x', width / 2)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .attr('font-size', '16px')
            .attr('font-weight', 'bold')
            .attr('fill', '#1f2937')
            .text('Top Counterfactual Substitutions');
        
        // Set up scales
        const yScale = d3.scaleBand()
            .domain(counterfactuals.map((_, i) => i))
            .range([50, height - 50])
            .padding(0.2);
        
        const xScale = d3.scaleLinear()
            .domain([0, d3.max(counterfactuals, d => d.impact || 0) * 1.1])
            .range([150, width - 50]);
        
        // Add y-axis (labels only)
        this.svg.selectAll('.cf-label')
            .data(counterfactuals)
            .enter()
            .append('text')
            .attr('class', 'cf-label')
            .attr('x', 145)
            .attr('y', (d, i) => yScale(i) + yScale.bandwidth() / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .attr('font-size', '12px')
            .attr('fill', '#4b5563')
            .text((d, i) => {
                const original = d.original || '';
                return original.length > 12 ? original.substring(0, 9) + '...' : original;
            });
        
        // Add x-axis
        this.svg.append('g')
            .attr('transform', `translate(0, ${height - 50})`)
            .call(d3.axisBottom(xScale).ticks(5));
        
        // X-axis label
        this.svg.append('text')
            .attr('y', height - 10)
            .attr('x', width / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#6b7280')
            .attr('font-size', '12px')
            .text('Impact Score');
        
        // Add bars
        this.svg.selectAll('.counterfactual-bar')
            .data(counterfactuals)
            .enter()
            .append('rect')
            .attr('class', 'counterfactual-bar')
            .attr('x', 150)
            .attr('y', (d, i) => yScale(i))
            .attr('width', d => Math.max(1, xScale(d.impact || 0) - 150))
            .attr('height', yScale.bandwidth())
            .attr('fill', d => d.flipped_output ? '#f97316' : '#4f46e5')
            .attr('opacity', 0.8)
            .on('mouseover', (event, d) => {
                const original = d.original || '[Empty]';
                const alternative = d.alternative || '[Empty]';
                const impact = (d.impact || 0).toFixed(2);
                const flipped = d.flipped_output ? 'Yes' : 'No';
                
                const tooltipContent = `
                    <div>
                        <div class="font-bold">Original: '${original}'</div>
                        <div class="font-bold">Alternative: '${alternative}'</div>
                        <div class="text-xs">Impact: ${impact}</div>
                        <div class="text-xs">Output flipped: ${flipped}</div>
                    </div>
                `;
                
                this.showTooltip(tooltipContent, event);
            })
            .on('mousemove', (event) => {
                const tooltipRect = this.tooltip.getBoundingClientRect();
                this.tooltip.style.left = `${event.pageX + 10}px`;
                this.tooltip.style.top = `${event.pageY - tooltipRect.height / 2}px`;
            })
            .on('mouseout', () => {
                this.hideTooltip();
            })
            .on('click', (event, d) => {
                this.showNodeDetails(d);
                event.stopPropagation();
            });
        
        // Add text on bars
        this.svg.selectAll('.counterfactual-text')
            .data(counterfactuals)
            .enter()
            .append('text')
            .attr('class', 'counterfactual-label')
            .attr('x', d => Math.max(155, xScale(d.impact || 0) - 100))
            .attr('y', (d, i) => yScale(i) + yScale.bandwidth() / 2)
            .attr('fill', 'white')
            .attr('font-size', '12px')
            .attr('pointer-events', 'none')
            .text(d => {
                const alt = d.alternative || '';
                return alt.length > 10 ? alt.substring(0, 7) + '...' : alt;
            });
        
        // Add legend
        const legend = this.svg.append('g')
            .attr('transform', `translate(${width - 200}, 20)`);
        
        legend.append('rect')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', 15)
            .attr('height', 15)
            .attr('fill', '#4f46e5')
            .attr('opacity', 0.8);
        
        legend.append('text')
            .attr('x', 20)
            .attr('y', 12)
            .attr('font-size', '12px')
            .attr('fill', '#6b7280')
            .text('Same Output');
        
        legend.append('rect')
            .attr('x', 0)
            .attr('y', 25)
            .attr('width', 15)
            .attr('height', 15)
            .attr('fill', '#f97316')
            .attr('opacity', 0.8);
        
        legend.append('text')
            .attr('x', 20)
            .attr('y', 37)
            .attr('font-size', '12px')
            .attr('fill', '#6b7280')
            .text('Flipped Output');
    }
    
    /**
     * Render a knowledge graph visualization
     * @param {number} width - Container width
     * @param {number} height - Container height
     */
    renderKnowledgeGraph(width, height) {
        const data = this.currentData;
        
        if (!data || !data.entities || data.entities.length === 0) {
            this.container.innerHTML = '<p class="text-gray-500 text-center">No knowledge graph data available</p>';
            return;
        }
        
        // Title
        this.svg.append('text')
            .attr('x', width / 2)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .attr('font-size', '16px')
            .attr('font-weight', 'bold')
            .attr('fill', '#1f2937')
            .text('Knowledge Graph Validation');
        
        // Filter entities based on importance threshold (if available)
        const entities = data.entities.filter(entity => 
            (entity.confidence || 0) >= this.importanceThreshold
        );
        
        // Create links between entities if they exist
        const links = data.links || [];
        
        // Set up force directed layout
        const simulation = d3.forceSimulation(entities)
            .force('link', d3.forceLink(links)
                .id(d => d.id)
                .distance(100)
            )
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(40));
        
        // Create links
        const linkElements = this.svg.append('g')
            .selectAll('line')
            .data(links)
            .enter()
            .append('line')
            .attr('stroke', '#d1d5db')
            .attr('stroke-width', d => Math.max(1, d.weight || 1) * 2)
            .attr('stroke-opacity', 0.6);
        
        // Create nodes
        const nodeElements = this.svg.append('g')
            .selectAll('g')
            .data(entities)
            .enter()
            .append('g')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended)
            );
        
        // Add circles for nodes
        nodeElements.append('circle')
            .attr('r', d => 15 + (d.confidence || 0.5) * 15)
            .attr('fill', d => {
                if (d.type === 'entity') return '#4f46e5';
                if (d.type === 'statement') return '#10b981';
                return '#f59e0b';
            })
            .attr('fill-opacity', 0.7)
            .attr('stroke', '#e5e7eb')
            .attr('stroke-width', 2);
        
        // Add text labels
        nodeElements.append('text')
            .attr('dy', 5)
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px')
            .attr('fill', 'white')
            .attr('pointer-events', 'none')
            .text(d => {
                const label = d.label || '';
                return label.length > 15 ? label.substring(0, 12) + '...' : label;
            });
        
        // Add events for node interaction
        nodeElements
            .on('mouseover', (event, d) => {
                const label = d.label || '';
                const type = d.type || 'Unknown';
                const confidence = (d.confidence || 0).toFixed(2);
                const source = d.source || 'Unknown';
                
                const tooltipContent = `
                    <div>
                        <div class="font-bold">${label}</div>
                        <div class="text-xs">Type: ${type}</div>
                        <div class="text-xs">Confidence: ${confidence}</div>
                        <div class="text-xs">Source: ${source}</div>
                    </div>
                `;
                
                this.showTooltip(tooltipContent, event);
            })
            .on('mousemove', (event) => {
                const tooltipRect = this.tooltip.getBoundingClientRect();
                this.tooltip.style.left = `${event.pageX + 10}px`;
                this.tooltip.style.top = `${event.pageY - tooltipRect.height / 2}px`;
            })
            .on('mouseout', () => {
                this.hideTooltip();
            })
            .on('click', (event, d) => {
                this.showNodeDetails(d);
                event.stopPropagation();
            });
        
        // Add legend
        const legend = this.svg.append('g')
            .attr('transform', `translate(20, 40)`);
        
        // Entity
        legend.append('circle')
            .attr('cx', 7.5)
            .attr('cy', 7.5)
            .attr('r', 7.5)
            .attr('fill', '#4f46e5')
            .attr('fill-opacity', 0.7);
        
        legend.append('text')
            .attr('x', 20)
            .attr('y', 12)
            .attr('font-size', '12px')
            .attr('fill', '#6b7280')
            .text('Entity');
        
        // Statement
        legend.append('circle')
            .attr('cx', 7.5)
            .attr('cy', 32.5)
            .attr('r', 7.5)
            .attr('fill', '#10b981')
            .attr('fill-opacity', 0.7);
        
        legend.append('text')
            .attr('x', 20)
            .attr('y', 37)
            .attr('font-size', '12px')
            .attr('fill', '#6b7280')
            .text('Statement');
        
        // Other
        legend.append('circle')
            .attr('cx', 7.5)
            .attr('cy', 57.5)
            .attr('r', 7.5)
            .attr('fill', '#f59e0b')
            .attr('fill-opacity', 0.7);
        
        legend.append('text')
            .attr('x', 20)
            .attr('y', 62)
            .attr('font-size', '12px')
            .attr('fill', '#6b7280')
            .text('Other');
        
        // Update positions on each simulation tick
        simulation.on('tick', () => {
            // Constrain nodes to visualization area
            entities.forEach(entity => {
                entity.x = Math.max(50, Math.min(width - 50, entity.x));
                entity.y = Math.max(50, Math.min(height - 50, entity.y));
            });
            
            linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            nodeElements
                .attr('transform', d => `translate(${d.x}, ${d.y})`);
        });
        
        // Drag functions
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    }
    
    /**
     * Render a divergence points visualization
     * @param {number} width - Container width
     * @param {number} height - Container height
     */
    renderDivergencePoints(width, height) {
        const data = this.currentData;
        
        if (!data || !data.tokens || !data.divergence_points || data.divergence_points.length === 0) {
            this.container.innerHTML = '<p class="text-gray-500 text-center">No divergence points data available</p>';
            return;
        }
        
        // Title
        this.svg.append('text')
            .attr('x', width / 2)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .attr('font-size', '16px')
            .attr('font-weight', 'bold')
            .attr('fill', '#1f2937')
            .text('Divergence Points Analysis');
        
        const tokens = data.tokens;
        const divergencePoints = data.divergence_points;
        
        // Extract token positions for divergence points
        const divergencePositions = new Set(divergencePoints.map(dp => dp.position));
        
        // Set up horizontal timeline
        const xScale = d3.scaleLinear()
            .domain([0, tokens.length - 1])
            .range([50, width - 50]);
        
        // Token circles
        const tokenCircles = this.svg.selectAll('.token-circle')
            .data(tokens)
            .enter()
            .append('circle')
            .attr('class', 'token-circle')
            .attr('cx', (d, i) => xScale(i))
            .attr('cy', height / 2)
            .attr('r', (d, i) => divergencePositions.has(i) ? 12 : 8)
            .attr('fill', (d, i) => divergencePositions.has(i) ? '#ef4444' : '#6366f1')
            .attr('stroke', '#e5e7eb')
            .attr('stroke-width', 2)
            .on('mouseover', (event, d, i) => {
                const index = tokens.indexOf(d);
                const token = d.text || d.token || '';
                const isDivergence = divergencePositions.has(index);
                
                let tooltipContent = `
                    <div>
                        <div class="font-bold">${token}</div>
                        <div class="text-xs">Position: ${index}</div>
                `;
                
                if (isDivergence) {
                    const dp = divergencePoints.find(dp => dp.position === index);
                    tooltipContent += `
                        <div class="text-xs text-red-600">Divergence Point</div>
                        <div class="text-xs">Severity: ${(dp.severity || 0).toFixed(2)}</div>
                        <div class="text-xs">Path IDs: ${dp.path_ids?.join(', ') || 'N/A'}</div>
                    `;
                }
                
                tooltipContent += '</div>';
                
                this.showTooltip(tooltipContent, event);
            })
            .on('mousemove', (event) => {
                const tooltipRect = this.tooltip.getBoundingClientRect();
                this.tooltip.style.left = `${event.pageX + 10}px`;
                this.tooltip.style.top = `${event.pageY - tooltipRect.height / 2}px`;
            })
            .on('mouseout', () => {
                this.hideTooltip();
            })
            .on('click', (event, d) => {
                const index = tokens.indexOf(d);
                if (divergencePositions.has(index)) {
                    const dp = divergencePoints.find(dp => dp.position === index);
                    this.showNodeDetails(dp);
                } else {
                    this.showNodeDetails(d);
                }
                event.stopPropagation();
            });
        
        // Token labels
        this.svg.selectAll('.token-label')
            .data(tokens)
            .enter()
            .append('text')
            .attr('class', 'token-label')
            .attr('x', (d, i) => xScale(i))
            .attr('y', height / 2 + 30)
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px')
            .attr('fill', '#4b5563')
            .text((d, i) => {
                const token = d.text || d.token || '';
                return token.length > 8 ? token.substring(0, 5) + '...' : token;
            });
        
        // Connecting line
        this.svg.append('line')
            .attr('x1', xScale(0))
            .attr('y1', height / 2)
            .attr('x2', xScale(tokens.length - 1))
            .attr('y2', height / 2)
            .attr('stroke', '#d1d5db')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '5,5');
        
        // Path comparison
        if (data.paths && data.paths.length >= 2) {
            const pathColors = ['#4f46e5', '#ef4444', '#f59e0b', '#10b981'];
            const pathHeight = Math.min(80, (height - 100) / data.paths.length);
            
            // Draw paths
            for (let i = 0; i < Math.min(data.paths.length, 4); i++) {
                const pathY = 80 + i * pathHeight;
                const path = data.paths[i];
                
                // Path label
                this.svg.append('text')
                    .attr('x', 20)
                    .attr('y', pathY)
                    .attr('text-anchor', 'start')
                    .attr('dominant-baseline', 'middle')
                    .attr('font-size', '12px')
                    .attr('fill', pathColors[i % pathColors.length])
                    .text(`Path ${i + 1}`);
                
                // Path line
                this.svg.append('line')
                    .attr('x1', 50)
                    .attr('y1', pathY)
                    .attr('x2', width - 50)
                    .attr('y2', pathY)
                    .attr('stroke', pathColors[i % pathColors.length])
                    .attr('stroke-width', 2)
                    .attr('stroke-opacity', 0.7);
                
                // Path divergence markers
                const pathDivergences = divergencePoints.filter(dp => 
                    dp.path_ids?.includes(i)
                );
                
                this.svg.selectAll(`.path-${i}-markers`)
                    .data(pathDivergences)
                    .enter()
                    .append('circle')
                    .attr('cx', d => xScale(d.position))
                    .attr('cy', pathY)
                    .attr('r', 6)
                    .attr('fill', pathColors[i % pathColors.length])
                    .attr('stroke', '#ffffff')
                    .attr('stroke-width', 2);
            }
        }
        
        // Add legend
        const legend = this.svg.append('g')
            .attr('transform', `translate(${width - 170}, 40)`);
        
        legend.append('circle')
            .attr('cx', 7.5)
            .attr('cy', 7.5)
            .attr('r', 7.5)
            .attr('fill', '#6366f1');
        
        legend.append('text')
            .attr('x', 20)
            .attr('y', 12)
            .attr('font-size', '12px')
            .attr('fill', '#6b7280')
            .text('Regular Token');
        
        legend.append('circle')
            .attr('cx', 7.5)
            .attr('cy', 32.5)
            .attr('r', 7.5)
            .attr('fill', '#ef4444');
        
        legend.append('text')
            .attr('x', 20)
            .attr('y', 37)
            .attr('font-size', '12px')
            .attr('fill', '#6b7280')
            .text('Divergence Point');
    }
    
    /**
     * Export the current visualization as an image
     * @returns {string} - Data URL of the exported image
     */
    exportVisualization() {
        if (!this.svg) {
            console.error('No visualization to export');
            return null;
        }
        
        try {
            // Create a copy of the SVG element
            const svgElement = this.container.querySelector('svg');
            const svgCopy = svgElement.cloneNode(true);
            
            // Set background for the SVG
            svgCopy.style.backgroundColor = 'white';
            
            // Create a canvas element
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            
            // Set canvas dimensions to match the SVG
            canvas.width = svgElement.width.baseVal.value;
            canvas.height = svgElement.height.baseVal.value;
            
            // Fill canvas with white background
            context.fillStyle = 'white';
            context.fillRect(0, 0, canvas.width, canvas.height);
            
            // Create image from SVG
            const svgData = new XMLSerializer().serializeToString(svgCopy);
            const img = new Image();
            
            // Create a data URL from the SVG
            const svgBlob = new Blob([svgData], {type: 'image/svg+xml;charset=utf-8'});
            const url = URL.createObjectURL(svgBlob);
            
            // Return a promise that resolves when the image is loaded
            return new Promise((resolve, reject) => {
                img.onload = () => {
                    // Draw the image on the canvas
                    context.drawImage(img, 0, 0);
                    
                    // Get data URL from canvas
                    const dataURL = canvas.toDataURL('image/png');
                    
                    // Clean up
                    URL.revokeObjectURL(url);
                    
                    resolve(dataURL);
                };
                
                img.onerror = () => {
                    console.error('Error loading SVG image');
                    URL.revokeObjectURL(url);
                    reject(new Error('Failed to export visualization'));
                };
                
                img.src = url;
            });
        } catch (error) {
            console.error('Error exporting visualization:', error);
            return null;
        }
    }
}