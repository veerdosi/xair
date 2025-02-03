// Interaction Handlers
interface InteractionState {
  isDragging: boolean;
  isZooming: boolean;
  lastPosition: { x: number; y: number };
  zoomLevel: number;
  transform: d3.ZoomTransform;
  mode: 'explore' | 'precise';
}

class InteractionManager {
  private state: InteractionState;
  private container: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private config: VisualizationConfig;
  private handlers: Map<string, Function>;
  
  constructor(
    container: SVGSVGElement,
    config: VisualizationConfig
  ) {
    this.container = d3.select(container);
    this.config = config;
    this.state = this.initializeState();
    this.handlers = new Map();
    
    this.setupInteractions();
  }
  
  private initializeState(): InteractionState {
    return {
      isDragging: false,
      isZooming: false,
      lastPosition: { x: 0, y: 0 },
      zoomLevel: 1,
      transform: d3.zoomIdentity,
      mode: 'explore'
    };
  }
  
  private setupInteractions() {
    // Set up zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 3])
      .on('zoom', (event) => this.handleZoom(event))
      .filter(event => {
        // Disable zoom during drag in precise mode
        if (this.state.mode === 'precise' && this.state.isDragging) {
          return false;
        }
        return true;
      });
      
    this.container.call(zoom);
    
    // Set up drag behavior
    const drag = d3.drag<SVGSVGElement, unknown>()
      .on('start', (event) => this.handleDragStart(event))
      .on('drag', (event) => this.handleDrag(event))
      .on('end', (event) => this.handleDragEnd(event));
      
    this.container.call(drag);
    
    // Set up other event listeners
    this.container
      .on('click', (event) => this.handleClick(event))
      .on('dblclick', (event) => this.handleDoubleClick(event))
      .on('mousemove', (event) => this.handleMouseMove(event))
      .on('mouseleave', (event) => this.handleMouseLeave(event));
  }
  
  private handleZoom(event: d3.D3ZoomEvent<SVGSVGElement, unknown>) {
    this.state.isZooming = true;
    this.state.zoomLevel = event.transform.k;
    this.state.transform = event.transform;
    
    // Apply zoom transform
    this.container.select('.zoom-group')
      .attr('transform', event.transform.toString());
      
    // Update node sizes based on zoom level
    this.container.selectAll('.node-container')
      .attr('transform', d => {
        const scale = this.calculateNodeScale(d, event.transform.k);
        return `translate(${d.x},${d.y}) scale(${scale})`;
      });
      
    this.notifyHandlers('zoom', event);
    
    // Use RAF for smooth animation
    requestAnimationFrame(() => {
      this.state.isZooming = false;
    });
  }
  
  private calculateNodeScale(node: any, zoomLevel: number): number {
    // Scale nodes based on importance and zoom level
    const baseScale = 1 / zoomLevel;
    const importanceScale = 1 + (node.importance_score * 0.5);
    return baseScale * importanceScale;
  }
  
  private handleDragStart(event: d3.D3DragEvent<SVGSVGElement, unknown, unknown>) {
    this.state.isDragging = true;
    this.state.lastPosition = { x: event.x, y: event.y };
    
    if (this.state.mode === 'explore') {
      // Add inertia in explore mode
      this.container.classed('dragging-with-inertia', true);
    }
  }
  
  private handleDrag(event: d3.D3DragEvent<SVGSVGElement, unknown, unknown>) {
    if (!this.state.isDragging) return;
    
    const dx = event.x - this.state.lastPosition.x;
    const dy = event.y - this.state.lastPosition.y;
    
    // Apply drag transform
    const transform = this.state.transform.translate(dx, dy);
    this.container.select('.zoom-group')
      .attr('transform', transform.toString());
      
    this.state.lastPosition = { x: event.x, y: event.y };
    this.state.transform = transform;
    
    this.notifyHandlers('drag', event);
  }
  
  private handleDragEnd(event: d3.D3DragEvent<SVGSVGElement, unknown, unknown>) {
    this.state.isDragging = false;
    
    if (this.state.mode === 'explore') {
      // Apply inertia
      const velocity = {
        x: event.x - this.state.lastPosition.x,
        y: event.y - this.state.lastPosition.y
      };
      
      this.applyInertia(velocity);
    }
    
    this.container.classed('dragging-with-inertia', false);
  }
  
  private applyInertia(velocity: { x: number; y: number }) {
    const decay = 0.95;
    const minVelocity = 0.1;
    
    const animate = () => {
      if (Math.abs(velocity.x) < minVelocity && 
          Math.abs(velocity.y) < minVelocity) {
        return;
      }
      
      const transform = this.state.transform.translate(
        velocity.x,
        velocity.y
      );
      
      this.container.select('.zoom-group')
        .attr('transform', transform.toString());
        
      this.state.transform = transform;
      
      velocity.x *= decay;
      velocity.y *= decay;
      
      requestAnimationFrame(animate);
    };
    
    animate();
  }
}

// Layout Manager
class LayoutManager {
  private tree: CGRTTree;
  private config: VisualizationConfig;
  private dimensions: { width: number; height: number };
  private layoutCache: Map<string, any>;
  
  constructor(
    tree: CGRTTree,
    config: VisualizationConfig,
    dimensions: { width: number; height: number }
  ) {
    this.tree = tree;
    this.config = config;
    this.dimensions = dimensions;
    this.layoutCache = new Map();
  }
  
  public calculateLayout(type: 'hierarchical' | 'force' | 'radial') {
    switch (type) {
      case 'hierarchical':
        return this.calculateHierarchicalLayout();
      case 'force':
        return this.calculateForceLayout();
      case 'radial':
        return this.calculateRadialLayout();
      default:
        return this.calculateHierarchicalLayout();
    }
  }
  
  private calculateHierarchicalLayout() {
    const layout = d3.tree()
      .size([this.dimensions.width, this.dimensions.height])
      .nodeSize([50, 100])
      .separation((a, b) => {
        // Adjust separation based on node importance
        const baseDistance = a.parent === b.parent ? 1 : 2;
        const importanceFactor = Math.max(
          a.data.importance_score,
          b.data.importance_score
        );
        return baseDistance * (1 + importanceFactor);
      });
      
    const root = d3.hierarchy(this.tree);
    const nodes = layout(root);
    
    // Handle cross-links
    this.processCrossLinks(nodes);
    
    return {
      nodes: nodes.descendants(),
      links: nodes.links()
    };
  }
  
  private calculateForceLayout() {
    const simulation = d3.forceSimulation()
      .force('link', d3.forceLink().id(d => d.id))
      .force('charge', d3.forceManyBody().strength(-100))
      .force('center', d3.forceCenter(
        this.dimensions.width / 2,
        this.dimensions.height / 2
      ))
      .force('collision', d3.forceCollide().radius(50));
      
    const nodes = this.tree.nodes.map(node => ({
      ...node,
      x: undefined,
      y: undefined
    }));
    
    const links = this.tree.links.map(link => ({
      ...link,
      source: link.source.id,
      target: link.target.id
    }));
    
    simulation.nodes(nodes);
    simulation.force('link').links(links);
    
    // Run simulation
    for (let i = 0; i < 300; ++i) simulation.tick();
    
    return { nodes, links };
  }
  
  private calculateRadialLayout() {
    const layout = d3.tree()
      .size([2 * Math.PI, Math.min(
        this.dimensions.width,
        this.dimensions.height
      ) / 2])
      .separation((a, b) => {
        return (a.parent === b.parent ? 1 : 2) / a.depth;
      });
      
    const root = d3.hierarchy(this.tree);
    const nodes = layout(root);
    
    // Convert to Cartesian coordinates
    nodes.each(node => {
      const x = node.x;
      const y = node.y;
      
      node.x = y * Math.cos(x - Math.PI / 2) + 
        this.dimensions.width / 2;
      node.y = y * Math.sin(x - Math.PI / 2) + 
        this.dimensions.height / 2;
    });
    
    return {
      nodes: nodes.descendants(),
      links: nodes.links()
    };
  }
  
  private processCrossLinks(nodes: d3.HierarchyNode<any>) {
    const crossLinks = this.tree.links.filter(link => 
      link.type === 'cross'
    );
    
    for (const link of crossLinks) {
      const source = nodes.find(n => n.data.id === link.source);
      const target = nodes.find(n => n.data.id === link.target);
      
      if (source && target) {
        // Create curved path
        const midX = (source.x + target.x) / 2;
        const midY = (source.y + target.y) / 2;
        
        link.path = `M${source.x},${source.y}
          Q${midX},${midY} ${target.x},${target.y}`;
      }
    }
  }
}