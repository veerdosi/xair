// Core types for visualization
interface TreeVisualizerProps {
  tree: CGRTTree;
  width: number;
  height: number;
  onNodeClick?: (nodeId: string) => void;
  onEdgeClick?: (sourceId: string, targetId: string) => void;
  highlightedPath?: string[];
  selectedNode?: string;
  config?: VisualizationConfig;
}

interface VisualizationConfig {
  nodeSizeScale: number;
  edgeWidthScale: number;
  colors: {
    mainPath: string;
    alternative: string;
    merge: string;
    highlight: string;
    attention: string;
    crossLink: string;
  };
  animation: {
    duration: number;
    easing: string;
  };
  layout: {
    type: 'hierarchical' | 'force' | 'radial';
    options: any;
  };
}

// Main visualizer component
const TreeVisualizer: React.FC<TreeVisualizerProps> = ({
  tree,
  width,
  height,
  onNodeClick,
  onEdgeClick,
  highlightedPath,
  selectedNode,
  config = defaultConfig
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const dependenTreeRef = useRef<any>(null);
  
  // State for visualization
  const [layout, setLayout] = useState<any>(null);
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<[string, string] | null>(null);
  
  // Memoized data transformation
  const visualData = useMemo(() => transformTreeData(tree), [tree]);
  
  // Layout calculation
  useEffect(() => {
    if (!visualData || !width || !height) return;
    
    const newLayout = calculateLayout(
      visualData,
      width,
      height,
      config.layout
    );
    setLayout(newLayout);
  }, [visualData, width, height, config.layout]);
  
  // DependenTree initialization
  useEffect(() => {
    if (!svgRef.current || !layout) return;
    
    const tree = new DependenTree(svgRef.current, {
      data: layout,
      nodeRadius: config.nodeSizeScale,
      linkWidth: config.edgeWidthScale,
      colors: config.colors,
      duration: config.animation.duration,
      easing: config.animation.easing,
      allowCycles: true,
      onNodeClick: handleNodeClick,
      onLinkClick: handleEdgeClick,
      renderNode: customNodeRenderer,
      renderLink: customLinkRenderer
    });
    
    dependenTreeRef.current = tree;
    
    return () => tree.destroy();
  }, [layout]);
  
  // Custom node renderer
  const customNodeRenderer = useCallback((node: any) => {
    return (
      <NodeContainer
        node={node}
        config={config}
        isHighlighted={highlightedPath?.includes(node.id)}
        isSelected={selectedNode === node.id}
        isHovered={hoveredNode === node.id}
      >
        <NodeContent node={node} />
        {node.mergePoint && <MergeIndicator />}
        <ImportanceRing score={node.importance} />
        <AttentionIndicator pattern={node.attention} />
      </NodeContainer>
    );
  }, [highlightedPath, selectedNode, hoveredNode, config]);
  
  // Custom link renderer
  const customLinkRenderer = useCallback((link: any) => {
    return (
      <LinkContainer
        link={link}
        config={config}
        isHighlighted={isLinkHighlighted(link, highlightedPath)}
        isHovered={isLinkHovered(link, hoveredEdge)}
      >
        <LinkPath link={link} />
        {link.type === 'merge' && <MergeDecorator />}
        {link.type === 'cross' && <CrossLinkDecorator />}
        <LinkStrength value={link.strength} />
      </LinkContainer>
    );
  }, [highlightedPath, hoveredEdge, config]);

  return (
    <div ref={containerRef} className="relative w-full h-full">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full h-full"
      >
        <defs>
          <marker
            id="arrowhead"
            viewBox="0 -5 10 10"
            refX={8}
            refY={0}
            markerWidth={6}
            markerHeight={6}
            orient="auto"
          >
            <path d="M0,-5L10,0L0,5" className="fill-current" />
          </marker>
        </defs>
        <g className="links" />
        <g className="nodes" />
      </svg>
      <Controls
        onZoomIn={() => handleZoom('in')}
        onZoomOut={() => handleZoom('out')}
        onReset={() => handleZoom('reset')}
      />
      <Tooltip node={hoveredNode} edge={hoveredEdge} />
    </div>
  );
};