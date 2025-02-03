// Node Components
interface NodeContainerProps {
  node: CGRTNode;
  config: VisualizationConfig;
  isHighlighted: boolean;
  isSelected: boolean;
  isHovered: boolean;
  children: React.ReactNode;
}

const NodeContainer: React.FC<NodeContainerProps> = ({
  node,
  config,
  isHighlighted,
  isSelected,
  isHovered,
  children
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [animationState, setAnimationState] = useState('entering');
  
  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    
    // Setup intersection observer for animation triggers
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            setAnimationState('visible');
          }
        });
      },
      { threshold: 0.1 }
    );
    
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  
  return (
    <motion.g
      ref={containerRef}
      initial={{ scale: 0, opacity: 0 }}
      animate={animationState === 'visible' ? {
        scale: 1,
        opacity: 1
      } : undefined}
      exit={{ scale: 0, opacity: 0 }}
      transition={{
        duration: config.animation.duration,
        ease: config.animation.easing
      }}
      className={clsx(
        'node-container',
        isHighlighted && 'highlighted',
        isSelected && 'selected',
        isHovered && 'hovered'
      )}
      role="button"
      aria-selected={isSelected}
      aria-label={`Node ${node.text} with importance ${node.importance_score}`}
    >
      {children}
    </motion.g>
  );
};

const NodeContent: React.FC<{ node: CGRTNode }> = ({ node }) => {
  const contentRef = useRef<SVGGElement>(null);
  
  useEffect(() => {
    if (!contentRef.current) return;
    
    // Setup content animations
    const element = contentRef.current;
    const textElements = element.querySelectorAll('text');
    
    textElements.forEach(text => {
      // Animate text appearance
      const length = text.getComputedTextLength();
      text.style.strokeDasharray = `${length} ${length}`;
      text.style.strokeDashoffset = `${length}`;
      
      text.animate(
        [
          { strokeDashoffset: length },
          { strokeDashoffset: 0 }
        ],
        {
          duration: 1000,
          fill: 'forwards',
          easing: 'ease-out'
        }
      );
    });
  }, [node.text]);
  
  return (
    <g ref={contentRef}>
      <circle
        r={10}
        fill={node.isCounterfactual ? '#f97316' : '#2563eb'}
        className="node-circle"
      />
      <text
        dy=".35em"
        textAnchor="middle"
        className="node-text"
        fill="white"
      >
        {node.text}
      </text>
    </g>
  );
};

const ImportanceRing: React.FC<{ score: number }> = ({ score }) => {
  const radius = 12;
  const circumference = 2 * Math.PI * radius;
  const strokeDasharray = `${circumference * score} ${circumference}`;
  
  return (
    <circle
      r={radius}
      fill="none"
      stroke="#10b981"
      strokeWidth={2}
      strokeDasharray={strokeDasharray}
      className="importance-ring"
    />
  );
};

const AttentionIndicator: React.FC<{ pattern: number[][] }> = ({ pattern }) => {
  const attentionScore = useMemo(() => 
    calculateAttentionScore(pattern),
    [pattern]
  );
  
  return (
    <motion.path
      d={createAttentionPath(attentionScore)}
      fill="none"
      stroke="#f59e0b"
      strokeWidth={2}
      initial={{ pathLength: 0 }}
      animate={{ pathLength: 1 }}
      transition={{ duration: 1 }}
      className="attention-indicator"
    />
  );
};

// Link Components
interface LinkContainerProps {
  link: any;
  config: VisualizationConfig;
  isHighlighted: boolean;
  isHovered: boolean;
  children: React.ReactNode;
}

const LinkContainer: React.FC<LinkContainerProps> = ({
  link,
  config,
  isHighlighted,
  isHovered,
  children
}) => {
  const pathRef = useRef<SVGPathElement>(null);
  
  useEffect(() => {
    if (!pathRef.current) return;
    
    // Setup path animations
    const path = pathRef.current;
    const length = path.getTotalLength();
    
    path.style.strokeDasharray = `${length} ${length}`;
    path.style.strokeDashoffset = `${length}`;
    
    path.animate(
      [
        { strokeDashoffset: length },
        { strokeDashoffset: 0 }
      ],
      {
        duration: config.animation.duration,
        fill: 'forwards',
        easing: config.animation.easing
      }
    );
  }, [link.path]);
  
  return (
    <motion.g
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{
        duration: config.animation.duration,
        ease: config.animation.easing
      }}
      className={clsx(
        'link-container',
        isHighlighted && 'highlighted',
        isHovered && 'hovered'
      )}
      role="graphics-symbol"
      aria-label={`Link from ${link.source.text} to ${link.target.text}`}
    >
      {children}
    </motion.g>
  );
};

const LinkPath: React.FC<{ link: any }> = ({ link }) => {
  const pathRef = useRef<SVGPathElement>(null);
  
  const path = useMemo(() => 
    generateLinkPath(link.source, link.target, link.type),
    [link]
  );
  
  return (
    <path
      ref={pathRef}
      d={path}
      fill="none"
      stroke={getLinkColor(link)}
      strokeWidth={getLinkWidth(link)}
      markerEnd="url(#arrowhead)"
      className="link-path"
    />
  );
};

const MergeDecorator: React.FC = () => (
  <circle
    r={4}
    fill="#9333ea"
    className="merge-decorator"
  />
);

const CrossLinkDecorator: React.FC = () => (
  <path
    d="M-4,-4L4,4M-4,4L4,-4"
    stroke="#f97316"
    strokeWidth={2}
    className="cross-link-decorator"
  />
);

// Utility functions
const generateLinkPath = (source: any, target: any, type: string) => {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const dr = Math.sqrt(dx * dx + dy * dy);
  
  if (type === 'merge') {
    // Generate curved path for merge links
    return `M${source.x},${source.y}A${dr},${dr} 0 0,1 ${target.x},${target.y}`;
  } else if (type === 'cross') {
    // Generate S-curved path for cross links
    const midX = (source.x + target.x) / 2;
    const midY = (source.y + target.y) / 2;
    return `M${source.x},${source.y}Q${midX},${source.y} ${midX},${midY}Q${midX},${target.y} ${target.x},${target.y}`;
  }
  
  // Default straight path
  return `M${source.x},${source.y}L${target.x},${target.y}`;
};

const calculateAttentionScore = (pattern: number[][]) => {
  if (!pattern || pattern.length === 0) return 0;
  return pattern.reduce((acc, row) => 
    acc + row.reduce((sum, val) => sum + val, 0),
    0
  ) / (pattern.length * pattern[0].length);
};

// Would you like me to continue with the interaction handlers and layout managers next?