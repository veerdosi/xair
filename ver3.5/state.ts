// State Management
interface TreeState {
  data: CGRTTree;
  layout: LayoutState;
  interaction: InteractionState;
  animation: AnimationState;
  meta: MetaState;
}

interface LayoutState {
  type: 'hierarchical' | 'force' | 'radial';
  nodes: Map<string, NodeLayout>;
  edges: Map<string, EdgeLayout>;
  dimensions: { width: number; height: number };
  dirty: boolean;
}

interface InteractionState {
  mode: 'explore' | 'precise';
  selected: Set<string>;
  hovered: string | null;
  transform: d3.ZoomTransform;
  dragState: DragState | null;
}

interface AnimationState {
  active: Map<string, Animation>;
  queue: Animation[];
  paused: boolean;
}

interface MetaState {
  error: Error | null;
  loading: boolean;
  lastUpdate: number;
  dirtyFlags: Set<StateComponent>;
}

type StateComponent = 'data' | 'layout' | 'interaction' | 'animation';

class TreeStateManager {
  private state: TreeState;
  private subscribers: Map<StateComponent, Set<(state: TreeState) => void>>;
  private updateQueue: Map<StateComponent, UpdateTask[]>;
  private snapshots: TreeState[];
  private maxSnapshots: number = 10;
  
  constructor(initialState: TreeState) {
    this.state = initialState;
    this.subscribers = new Map();
    this.updateQueue = new Map();
    this.snapshots = [];
    
    this.setupStateManagement();
  }
  
  private setupStateManagement() {
    // Set up update loop
    requestAnimationFrame(() => this.processUpdates());
    
    // Set up error boundary
    window.addEventListener('error', (event) => {
      this.handleError(event.error);
    });
  }
  
  public subscribe(
    component: StateComponent,
    callback: (state: TreeState) => void
  ): () => void {
    if (!this.subscribers.has(component)) {
      this.subscribers.set(component, new Set());
    }
    
    this.subscribers.get(component)!.add(callback);
    
    return () => {
      this.subscribers.get(component)?.delete(callback);
    };
  }
  
  public dispatch(
    component: StateComponent,
    update: (state: TreeState) => Partial<TreeState>,
    priority: 'high' | 'medium' | 'low' = 'medium'
  ) {
    // Create update task
    const task: UpdateTask = {
      update,
      priority,
      timestamp: Date.now(),
      id: Math.random().toString(36).substr(2, 9)
    };
    
    // Add to queue
    if (!this.updateQueue.has(component)) {
      this.updateQueue.set(component, []);
    }
    this.updateQueue.get(component)!.push(task);
    
    // Mark component as dirty
    this.state.meta.dirtyFlags.add(component);
    
    // Process immediately if high priority
    if (priority === 'high') {
      this.processComponentUpdates(component);
    }
  }
  
  private processUpdates() {
    // Process updates by priority
    this.processHighPriorityUpdates();
    this.processMediumPriorityUpdates();
    this.processLowPriorityUpdates();
    
    // Schedule next update
    requestAnimationFrame(() => this.processUpdates());
  }
  
  private processComponentUpdates(component: StateComponent) {
    const tasks = this.updateQueue.get(component) || [];
    if (tasks.length === 0) return;
    
    // Take snapshot before updates
    this.takeSnapshot();
    
    try {
      // Process all tasks for component
      tasks.sort((a, b) => {
        // Sort by priority then timestamp
        if (a.priority !== b.priority) {
          return getPriorityValue(a.priority) - getPriorityValue(b.priority);
        }
        return a.timestamp - b.timestamp;
      });
      
      let newState = { ...this.state };
      
      for (const task of tasks) {
        newState = {
          ...newState,
          ...task.update(newState)
        };
      }
      
      // Validate state before applying
      if (this.validateState(newState)) {
        this.state = newState;
        this.notifySubscribers(component);
      } else {
        throw new Error('Invalid state after updates');
      }
      
    } catch (error) {
      // Rollback to last snapshot
      this.rollback();
      this.handleError(error);
    }
    
    // Clear processed tasks
    this.updateQueue.set(component, []);
    this.state.meta.dirtyFlags.delete(component);
  }
  
  private takeSnapshot() {
    this.snapshots.push({ ...this.state });
    if (this.snapshots.length > this.maxSnapshots) {
      this.snapshots.shift();
    }
  }
  
  private rollback() {
    if (this.snapshots.length > 0) {
      this.state = this.snapshots.pop()!;
      this.notifySubscribers('data');
      this.notifySubscribers('layout');
      this.notifySubscribers('interaction');
      this.notifySubscribers('animation');
    }
  }
  
  private validateState(state: TreeState): boolean {
    // Perform state validation
    try {
      // Check data consistency
      if (!this.validateDataState(state.data)) return false;
      
      // Check layout validity
      if (!this.validateLayoutState(state.layout)) return false;
      
      // Check interaction state
      if (!this.validateInteractionState(state.interaction)) return false;
      
      // Check animation state
      if (!this.validateAnimationState(state.animation)) return false;
      
      return true;
    } catch (error) {
      this.handleError(error);
      return false;
    }
  }
  
  private notifySubscribers(component: StateComponent) {
    this.subscribers.get(component)?.forEach(callback => {
      try {
        callback(this.state);
      } catch (error) {
        this.handleError(error);
      }
    });
  }
  
  private handleError(error: Error) {
    console.error('State management error:', error);
    
    // Update error state
    this.state.meta.error = error;
    
    // Notify error subscribers
    this.notifySubscribers('meta');
    
    // Attempt recovery if possible
    this.attemptRecovery();
  }
  
  private attemptRecovery() {
    // If we have snapshots, roll back
    if (this.snapshots.length > 0) {
      this.rollback();
      return;
    }
    
    // If no snapshots, try to reset to initial state
    // This is a last resort
    this.state = this.createInitialState();
    this.notifyAllSubscribers();
  }
  
  private notifyAllSubscribers() {
    for (const component of this.subscribers.keys()) {
      this.notifySubscribers(component);
    }
  }
}

// Update System
class UpdateCoordinator {
  private stateManager: TreeStateManager;
  private layoutManager: LayoutManager;
  private interactionManager: InteractionManager;
  private animationManager: AnimationManager;
  
  constructor(
    stateManager: TreeStateManager,
    layoutManager: LayoutManager,
    interactionManager: InteractionManager,
    animationManager: AnimationManager
  ) {
    this.stateManager = stateManager;
    this.layoutManager = layoutManager;
    this.interactionManager = interactionManager;
    this.animationManager = animationManager;
    
    this.setupCoordination();
  }
  
  private setupCoordination() {
    // Subscribe to state changes
    this.stateManager.subscribe('data', state => {
      this.onDataUpdate(state);
    });
    
    this.stateManager.subscribe('layout', state => {
      this.onLayoutUpdate(state);
    });
    
    this.stateManager.subscribe('interaction', state => {
      this.onInteractionUpdate(state);
    });
    
    this.stateManager.subscribe('animation', state => {
      this.onAnimationUpdate(state);
    });
  }
  
  public requestUpdate(
    component: StateComponent,
    update: UpdateRequest
  ) {
    // Validate update request
    if (!this.validateUpdateRequest(update)) {
      return;
    }
    
    // Determine update priority
    const priority = this.calculateUpdatePriority(component, update);
    
    // Create state update function
    const updateFn = (state: TreeState) => {
      switch (component) {
        case 'data':
          return this.createDataUpdate(state, update);
        case 'layout':
          return this.createLayoutUpdate(state, update);
        case 'interaction':
          return this.createInteractionUpdate(state, update);
        case 'animation':
          return this.createAnimationUpdate(state, update);
        default:
          throw new Error(`Unknown component: ${component}`);
      }
    };
    
    // Dispatch update
    this.stateManager.dispatch(component, updateFn, priority);
  }
  
  private validateUpdateRequest(update: UpdateRequest): boolean {
    // Perform update validation
    try {
      return true;
    } catch (error) {
      console.error('Invalid update request:', error);
      return false;
    }
  }
  
  private calculateUpdatePriority(
    component: StateComponent,
    update: UpdateRequest
  ): 'high' | 'medium' | 'low' {
    // Calculate priority based on component and update type
    switch (component) {
      case 'interaction':
        return 'high';
      case 'layout':
        return update.type === 'immediate' ? 'high' : 'medium';
      case 'animation':
        return 'low';
      default:
        return 'medium';
    }
  }
}