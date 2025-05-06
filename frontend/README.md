# XAI Chat Interface

An interactive chat interface for the Explainable AI (XAI) system that visualizes AI reasoning processes using the Counterfactual Graph Reasoning Tree (CGRT) framework.

## Features

- **Interactive Chat:** Natural language conversation with an explainable AI system
- **Multiple Visualization Types:**
  - Reasoning Tree: View the AI's decision-making process as a tree structure
  - Token Importance: See the relative importance of each token in the reasoning
  - Counterfactual Analysis: Discover how changing specific tokens would affect the outcome
  - Knowledge Graph: Explore connections between reasoning and external knowledge
  - Divergence Points: Identify where different reasoning paths diverge
- **Node Details:** Click on any node in the visualization to see detailed information
- **Customizable Settings:** Configure model parameters, CGRT settings, and visualization options
- **Export Functionality:** Save visualizations for documentation or further analysis

## Getting Started

### Prerequisites

- Python 3.8 or higher
- All requirements listed in the main project's `requirements.txt`

### Installation

The XAI Chat Interface is integrated with the main XAI project. No additional installation is required beyond the main project setup.

### Running the Interface

From the project root directory, run:

```bash
python run.py
```

This will start both the backend API server and the frontend interface. A browser window should open automatically at http://localhost:5000.

#### Command Line Options

- `--port PORT`: Set the port for the API server (default: 5000)
- `--no-browser`: Do not open a browser window automatically
- `--debug`: Run in debug mode with detailed logging
- `--model MODEL`: Specify the model name or path
- `--skip-kg`: Skip Knowledge Graph processing (useful for faster startup on slower machines)

Example:
```bash
python run.py --port 8000 --model meta-llama/Llama-3.2-8B
```

## Using the Interface

1. **Chat Interaction:**
   - Type your message in the input box at the bottom of the chat panel
   - Press Enter or click the send button to submit your message
   - The AI will respond and generate visualizations of its reasoning process

2. **Visualization Options:**
   - Select the visualization type from the dropdown in the sidebar
   - Adjust the importance threshold slider to filter nodes
   - Check/uncheck paths to compare different reasoning paths
   - Click on nodes in the visualization for detailed information
   - Use the "Export Visualization" button to save the current visualization

3. **Settings:**
   - Click the "Settings" button in the header to configure the system
   - Adjust model settings, temperature values, and other parameters
   - Click "Save Settings" to apply your changes
   - Settings are saved locally and will persist between sessions

## Understanding the Visualizations

### Reasoning Tree

The reasoning tree visualization shows the AI's thought process as a directed graph. Each node represents a reasoning step, with:

- **Node Size:** Represents the importance of the node
- **Node Color:** Indicates importance (darker = more important)
- **Red Outline:** Marks divergence points where reasoning paths split
- **Edge Thickness:** Shows the strength of connection between nodes

### Token Importance

This visualization displays the relative importance of each token in the reasoning:

- **Bar Height:** Represents token importance
- **Color Intensity:** Indicates importance (darker = more important)
- **Red Markers:** Show tokens with high attention scores

### Counterfactual Analysis

The counterfactual visualization shows how changing specific tokens would affect the outcome:

- **Bar Height:** Represents the impact of the change
- **Color:** Red for changes that flip the outcome, blue for those that don't
- **Token Pairs:** Original token → Alternative token

### Knowledge Graph Validation

This visualization compares reasoning statements with a knowledge graph:

- **Green Bars:** Statements supported by the knowledge graph
- **Red Bars:** Statements contradicted by the knowledge graph
- **Gray Bars:** Statements not verified by the knowledge graph
- **Blue Line:** Overall trustworthiness score for each reasoning path

### Divergence Points

This visualization highlights points where reasoning paths diverge:

- **Orange Highlights:** Mark tokens where paths diverge
- **Red Line:** Shows severity of divergence at each point

## Architecture

The XAI Chat Interface consists of:

1. **Frontend:**
   - HTML/CSS/JavaScript interface using Tailwind CSS for styling
   - D3.js for interactive visualizations
   - Modular JS architecture with separated concerns

2. **Backend API:**
   - Flask-based REST API
   - Integrates with CGRT, Counterfactual, and Knowledge Graph components
   - JSON-based communication with the frontend

## License

This interface is part of the main XAI project and shares the same license.

## Acknowledgments

This interface was created to provide an intuitive way to explore and understand AI reasoning processes using the CGRT framework.
