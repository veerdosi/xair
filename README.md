# XAIR - Explainable AI Reasoning

**A comprehensive system for visualizing and understanding the decision-making processes of large language models through reasoning trees and counterfactual analysis.**

## Overview

XAIR (Explainable AI Reasoning) is a framework designed to provide in-depth explanations of large language model (LLM) decision-making. The system generates visual reasoning trees, creates counterfactual scenarios, and analyzes the impact of these counterfactuals on the model's reasoning. XAIR is aimed at domain experts and technical users who require high-detail, transparent explanations of how LLMs arrive at their conclusions.

## Key Features

- **Cross-Generation Reasoning Trees (CGRT)**: Visual representations showing the LLM's decision path, including divergence points and alternative reasoning routes
- **Counterfactual Generation**: Automatic generation of alternative scenarios that highlight how slight changes in input could lead to different outcomes
- **Impact Analysis**: Comprehensive assessment of which decision points have the most significant impact on the final output
- **Interactive Visualization**: Rich, responsive interface for exploring reasoning paths and their variants

## Architecture

XAIR is built with a modular architecture consisting of:

### Backend

- **LLM Interface**: Python-based API for interacting with GPT-4o
- **CGRT Generator**: Creates reasoning trees from multiple generation paths
- **Counterfactual Generator**: Generates realistic alternative scenarios
- **Impact Analyzer**: Assesses the significance of different counterfactuals
- **FastAPI Server**: Exposes backend functionality through RESTful endpoints

### Frontend

- **React-based UI**: Modern, responsive interface
- **D3.js Visualizations**: Interactive tree diagrams and impact charts
- **Custom Hooks**: Specialized data fetching and processing
- **Tailwind CSS**: Consistent, clean styling system

## Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- OpenAI API key (for GPT-4o access)

### Backend Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/xair.git
cd xair
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install backend dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_api_key_here
```

5. Start the backend server:

```bash
python server.py
```

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm start
```

4. The application will be available at http://localhost:3000

## Usage Guide

### Generating a Reasoning Tree

1. Enter a prompt in the input field on the dashboard
2. Adjust generation settings if needed
3. Click "Generate"
4. The system will create a reasoning tree showing how the LLM processes the prompt

### Exploring the Tree

1. Navigate the tree using pan and zoom controls
2. Click on nodes to see detailed information
3. Toggle between tree and list views for different perspectives
4. Use filters to focus on specific aspects of the reasoning

### Generating Counterfactuals

1. From the tree view, click "Generate Counterfactuals"
2. The system will automatically create alternative scenarios
3. View counterfactuals in the side panel

### Analyzing Impact

1. After generating counterfactuals, the system automatically analyzes their impact
2. View the impact analysis to understand which factors most influence the model's reasoning
3. See key insights derived from the impact analysis

## Technical Implementation

### Cross-Generation Reasoning Tree (CGRT)

XAIR uses the CGRT algorithm to construct tree representations of LLM decision-making:

1. **Multiple Path Generation**: Creates several reasoning paths with varying temperature settings
2. **Divergence Point Identification**: Pinpoints where predicted tokens differ
3. **Tree Construction**: Builds a network structure representing decision points
4. **Attention Analysis**: Uses attention flow to determine node importance

### Counterfactual Generation (VCNet-inspired)

The system creates meaningful counterfactuals using:

1. **Token Substitutions**: Replacing key tokens that could change reasoning
2. **Context Modifications**: Altering the context to explore different reasoning paths
3. **Semantic Alternatives**: Generating completely different approaches to the same problem

### Impact Analysis

XAIR evaluates counterfactuals using multiple metrics:

1. **Local Impact**: Changes to the immediate decision point
2. **Global Impact**: Effects on final outcomes
3. **Structural Impact**: Changes to the overall reasoning structure
4. **Plausibility**: How realistic the alternative is

## Evaluation Metrics

XAIR assesses explanation quality using:

- **Performance Difference (D)**: How well explanation logic matches model behavior
- **Rule Count (R)**: Complexity measure of the explanation
- **Feature Count (F)**: Number of features used in the explanation
- **Explanation Stability (S)**: Consistency across similar inputs
- **User Comprehension**: Evaluated via expert review
- **Trust**: Measured through user reliance on the system
- **Task Performance**: Comparing outcomes with and without explanations

## Folder Structure

```
xair/
├── backend/
│   ├── cgrt.py                  # Cross-Generation Reasoning Tree implementation
│   ├── cgrt_tree.py             # Tree structure builder
│   ├── counterfactual.py        # Counterfactual generation
│   ├── counterfactual_integrator.py  # Integrates counterfactuals into trees
│   ├── impact_analyzer.py       # Analyzes counterfactual impact
│   ├── llm_interface.py         # Interface to GPT-4o
│   ├── error_handling.py        # Error handling utilities
│   └── visualization-engine.js  # D3.js visualization helpers
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── layout/          # Layout components
│   │   │   ├── visualizations/  # Visualization components
│   │   │   ├── ui/              # UI components
│   │   │   ├── forms/           # Form components
│   │   │   └── common/          # Common utilities
│   │   ├── hooks/               # Custom React hooks
│   │   ├── services/            # API services
│   │   ├── utils/               # Utility functions
│   │   ├── contexts/            # React contexts
│   │   ├── pages/               # Page components
│   │   └── styles/              # CSS styles
│   ├── public/                  # Static assets
│   └── package.json             # Frontend dependencies
├── tests/                       # Test files
├── .env                         # Environment variables
├── LICENSE                      # License file
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── server.py                    # FastAPI server
```

## API Endpoints

### `/api/generate`

- **Method**: POST
- **Purpose**: Generate a reasoning tree
- **Request Body**: Prompt and generation settings
- **Response**: Tree data with visualization information

### `/api/counterfactuals`

- **Method**: POST
- **Purpose**: Generate counterfactuals for a tree
- **Request Body**: Tree ID
- **Response**: Counterfactual data

### `/api/analyze`

- **Method**: POST
- **Purpose**: Analyze impact of counterfactuals
- **Request Body**: Tree ID and optional counterfactual IDs
- **Response**: Impact analysis data

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Special thanks to the developers of D3.js, React, and FastAPI
- Inspired by work on counterfactual explanations and reasoning trees in AI systems

---
