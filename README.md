# XAIR - Xplainable AI Reasoning

**Explainable LLM Reasoning and Counterfactual Explanation Framework**

This repository implements a comprehensive framework designed to provide in-depth explanations of a large language model's (LLM) decision-making process. The system generates reasoning trees, counterfactual explanations, and impact analyses for tasks such as text classification, question answering, and complex reasoning. The primary LLM under examination is GPT-4, and the framework is tailored for domain experts and technical users who require high-detail, transparent explanations.

## Overview

The goal of this framework is to not only obtain model outputs but to also provide interpretable explanations that include:

- Reasoning Trees: Visual representations showing the LLM's decision path, including divergence points.
- Counterfactual Explanations: Alternative scenarios illustrating how slight changes in input could lead to different outcomes.
- Feature Importance Analysis: Assessing which features or decision points have the most significant impact on the final output.

These explanations are evaluated using several metrics (described below) to ensure both the performance of the underlying model and the quality of the explanations.

## Problem Scoping

- LLM to Explain: GPT-4.
- Target Tasks:
  - Text Classification
  - Question Answering
  - Complex Reasoning Tasks
- Target Audience: Domain experts and technical users.
- Level of Explanation Detail:
  - In-depth explanations including reasoning trees that show the model's decision path.
  - Counterfactual explanations to illustrate alternative outcomes.
  - Feature importance analysis to highlight which parts of the reasoning are most critical.
- Evaluation Metrics:
  - D: Performance difference between the agent's model and the explanation logic.
  - R: Number of rules in the agent's explanation.
  - F: Number of features used to construct the explanation.
  - S: Stability of the agent's explanation.
  - User comprehension: Evaluated via surveys or interviews with domain experts.
  - Trust in the system: Monitored by tracking user reliance on model outputs.
  - Task performance improvement: Comparing user performance on tasks with and without the explainability module.

## Framework Design

The repository is organized into several core components:

### 1. LLM Interface

A dedicated module (`response-llm.py`) interacts with the underlying LLM (GPT-4) to send queries and retrieve responses along with token probabilities and attention data.

### 2. Reasoning Tree Generator

Implemented mainly in `ver3.0/cgrt.py` and `ver3.0/cgrt-tree.py`, this component:

- Generates a Cross-Generation Reasoning Tree (CGRT) by analyzing multiple generation paths.
- Identifies divergence points where the decision paths split.
- Uses attention flow techniques to compute node importance.

### 3. Counterfactual Generator and Integrator

- Counterfactual Generator: Based on the VCNet model (see `ver3.0/counterfactual.py`), this module creates plausible and diverse counterfactual scenarios using a variational approach.
- Counterfactual Integrator: Located in `ver3.0/counterfactual-integrator.py`, this module integrates the counterfactuals into the reasoning tree by matching similar nodes and adding cross-links.

### 4. Impact Analyzer

Implemented in `ver3.0/impact-analyzer.py`, this module assesses the influence of generated counterfactuals on the overall decision-making process using metrics such as:

- Local impact
- Global impact
- Structural impact
- Plausibility

### 5. Visualization Engine and User Interface

- Visualization Engine: Built on top of D3.js (using the DependenTree library as seen in `ver3.0/visualization-engine.js`), this engine renders interactive reasoning trees with integrated counterfactuals.
- User Interface: Provides a main display panel for the reasoning tree, a sidebar for interaction controls (zoom, filtering, counterfactual explorer), and a bottom panel for detailed textual explanations.

## Technical Implementation

The system uses a dual-approach:

- Main LLM: GPT-4 for generating responses.
- Supporting Algorithms: Separate models/algorithms (CGRT, VCNet, GYC Framework) are used to create, merge, and analyze explanations. This allows for flexibility and improved performance in generating rational inference paths and realistic counterfactuals.

Key steps include:

1. Generation of multiple reasoning paths with varying temperatures.
2. Identification of a shared prefix and divergence points across paths.
3. Construction of a reasoning tree that integrates counterfactual alternatives.
4. Analysis and ranking of counterfactual impacts using a composite score based on multiple contributing factors.

## Repository Structure

```
ver3.0/
  ├── cgrt-tree.py            # Constructs reasoning trees using CGRT
  ├── cgrt.py                 # Implements the generation of cross paths
  ├── counterfactual-integrator.py  # Integrates counterfactuals into trees
  ├── counterfactual.py       # Implements VCNet for counterfactual generation
  ├── impact-analyzer.py      # Analyzes counterfactual impact on decision paths
  ├── response-llm.py         # LLM interface for querying GPT-4
  ├── visualization-engine.js # Visualization of reasoning trees with D3.js/DependenTree
.gitignore
LICENSE
README.md                   # This file
requirements.txt            # Python package dependencies
```

## Getting Started

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

2. Install dependencies:
   Ensure you have Python 3.8+ installed. Then run:

```bash
pip install -r requirements.txt
```

You may also need to install Node.js dependencies for the visualization UI if you plan to run the frontend.

### Usage

- Run Examples:
  - For synchronous counterfactual generation using VCNet:

```bash
python ver3.0/counterfactual.py
```

- For asynchronous LLM integration and counterfactual integration:

```bash
python ver3.0/impact-analyzer.py
```

- User Interface:
  Launch the UI (if provided) by following the instructions in the documentation for the front-end application.

### Configuration

Update API keys and configuration settings in `response-llm.py` and other modules as needed. For example, replace "your_api_key_here" with your actual API key for the DeepSeek service.

## Evaluation Metrics

The framework includes multiple metrics to assess the effectiveness of the generated explanations:

- Performance Difference (D)
- Rule Count (R)
- Feature Count (F)
- Explanation Stability (S)
- User Comprehension & Trust: Evaluated via surveys or user feedback.
- Task Performance Improvement: Comparing outcomes with and without the explanation system.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests. For major changes, open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.
