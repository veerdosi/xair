# XAIR - Explainable AI Reasoning

**Explainable LLM Reasoning and Counterfactual Explanation Framework**

This framework implements a comprehensive system designed to provide in-depth explanations of a large language model's (LLM) decision-making process. The system generates reasoning trees, counterfactual explanations, and impact analyses for tasks such as text classification, question answering, and complex reasoning. The primary LLM under examination is GPT-4o, and the framework is tailored for domain experts and technical users who require high-detail, transparent explanations.

## Overview

The goal of this framework is to not only obtain model outputs but to also provide interpretable explanations that include:

- **Reasoning Trees**: Visual representations showing the LLM's decision path, including divergence points.
- **Counterfactual Explanations**: Alternative scenarios illustrating how slight changes in input could lead to different outcomes.
- **Feature Importance Analysis**: Assessing which features or decision points have the most significant impact on the final output.

These explanations are evaluated using several metrics to ensure both the performance of the underlying model and the quality of the explanations.

## Problem Scoping

- **LLM to Explain**: GPT-4o.
- **Target Tasks**:
  - Text Classification
  - Question Answering
  - Complex Reasoning Tasks
- **Target Audience**: Domain experts and technical users.
- **Level of Explanation Detail**:
  - In-depth explanations including reasoning trees that show the model's decision path.
  - Counterfactual explanations to illustrate alternative outcomes.
  - Feature importance analysis to highlight which parts of the reasoning are most critical.
- **Evaluation Metrics**:
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

A dedicated module (`backend/llm_interface.py`) interacts with the underlying LLM (GPT-4o) to send queries and retrieve responses along with token probabilities and attention data.

### 2. Reasoning Tree Generator

Implemented mainly in `backend/cgrt.py` and `backend/cgrt_tree.py`, this component:

- Generates a Cross-Generation Reasoning Tree (CGRT) by analyzing multiple generation paths.
- Identifies divergence points where the decision paths split.
- Uses attention flow techniques to compute node importance.

### 3. Counterfactual Generator and Integrator

- **Counterfactual Generator**: Based on the VCNet model (see `backend/counterfactual.py`), this module creates plausible and diverse counterfactual scenarios using a variational approach.
- **Counterfactual Integrator**: Located in `backend/counterfactual_integrator.py`, this module integrates the counterfactuals into the reasoning tree by matching similar nodes and adding cross-links.

### 4. Impact Analyzer

Implemented in `backend/impact_analyzer.py`, this module assesses the influence of generated counterfactuals on the overall decision-making process using metrics such as:

- Local impact
- Global impact
- Structural impact
- Plausibility

### 5. Visualization Engine and User Interface

- **Visualization Engine**: Built on top of D3.js (using the DependenTree library as seen in `backend/visualization_engine.js`), this engine renders interactive reasoning trees with integrated counterfactuals.
- **User Interface**: Provides a main display panel for the reasoning tree, a sidebar for interaction controls (zoom, filtering, counterfactual explorer), and a bottom panel for detailed textual explanations.

## Technical Implementation

The system uses a dual-approach:

- **Main LLM**: GPT-4o for generating responses.
- **Supporting Algorithms**: Separate models/algorithms (CGRT, VCNet, GYC Framework) are used to create, merge, and analyze explanations. This allows for flexibility and improved performance in generating rational inference paths and realistic counterfactuals.

Key steps include:

1. Generation of multiple reasoning paths with varying temperatures.
2. Identification of a shared prefix and divergence points across paths.
3. Construction of a reasoning tree that integrates counterfactual alternatives.
4. Analysis and ranking of counterfactual impacts using a composite score based on multiple contributing factors.

## Installation and Setup

### Prerequisites

- Python 3.8+
- OpenAI API key (for GPT-4o access)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/xair.git
cd xair
```

2. Install the dependencies:

```bash
pip install -r requirements.txt

# Install spaCy model (for advanced attention flow calculation)
python -m spacy download en_core_web_sm
```

3. Create a `.env` file in the root directory with your API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running the Main Application

```bash
python main.py --prompt "What is explainable AI?" --model "gpt-4o" --temperature 0.7
```

### Testing Components

```bash
# Test the LLM interface
python test_components.py --component llm

# Test the reasoning tree generator
python test_components.py --component tree

# Test all components
python test_components.py --component all
```

## Repository Structure

```
backend/
  ├── __init__.py              # Package initialization file
  ├── cgrt.py                  # Implements the generation of cross paths
  ├── cgrt_tree.py             # Constructs reasoning trees using CGRT
  ├── counterfactual.py        # Implements VCNet for counterfactual generation
  ├── counterfactual_integrator.py  # Integrates counterfactuals into trees
  ├── impact_analyzer.py       # Analyzes counterfactual impact on decision paths
  ├── llm_interface.py         # LLM interface for querying GPT-4o
  └── visualization_engine.js  # Visualization of reasoning trees with D3.js/DependenTree
.env                           # Environment variables (API keys, etc.)
.gitattributes
.gitignore
LICENSE
main.py                        # Main entry point for the application
README.md                      # This readme file
requirements.txt               # Python package dependencies
test_components.py             # Script to test individual components
```

## Evaluation Metrics

The framework includes multiple metrics to assess the effectiveness of the generated explanations:

- **Performance Difference (D)**: Measures how well the explanation logic matches the actual model's behavior.
- **Rule Count (R)**: Counts the number of rules in the explanation to evaluate complexity.
- **Feature Count (F)**: Measures how many features are used in the explanation.
- **Explanation Stability (S)**: Evaluates how consistent explanations are across similar inputs.
- **User Comprehension & Trust**: Evaluated via surveys or user feedback.
- **Task Performance Improvement**: Comparing outcomes with and without the explanation system.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests. For major changes, open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
