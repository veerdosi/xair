# XAIR: Explainable AI Reasoning System

XAIR is a comprehensive system for generating, analyzing, and validating reasoning paths of large language models (LLMs). It provides powerful tools for understanding and evaluating how LLMs reason, with a focus on counterfactual analysis and knowledge graph validation.

![XAIR Overview](https://raw.githubusercontent.com/veerdosi/xai/assets/xair_overview.png)

## Overview

XAIR consists of three main components:

1. **CGRT (Counterfactual Graph Reasoning Tree)**: Generates multiple reasoning paths and builds a tree structure to visualize the model's reasoning process.
2. **Counterfactual Analysis**: Identifies critical tokens in the reasoning process and explores what happens when they're modified.
3. **Knowledge Graph Validation**: Maps reasoning statements to external knowledge in Wikidata and validates factual accuracy.

## Features

- Generate multiple reasoning paths with different temperature settings
- Identify divergence points where reasoning paths differ
- Construct a graph representation of the reasoning process
- Analyze attention patterns to identify important tokens
- Generate counterfactual alternatives to explore causal relationships
- Calculate Counterfactual Flip Rate (CFR) to quantify decision sensitivity
- Validate reasoning against Wikidata knowledge graph
- Calculate trustworthiness scores for reasoning paths
- Export visualizations for understanding model reasoning
- Optimized for CPUs, GPUs, and Apple Silicon (MPS)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Hugging Face Transformers 4.35+
- NetworkX 3.1+
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/veerdosi/xai.git
cd xai
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. For knowledge graph validation (optional):

```bash
python -m spacy download en_core_web_sm
```

## Quick Start

Run the main script with default settings:

```bash
python main.py
```

You'll be prompted to enter a query. The system will generate multiple reasoning paths, analyze them, and show you the results.

## Using Configuration Files

You can save and load configurations to easily reuse settings:

1. Create a configuration file:

```bash
python main.py --save-config my_config.json
```

2. Load a configuration file:

```bash
python main.py --config my_config.json
```

## Command Line Arguments

### Basic Settings

- `--model`: Model name or path (default: "meta-llama/Llama-3.2-1B")
- `--device`: Device to use - "cpu", "cuda", "mps", or "auto" (default: "auto")
- `--max-tokens`: Maximum tokens to generate (default: 256)
- `--verbose`: Enable verbose logging (flag)
- `--output-dir`: Output directory (default: "output")

### CGRT Settings

- `--temperatures`: Comma-separated temperatures for generation (default: "0.2,0.7,1.0")
- `--paths-per-temp`: Paths to generate per temperature (default: 1)

### Counterfactual Settings

- `--counterfactual-tokens`: Top-k tokens for counterfactual generation (default: 5)
- `--attention-threshold`: Minimum attention threshold for counterfactuals (default: 0.3)
- `--max-counterfactuals`: Maximum counterfactuals to generate (default: 20)

### Knowledge Graph Settings

- `--kg-use-local-model`: Use local sentence transformer model (flag)
- `--kg-similarity-threshold`: Minimum similarity threshold for KG entity mapping (default: 0.6)
- `--kg-skip`: Skip Knowledge Graph processing (useful for slower machines) (flag)

### Visualization Settings

- `--generate-visualizations`: Generate visualizations for the results (flag)

### Configuration Management

- `--config`: Path to configuration file
- `--save-config`: Save configuration to the specified file path

## Device Optimizations

XAIR automatically detects and optimizes for your available hardware:

### NVIDIA GPUs

```bash
python main.py --device cuda
```

### Apple Silicon (M1/M2/M3)

```bash
python main.py --device mps --max-tokens 256
```

### CPU Only

```bash
python main.py --device cpu --max-tokens 256
```

## Component Details

### CGRT

The Counterfactual Graph Reasoning Tree component:

- Generates multiple reasoning paths with different temperature settings
- Identifies points where reasoning paths diverge
- Analyzes token-level probabilities and attention patterns
- Constructs a directed graph representing all reasoning paths
- Calculates importance scores for nodes based on multiple factors

### Counterfactual Analysis

The Counterfactual component:

- Identifies tokens with high importance/attention scores
- Generates alternative versions by substituting these tokens
- Evaluates the impact of substitutions on the output
- Identifies "flip points" where small changes cause different conclusions
- Calculates Counterfactual Flip Rate (CFR) to quantify reasoning stability

### Knowledge Graph Validation

The Knowledge Graph component:

- Maps tokens and statements to Wikidata entities
- Validates factual statements against external knowledge
- Identifies supported statements, contradicted statements, and unverified claims
- Calculates trustworthiness scores for reasoning paths
- Provides detailed validation reports

## Visualizations

When you run XAIR with the `--generate-visualizations` flag, it creates several visualizations:

1. **Reasoning Tree**: Shows the structural relationships between reasoning steps
2. **Token Importance**: Highlights tokens with high importance and attention scores
3. **Counterfactual Impact**: Visualizes the impact of different token substitutions
4. **Knowledge Graph Validation**: Shows trustworthiness scores across reasoning paths
5. **Divergence Points**: Highlights where reasoning paths diverge

Visualizations are saved in the `output/visualizations` directory and can be viewed through the generated `index.html` file.

## Output Files

The system generates several outputs in the specified output directory:

### CGRT Outputs (`output/cgrt/`)

- `generation_results.json`: Raw generation results from the model
- `divergence_points.json`: Detected divergence points between paths
- `reasoning_tree.json`: The constructed reasoning tree in JSON format
- `path_comparison.txt`: Detailed comparison of different reasoning paths

### Counterfactual Outputs (`output/counterfactual/`)

- `counterfactuals.json`: Generated counterfactual candidates
- `counterfactual_evaluation.json`: Evaluation metrics for counterfactuals
- `counterfactual_comparison.txt`: Detailed comparison of counterfactuals
- `counterfactual_state.json`: Complete state of counterfactual analysis

### Knowledge Graph Outputs (`output/knowledge_graph/`)

- `entity_mapping.json`: Mapping of tokens to knowledge graph entities
- `validation_results.json`: Results of knowledge graph validation
- `validation_report.txt`: Detailed report of validation findings
- `kg_cache/`: Cache directory for knowledge graph requests

### Visualization Outputs (`output/visualizations/`)

- `index.html`: Entry point for viewing all visualizations
- `reasoning_tree.png`: Visualization of the reasoning tree
- `token_importance.png`: Chart of token importance scores
- `counterfactual_impact.png`: Visualization of counterfactual impact
- `kg_validation.png`: Knowledge graph validation results
- `divergence_points.png`: Visualization of divergence points

## Advanced Usage

### Processing Results Programmatically

You can import XAIR components into your own Python scripts:

```python
from backend.models.llm_interface import LlamaInterface
from backend.cgrt.cgrt_main import CGRT
from backend.counterfactual.counterfactual_main import Counterfactual
from backend.knowledge_graph.kg_main import KnowledgeGraph
from backend.utils.config import XAIRConfig

# Load configuration
config = XAIRConfig()
config.model_name_or_path = "meta-llama/Llama-3.2-1B"

# Initialize components
llm = LlamaInterface(model_name_or_path=config.model_name_or_path)
cgrt = CGRT(model_name_or_path=config.model_name_or_path)
counterfactual = Counterfactual()

# Process input
tree = cgrt.process_input("What is the capital of France?")
paths = cgrt.get_paths_text()
counterfactuals = counterfactual.generate_counterfactuals(cgrt.tree_builder, llm, "What is the capital of France?", cgrt.paths)

# Print results
print(f"Generated {len(paths)} reasoning paths")
for i, path_text in enumerate(paths):
    print(f"Path {i+1}: {path_text[:100]}...")
```

### Customizing Visualizations

You can customize visualizations using the functions in `backend/utils/viz_utils.py`:

```python
from backend.utils.viz_utils import plot_reasoning_tree, setup_visualization_style

# Set visualization style
setup_visualization_style(style="whitegrid", context="paper", font_scale=1.5, palette="viridis")

# Create custom tree visualization
plot_reasoning_tree(
    cgrt.tree_builder.graph,
    output_path="custom_tree.png",
    title="My Custom Reasoning Tree",
    highlight_nodes=["node_1", "node_5"],
    show_edge_labels=True
)
```

## Performance Tuning

### Limited Memory Environments

For systems with limited memory:

```bash
python main.py --model meta-llama/Llama-3.2-1B --max-tokens 128 --paths-per-temp 1 --kg-skip
```

### High Performance Setup

For detailed analysis on powerful hardware:

```bash
python main.py --model meta-llama/Llama-3.2-70B-Instruct --max-tokens 512 --paths-per-temp 3 --temperatures 0.1,0.5,0.9,1.3 --generate-visualizations
```

## Development

To contribute to XAIR:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## Running Tests

We use pytest for testing:

```bash
pytest tests/
```

For specific test files:

```bash
pytest tests/test_config.py
```

## License

[MIT License](LICENSE)

## Acknowledgements

This system builds on research in explainable AI, counterfactual analysis, and knowledge graph integration for language models. It incorporates techniques from:

- Counterfactual explanations
- Attention flow analysis
- Knowledge graph entity linking
- Semantic similarity measurement
- Token-level probability analysis

## Citation

If you use XAIR in your research, please cite:

```bibtex
@software{xair2023,
  author = {Veerdosi},
  title = {XAIR: Explainable AI Reasoning System},
  year = {2023},
  url = {https://github.com/veerdosi/xair}
}
```