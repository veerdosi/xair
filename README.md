# XAIR: Explainable AI Reasoning System

XAIR is a comprehensive system for generating, analyzing, and validating reasoning paths of large language models (LLMs). It provides powerful tools for understanding and evaluating how LLMs reason, with a focus on counterfactual analysis and knowledge graph validation.

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
- MacOS friendly optimizations for efficient processing

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Hugging Face Transformers
- NetworkX
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd xair
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. For knowledge graph validation (optional):

```bash
pip install -r knowledge_graph/requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

Run the main script:

```bash
python main.py --model meta-llama/Llama-3.2-1B --device auto
```

### Command Line Arguments

- `--model`: Model name or path (default: "meta-llama/Llama-3.2-1B")
- `--device`: Device to use - "cpu", "cuda", "mps", or "auto" (default: "auto")
- `--max-tokens`: Maximum tokens to generate (default: 512)
- `--temperatures`: Comma-separated temperatures for generation (default: "0.2,0.7,1.0")
- `--paths-per-temp`: Paths to generate per temperature (default: 1)
- `--counterfactual-tokens`: Top-k tokens for counterfactual generation (default: 5)
- `--attention-threshold`: Minimum attention threshold for counterfactuals (default: 0.3)
- `--max-counterfactuals`: Maximum counterfactuals to generate (default: 20)
- `--kg-use-local-model`: Use local sentence transformer model (flag)
- `--kg-similarity-threshold`: Minimum similarity threshold for KG entity mapping (default: 0.6)
- `--kg-skip`: Skip Knowledge Graph processing (useful for slower machines) (flag)
- `--output-dir`: Output directory (default: "output")
- `--verbose`: Enable verbose logging (flag)

### MacOS Specific Optimizations

For MacOS users, we recommend the following settings for optimal performance:

```bash
python main.py --model meta-llama/Llama-3.2-1B --device mps --max-tokens 256 --kg-skip
```

If you want to use the Knowledge Graph component on a Mac, consider using these settings:

```bash
python main.py --model meta-llama/Llama-3.2-1B --device mps --max-tokens 256 --kg-use-local-model --kg-similarity-threshold 0.7
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

## Example

Here's a simple example of using XAIR:

```bash
python main.py --model meta-llama/Llama-3.2-1B --device auto
```

Then, enter a prompt when prompted:

```
Enter your prompt (or 'quit' to exit): What is the capital of France and when did the Eiffel Tower open?
```

XAIR will:

1. Generate multiple reasoning paths
2. Identify key tokens and divergence points
3. Generate counterfactual alternatives
4. Validate statements against Wikidata
5. Provide a summary of findings with trustworthiness scores

## Development

To contribute to XAIR:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

[MIT License](LICENSE)

## Acknowledgements

This system builds on research in explainable AI, counterfactual analysis, and knowledge graph integration for language models. It incorporates techniques from:

- Counterfactual explanations
- Attention flow analysis
- Knowledge graph entity linking
- Semantic similarity measurement
- Token-level probability analysis
