# XAIR - Xplainable AI Reasoning

## Explainable AI Pipeline for Multi-Domain Reasoning

## Overview

This project implements a comprehensive explanation generation pipeline that combines large language models with structured reasoning approaches. The system generates detailed explanations for various domains including ethical dilemmas, product recommendations, and career advice, while providing evidence, counterfactuals, and interactive probing capabilities.

## Features

- Multi-domain explanation generation
- Evidence-based reasoning with citation support
- Counterfactual scenario exploration
- Interactive probing questions
- Domain-specific evaluation metrics
- Enhanced model checkpointing
- Data augmentation capabilities

## Components

### Core Components (explanation.py)

- **EvidenceBase**: Manages evidence and citations for explanations
- **CounterfactualGenerator**: Generates alternative scenarios and variations
- **Prober**: Generates probing questions for interactive exploration

### Pipeline Components (pipeline.py)

- **FinalExplanationPipeline**: Main pipeline integrating all components
- **DataPreprocessor**: Handles data preprocessing and formatting
- **ExplanationEvaluator**: Evaluates generated explanations
- **EnhancedModelCheckpointer**: Manages model checkpoints
- **DomainMetrics**: Domain-specific evaluation metrics

### Data Generation (gpt4-data-gen.py)

- GPT-4 based data generation utilities
- Domain-specific templates
- Quality validation
- Batch processing capabilities

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/explainable-ai-pipeline.git
cd explainable-ai-pipeline
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
export OPENAI_API_KEY="your-api-key"  # On Windows: set OPENAI_API_KEY=your-api-key
```

## Usage

### Data Generation

```python
from gpt4_data_gen import ComprehensiveExplanationGenerator

generator = ComprehensiveExplanationGenerator(openai_api_key)
await generator.generate_dataset(num_examples=15000, output_file="data.json")
```

### Training the Pipeline

```python
from pipeline import FinalExplanationPipeline

pipeline = FinalExplanationPipeline(
    openai_api_key="your-api-key",
    llama_model_path="meta-llama/Llama-2-7b",
    training_data_path="data/data.json",
    output_dir="explanation_model"
)

checkpoint_path = pipeline.fine_tune(
    validation_split=0.1,
    save_format='complete',
    augment_data=True
)
```

### Generating Explanations

```python
explanation = pipeline.generate_explanation(
    query="Should I prioritize renewable energy investment over educational funding?",
    domain="ethical_dilemmas"
)

print(explanation['explanation'])
print("\nCounterfactuals:", explanation['counterfactuals'])
print("\nProbing Questions:", explanation['probing_questions'])
```

## Supported Domains

1. Ethical Dilemmas

   - Moral decision-making
   - Resource allocation
   - Policy impacts

2. Product Recommendations

   - Feature comparison
   - Value assessment
   - Use case analysis

3. Career Advice

   - Skill development
   - Market analysis
   - Career transitions

4. Multi-Criteria Decision Making (MCDM)
   - Trade-off analysis
   - Risk assessment
   - Option comparison

## Model Training

### Data Requirements

- Minimum 10,000 examples per domain
- Balanced distribution across domains
- High-quality explanations with evidence
- Diverse counterfactual scenarios

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.20+
- OpenAI API access
- CUDA-capable GPU (recommended)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the Llama team at Meta AI for the base model
- OpenAI for GPT-4 API access
- Dr. Shen Zhiqi for his support throughout
