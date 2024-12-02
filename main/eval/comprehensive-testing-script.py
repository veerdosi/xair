import torch
from torch.utils.data import DataLoader
from explainable_llm import ExplainableLLM
from main.training.trainer import Trainer, QAExplanationDataset
from evaluation_module import Evaluator
from visualization_module import Visualizer
from human_evaluation_protocol import HumanEvaluationProtocol
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_testing(model, tokenizer, test_dataset, baseline_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(model, tokenizer, device)
    visualizer = Visualizer(tokenizer)
    human_eval = HumanEvaluationProtocol()

    # Automated evaluation
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    metrics = evaluator.evaluate_batch(test_dataloader)
    print("Automated Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Baseline comparison (if available)
    if baseline_model:
        baseline_metrics = Evaluator(baseline_model, tokenizer, device).evaluate_batch(test_dataloader)
        print("\nBaseline Model Metrics:")
        for metric, value in baseline_metrics.items():
            print(f"{metric}: {value:.4f}")

    # Human evaluation
    human_eval_results = []
    for i, sample in enumerate(test_dataset):
        if i >= 50:  # Limit to 50 samples for human evaluation
            break
        question = tokenizer.decode(sample['question'], skip_special_tokens=True)
        answer, explanation, _, _ = model.generate_qa_with_explanation(question)
        eval_result = human_eval.evaluate_explanation(question, answer, explanation, evaluator_id=1)
        human_eval_results.append(eval_result)

    # Analyze human evaluation results
    df = pd.DataFrame(human_eval_results)
    print("\nHuman Evaluation Results:")
    print(df.describe())

    # Visualize human evaluation results
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['plausibility', 'faithfulness', 'coherence', 'usefulness']])
    plt.title("Distribution of Human Evaluation Scores")
    plt.show()

    # Error analysis
    error_cases = []
    for sample in test_dataset:
        question = tokenizer.decode(sample['question'], skip_special_tokens=True)
        true_answer = tokenizer.decode(sample['answer'], skip_special_tokens=True)
        generated_answer, explanation, _, _ = model.generate_qa_with_explanation(question)
        if generated_answer.strip().lower() != true_answer.strip().lower():
            error_cases.append({
                'question': question,
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'explanation': explanation
            })

    print(f"\nNumber of error cases: {len(error_cases)}")
    if error_cases:
        print("Sample error case:")
        print(f"Question: {error_cases[0]['question']}")
        print(f"True Answer: {error_cases[0]['true_answer']}")
        print(f"Generated Answer: {error_cases[0]['generated_answer']}")
        print(f"Explanation: {error_cases[0]['explanation']}")

    # Visualization of attention patterns for a sample
    sample_question = tokenizer.decode(test_dataset[0]['question'], skip_special_tokens=True)
    answer, explanation, attention_weights, attention_patterns = model.generate_qa_with_explanation(sample_question)
    visualizer.visualize_explanation(sample_question, answer, explanation, attention_weights, attention_patterns)

# Usage in main script:
# comprehensive_testing(model, tokenizer, test_dataset, baseline_model)
