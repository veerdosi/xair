import torch
from transformers import GPT2Tokenizer
from data_generator import DataGenerator
from explainable_llm import ExplainableLLM
from trainer import Trainer, QAExplanationDataset
from evaluation_module import Evaluator
from visualization_module import Visualizer

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Set hyperparameters
    vocab_size = len(tokenizer)
    d_model = 768
    nhead = 12
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 3072
    dropout = 0.1

    # Initialize model
    model = ExplainableLLM(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
    print("Model initialized")

    # Generate or load dataset
    data_file = "diverse_qa_dataset.pth"
    try:
        dataset = DataGenerator.load_dataset(data_file)
        print(f"Dataset loaded from {data_file}")
    except FileNotFoundError:
        print("Generating new dataset...")
        data_generator = DataGenerator()
        dataset = data_generator.generate_dataset(num_samples=1000)
        data_generator.save_dataset(dataset, data_file)
        print(f"Dataset saved to {data_file}")

    # Prepare datasets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_dataset = QAExplanationDataset(dataset[:train_size], tokenizer, max_length=100)
    val_dataset = QAExplanationDataset(dataset[train_size:train_size+val_size], tokenizer, max_length=100)
    test_dataset = QAExplanationDataset(dataset[train_size+val_size:], tokenizer, max_length=100)

    # Initialize trainer
    trainer = Trainer(model, tokenizer, device, checkpoint_dir='checkpoints', log_dir='logs')

    # Train the model
    trainer.train(train_dataset, val_dataset, batch_size=16, num_epochs=5, checkpoint_interval=1)

    # Initialize evaluator
    evaluator = Evaluator(model, tokenizer, device)

    # Evaluate the model
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
    metrics = evaluator.evaluate_batch(test_dataloader)
    print("Evaluation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Initialize visualizer
    visualizer = Visualizer(tokenizer)

    # Generate QA with explanation and visualize
    while True:
        question = input("Enter a question (or 'q' to quit): ")
        if question.lower() == 'q':
            break
        
        answer, explanation, attention_weights, attention_patterns = trainer.generate_qa_with_explanation(question)
        print(f"\nAnswer: {answer}")
        print(f"\nExplanation: {explanation}")
        
        # Visualize attention
        visualizer.visualize_explanation(question, answer, explanation, attention_weights, attention_patterns)
        
        # Evaluate the generated answer and explanation
        faithfulness = evaluator.evaluate_faithfulness(question, answer, explanation)
        coherence = evaluator.evaluate_coherence(explanation)
        print(f"\nFaithfulness: {faithfulness:.4f}")
        print(f"Coherence: {coherence:.4f}")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
