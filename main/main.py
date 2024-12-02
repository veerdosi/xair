# main.py

import torch
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
import logging
import argparse
from pathlib import Path
import yaml

from core.explainable_llm import ExplainableLLM
from data.data_generator import DataGenerator
from eval.evaluation_module import Evaluator
from training.trainer import Trainer
from interface.dashboard import ExplanationDashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_model(config: dict, device: torch.device) -> tuple:
    """Setup model and tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['tokenizer_name'])
    tokenizer.pad_token = tokenizer.eos_token

    model = ExplainableLLM(
        vocab_size=len(tokenizer),
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout']
    ).to(device)
    
    return model, tokenizer

def prepare_data(config: dict, tokenizer) -> tuple:
    """Prepare training, validation and test datasets"""
    data_generator = DataGenerator()
    
    try:
        dataset = DataGenerator.load_dataset(config['data']['dataset_path'])
        logger.info(f"Dataset loaded from {config['data']['dataset_path']}")
    except FileNotFoundError:
        logger.info("Generating new dataset...")
        dataset = data_generator.generate_dataset(num_samples=config['data']['num_samples'])
        data_generator.save_dataset(dataset, config['data']['dataset_path'])
        logger.info(f"Dataset saved to {config['data']['dataset_path']}")

    # Split dataset
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = int(config['data']['val_split'] * len(dataset))
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    
    return train_dataset, val_dataset, test_dataset

def train_model(model, train_dataset, val_dataset, config: dict, device: torch.device):
    """Train the model"""
    trainer = Trainer(
        model=model,
        device=device,
        config=config['training']
    )
    
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=config['training']['num_epochs']
    )
    
    return trainer

def evaluate_model(model, test_dataset, tokenizer, device: torch.device):
    """Evaluate the model"""
    evaluator = Evaluator(model, tokenizer, device)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    metrics = evaluator.evaluate_batch(test_dataloader)
    
    logger.info("Evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return metrics

def launch_dashboard(model, tokenizer, config: dict):
    """Launch the interactive dashboard"""
    dashboard = ExplanationDashboard(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    dashboard.launch(share=config['dashboard']['share'])

def main():
    parser = argparse.ArgumentParser(description='Explainable LLM Training and Evaluation')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'dashboard'], default='train', help='Operation mode')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup model and tokenizer
    model, tokenizer = setup_model(config, device)
    logger.info("Model initialized")

    if args.mode == 'train':
        # Prepare data
        train_dataset, val_dataset, test_dataset = prepare_data(config, tokenizer)
        
        # Train model
        trainer = train_model(model, train_dataset, val_dataset, config, device)
        
        # Evaluate model
        metrics = evaluate_model(model, test_dataset, tokenizer, device)
        
    elif args.mode == 'eval':
        # Load trained model
        checkpoint_path = Path(config['model']['checkpoint_path'])
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model from {checkpoint_path}")
        else:
            logger.error(f"No checkpoint found at {checkpoint_path}")
            return

        # Prepare test data and evaluate
        _, _, test_dataset = prepare_data(config, tokenizer)
        metrics = evaluate_model(model, test_dataset, tokenizer, device)
        
    elif args.mode == 'dashboard':
        # Load trained model if available
        checkpoint_path = Path(config['model']['checkpoint_path'])
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model from {checkpoint_path}")
            
        # Launch dashboard
        launch_dashboard(model, tokenizer, config)

if __name__ == "__main__":
    main()