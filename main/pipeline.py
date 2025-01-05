import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    IntervalStrategy
)
from torch.utils.data import Dataset
from openai import OpenAI
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import os
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import evaluate
from datetime import datetime
import logging
from rouge_score import rouge_scorer
import random
import shutil
from nltk.translate.meteor_score import meteor_score
import nltk
from datasets import load_metric
import pandas as pd

class DataAugmentor:
    """Handles data augmentation strategies"""
    
    def __init__(self, domains: List[str]):
        self.domains = domains
        nltk.download('wordnet')
        from nltk.corpus import wordnet
        self.wordnet = wordnet
        
    def synonym_replacement(self, text: str, n_words: int = 3) -> str:
        """Replace random words with synonyms"""
        words = text.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if len(word) > 3]))
        n = min(n_words, len(random_word_list))
        
        for _ in range(n):
            random_word = random.choice(random_word_list)
            synsets = self.wordnet.synsets(random_word)
            if synsets:
                synonym = random.choice(list(set([lemma.name() for synset in synsets for lemma in synset.lemmas()])))
                random_idx = random.randint(0, len(new_words)-1)
                new_words[random_idx] = synonym
                
        return ' '.join(new_words)
    
    def generate_domain_variations(self, example: Dict) -> List[Dict]:
        """Generate domain-specific variations of examples"""
        variations = [example]
        
        # Add domain-specific variations
        if example['domain'] == 'ethical_dilemmas':
            variations.extend(self._generate_ethical_variations(example))
        elif example['domain'] == 'product_recommendation':
            variations.extend(self._generate_product_variations(example))
            
        return variations
    
    def _generate_ethical_variations(self, example: Dict) -> List[Dict]:
        # Generate variations specific to ethical dilemmas
        variations = []
        base_query = example['query']
        
        # Add stakeholder perspective variation
        stakeholders = ["individual", "community", "society", "future generations"]
        for stakeholder in stakeholders:
            new_query = f"From the perspective of {stakeholder}, {base_query}"
            variations.append({
                **example,
                'query': new_query,
                'augmentation_type': 'stakeholder_perspective'
            })
            
        return variations
    
    def _generate_product_variations(self, example: Dict) -> List[Dict]:
        # Generate variations specific to product recommendations
        variations = []
        base_query = example['query']
        
        # Add budget variations
        budgets = ["limited budget", "moderate budget", "high budget"]
        for budget in budgets:
            new_query = f"With a {budget}, {base_query}"
            variations.append({
                **example,
                'query': new_query,
                'augmentation_type': 'budget_consideration'
            })
            
        return variations

class DomainMetrics:
    """Handles domain-specific evaluation metrics"""
    
    def __init__(self):
        self.domain_metrics = {
            'ethical_dilemmas': self._evaluate_ethical_dilemma,
            'product_recommendation': self._evaluate_product_recommendation,
            'career_advice': self._evaluate_career_advice,
            'argument_generation': self._evaluate_argument,
            'policy_analysis': self._evaluate_policy
        }
        
    def calculate_domain_metrics(self, 
                               prediction: str, 
                               reference: str, 
                               domain: str) -> Dict[str, float]:
        """Calculate domain-specific metrics"""
        if domain in self.domain_metrics:
            return self.domain_metrics[domain](prediction, reference)
        return {}
    
    def _evaluate_ethical_dilemma(self, pred: str, ref: str) -> Dict[str, float]:
        metrics = {}
        
        # Check for stakeholder consideration
        stakeholder_terms = ['stakeholder', 'community', 'impact', 'affect']
        metrics['stakeholder_score'] = sum(term in pred.lower() for term in stakeholder_terms) / len(stakeholder_terms)
        
        # Check for ethical framework usage
        framework_terms = ['principle', 'value', 'moral', 'ethical', 'justice']
        metrics['framework_score'] = sum(term in pred.lower() for term in framework_terms) / len(framework_terms)
        
        return metrics
    
    def _evaluate_product_recommendation(self, pred: str, ref: str) -> Dict[str, float]:
        metrics = {}
        
        # Check for feature comparison
        feature_terms = ['feature', 'specification', 'comparison', 'versus', 'better']
        metrics['feature_score'] = sum(term in pred.lower() for term in feature_terms) / len(feature_terms)
        
        # Check for value consideration
        value_terms = ['price', 'cost', 'value', 'worth', 'budget']
        metrics['value_score'] = sum(term in pred.lower() for term in value_terms) / len(value_terms)
        
        return metrics
        
    def _evaluate_career_advice(self, pred: str, ref: str) -> Dict[str, float]:
        metrics = {}
        
        # Check for market awareness
        market_terms = ['market', 'industry', 'demand', 'growth', 'opportunity']
        metrics['market_awareness'] = sum(term in pred.lower() for term in market_terms) / len(market_terms)
        
        # Check for skill consideration
        skill_terms = ['skill', 'experience', 'qualification', 'requirement', 'competency']
        metrics['skill_consideration'] = sum(term in pred.lower() for term in skill_terms) / len(skill_terms)
        
        return metrics
        
    def _evaluate_argument(self, pred: str, ref: str) -> Dict[str, float]:
        metrics = {}
        
        # Check for logical structure
        logic_terms = ['therefore', 'because', 'consequently', 'thus', 'hence']
        metrics['logic_score'] = sum(term in pred.lower() for term in logic_terms) / len(logic_terms)
        
        # Check for evidence usage
        evidence_terms = ['evidence', 'study', 'research', 'data', 'shows']
        metrics['evidence_score'] = sum(term in pred.lower() for term in evidence_terms) / len(evidence_terms)
        
        return metrics
        
    def _evaluate_policy(self, pred: str, ref: str) -> Dict[str, float]:
        metrics = {}
        
        # Check for policy analysis components
        policy_terms = ['implementation', 'effect', 'outcome', 'impact', 'effectiveness']
        metrics['policy_analysis'] = sum(term in pred.lower() for term in policy_terms) / len(policy_terms)
        
        # Check for stakeholder consideration
        stakeholder_terms = ['public', 'community', 'citizen', 'affected', 'constituency']
        metrics['stakeholder_consideration'] = sum(term in pred.lower() for term in stakeholder_terms) / len(stakeholder_terms)
        
        return metrics

class EnhancedModelCheckpointer:
    """Enhanced model checkpointing with versioning and metadata"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.checkpoints_file = os.path.join(base_path, 'checkpoints.json')
        self.create_base_directory()
        
    def create_base_directory(self):
        os.makedirs(self.base_path, exist_ok=True)
        if not os.path.exists(self.checkpoints_file):
            self._save_checkpoints_info([])
            
    def save_checkpoint(self, 
                       model: AutoModelForCausalLM,
                       tokenizer: AutoTokenizer,
                       metrics: Dict[str, float],
                       step: int,
                       save_format: str = 'complete') -> str:
        """
        Save model checkpoint with enhanced options
        save_format: 'complete' or 'efficient' (saves in safetensors format)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = os.path.join(self.base_path, f'checkpoint_{step}_{timestamp}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        if save_format == 'efficient':
            # Save in safetensors format
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
        else:
            # Save complete model
            model.save_pretrained(checkpoint_dir)
            
        # Save tokenizer
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save configuration and metrics
        metadata = {
            'step': step,
            'timestamp': timestamp,
            'metrics': metrics,
            'path': checkpoint_dir,
            'format': save_format,
            'model_size': self._get_model_size(checkpoint_dir),
            'hardware_info': self._get_hardware_info()
        }
        
        self._update_checkpoints_info(metadata)
        return checkpoint_dir
        
    def load_best_checkpoint(self, metric: str = 'eval_loss') -> str:
        """Load the best checkpoint based on a specific metric"""
        checkpoints = self._load_checkpoints_info()
        if not checkpoints:
            raise ValueError("No checkpoints found")
            
        best_checkpoint = min(checkpoints, 
                            key=lambda x: x['metrics'].get(metric, float('inf')))
        return best_checkpoint['path']
        
    def cleanup_old_checkpoints(self, keep_top_k: int = 3, metric: str = 'eval_loss'):
        """Remove old checkpoints keeping only the top K based on metric"""
        checkpoints = self._load_checkpoints_info()
        if len(checkpoints) <= keep_top_k:
            return
            
        # Sort checkpoints by metric
        sorted_checkpoints = sorted(
            checkpoints,
            key=lambda x: x['metrics'].get(metric, float('inf'))
        )
        
        # Remove excess checkpoints
        for checkpoint in sorted_checkpoints[keep_top_k:]:
            path = checkpoint['path']
            if os.path.exists(path):
                shutil.rmtree(path)
                
        # Update checkpoints info
        self._save_checkpoints_info(sorted_checkpoints[:keep_top_k])
        
    def _get_model_size(self, path: str) -> int:
        """Calculate the size of the saved model in bytes"""
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size
        
    def _get_hardware_info(self) -> Dict[str, str]:
        """Get hardware information"""
        info = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
        return info

class FinalExplanationPipeline:
    def __init__(
        self,
        openai_api_key: str,
        llama_model_path: str,
        training_data_path: str = "data.json",
        output_dir: str = "explanation_model",
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        
        # Initialize components
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            llama_model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        
        # Initialize enhanced components
        self.preprocessor = DataPreprocessor(self.tokenizer)
        self.evaluator = ExplanationEvaluator(self.tokenizer)
        self.checkpointer = EnhancedModelCheckpointer(output_dir)
        self.domain_metrics = DomainMetrics()
        self.augmentor = DataAugmentor(self._get_domains())
        
        self.training_data_path = training_data_path
        
    def _get_domains(self) -> List[str]:
        """Get list of domains from training data"""
        with open(self.training_data_path, 'r') as f:
            data = json.load(f)
        return list(set(item['domain'] for item in data))
        
    def fine_tune(self, 
                  validation_split: float = 0.1,
                  save_format: str = 'complete',
                  augment_data: bool = True):
        """Enhanced fine-tuning with all features"""
        
        # Preprocess and augment data
        processed_data = self.preprocessor.preprocess_dataset(
            self.training_data_path,
            save_path=f"{self.output_dir}/processed_data.json"
        )
        
        if augment_data:
            augmented_data = []
            for example in processed_data:
                augmented_data.extend(
                    self.augmentor.generate_domain_variations(example)
                )
            processed_data = augmented_data
        
        # Split data
        train_data, val_data = train_test_split(
            processed_data,
            test_size=validation_split,
            random_state=42
        )
        
        # Create datasets
        train_dataset = ExplanationDataset(train_data, self.tokenizer)
        val_dataset = ExplanationDataset(val_data, self.tokenizer)
        
        # Enhanced training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=500,
            learning_rate=2e-5,
            fp16=True if self.device == 'cuda' else False,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            evaluation_strategy=IntervalStrategy.STEPS
            eval_steps=100,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            weight_decay=0.01,
            report_to='tensorboard'
        )
        
        def compute_metrics(eval_pred):
            """Enhanced metric computation with domain-specific metrics"""
            predictions, labels = eval_pred
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Calculate general metrics
            metrics = {}
            for pred, label, example in zip(decoded_preds, decoded_labels, val_data):
                # Standard metrics
                sample_metrics = self.evaluator.calculate_metrics(pred, label)
                
                # Domain-specific metrics
                domain_metrics = self.domain_metrics.calculate_domain_metrics(
                    pred, label, example['domain']
                )
                
                # Combine metrics
                all_metrics = {**sample_metrics, **domain_metrics}
                for key, value in all_metrics.items():
                    metrics[key] = metrics.get(key, 0) + value
                    
            # Average metrics
            for key in metrics:
                metrics[key] /= len(decoded_preds)
                
            return metrics
        
        # Initialize trainer with enhanced features
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Training with enhanced progress tracking
        try:
            trainer.train()
            
            # Save final model with enhanced checkpointing
            final_metrics = trainer.evaluate()
            checkpoint_path = self.checkpointer.save_checkpoint(
                self.model,
                self.tokenizer,
                final_metrics,
                step='final',
                save_format=save_format
            )
            
            # Cleanup old checkpoints
            self.checkpointer.cleanup_old_checkpoints(keep_top_k=3)
            
            # Save training summary
            self._save_training_summary(
                final_metrics,
                checkpoint_path,
                len(train_data),
                len(val_data)
            )
            
            return checkpoint_path
            
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            # Save emergency checkpoint if possible
            try:
                emergency_path = self.checkpointer.save_checkpoint(
                    self.model,
                    self.tokenizer,
                    {"error": str(e)},
                    step='emergency',
                    save_format='complete'
                )
                logging.info(f"Emergency checkpoint saved at: {emergency_path}")
            except:
                logging.error("Failed to save emergency checkpoint")
            raise
            
    def _save_training_summary(self,
                             metrics: Dict[str, float],
                             checkpoint_path: str,
                             train_size: int,
                             val_size: int):
        """Save comprehensive training summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'checkpoint_path': checkpoint_path,
            'dataset_stats': {
                'train_size': train_size,
                'validation_size': val_size,
                'total_size': train_size + val_size
            },
            'model_info': {
                'base_model': 'Llama-2-7b',
                'device': self.device,
                'checkpoint_format': 'complete'
            },
            'hardware_info': self.checkpointer._get_hardware_info()
        }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    """Main function to run the pipeline"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Initialize pipeline
    pipeline = FinalExplanationPipeline(
        openai_api_key=openai_api_key,
        llama_model_path="meta-llama/Llama-2-7b",
        training_data_path="data.json",
        output_dir="final_explanation_model"
    )
    
    try:
        # Run fine-tuning with all enhancements
        final_checkpoint = pipeline.fine_tune(
            validation_split=0.1,
            save_format='complete',
            augment_data=True
        )
        
        logging.info(f"Training complete. Final checkpoint saved at: {final_checkpoint}")
        
        # Load and verify the saved model
        loaded_model = AutoModelForCausalLM.from_pretrained(final_checkpoint)
        loaded_tokenizer = AutoTokenizer.from_pretrained(final_checkpoint)
        
        logging.info("Successfully loaded and verified saved model")
        
    except Exception as e:
        logging.error(f"Pipeline error: {str(e)}")
        raise

if __name__ == "__main__":
    main()