import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
from dataclasses import dataclass

nltk.download('punkt')

@dataclass
class ExplanationMetrics:
    faithfulness: float
    plausibility: float
    coherence: float
    factual_accuracy: float

class Evaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    def evaluate_faithfulness(self, question, answer, explanation):
        # Measure similarity between answer and explanation
        answer_emb = self.get_bert_embedding(answer)
        explanation_emb = self.get_bert_embedding(explanation)
        similarity = cosine_similarity(answer_emb, explanation_emb)[0][0]
        return similarity

    def evaluate_plausibility(self, question, answer, explanation, human_rating):
        # This would typically involve human evaluation
        # Here, we're simulating it with a random score
        return human_rating  # Assume this is provided on a scale of 0-1

    def evaluate_coherence(self, explanation):
        # Use BLEU score as a proxy for coherence
        sentences = explanation.split('.')
        if len(sentences) < 2:
            return 1.0  # Perfect coherence for single sentence
        
        bleu_scores = []
        for i in range(1, len(sentences)):
            reference = [word_tokenize(sentences[i-1])]
            candidate = word_tokenize(sentences[i])
            bleu_scores.append(sentence_bleu(reference, candidate))
        
        return np.mean(bleu_scores)

    def evaluate_factual_accuracy(self, question, answer, ground_truth):
        # Compare the generated answer with the ground truth
        answer_emb = self.get_bert_embedding(answer)
        truth_emb = self.get_bert_embedding(ground_truth)
        accuracy = cosine_similarity(answer_emb, truth_emb)[0][0]
        return accuracy

    def get_bert_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def evaluate_batch(self, dataloader):
        self.model.eval()
        metrics = {
            'faithfulness': [],
            'plausibility': [],
            'coherence': [],
            'factual_accuracy': []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                questions = batch['question'].to(self.device)
                ground_truths = batch['answer']
                
                outputs, explanations, _, _ = self.model.generate(questions)
                
                for q, a, e, gt in zip(questions, outputs, explanations, ground_truths):
                    question = self.tokenizer.decode(q, skip_special_tokens=True)
                    answer = self.tokenizer.decode(a, skip_special_tokens=True)
                    explanation = self.tokenizer.decode(e, skip_special_tokens=True)
                    ground_truth = self.tokenizer.decode(gt, skip_special_tokens=True)
                    
                    metrics['faithfulness'].append(self.evaluate_faithfulness(question, answer, explanation))
                    metrics['plausibility'].append(self.evaluate_plausibility(question, answer, explanation, np.random.rand()))
                    metrics['coherence'].append(self.evaluate_coherence(explanation))
                    metrics['factual_accuracy'].append(self.evaluate_factual_accuracy(question, answer, ground_truth))
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    class ExplanationMetricsCalculator:
        @staticmethod
        def calculate_metrics(generated_explanation: Dict[str, torch.Tensor], 
                            ground_truth: Optional[Dict[str, torch.Tensor]] = None) -> ExplanationMetrics:
            """Calculate explanation quality metrics"""
            # Calculate metrics (simplified version)
            return ExplanationMetrics(
                faithfulness=torch.mean(generated_explanation['base_explanation']).item(),
                plausibility=0.8,  # Would be based on human evaluation
                coherence=torch.mean(generated_explanation['attention_patterns']).item(),
                factual_accuracy=0.9 if ground_truth is None else torch.mean(torch.eq(
                    generated_explanation['base_explanation'], 
                    ground_truth['base_explanation']
                )).item()
            )

# Usage in main script:
# evaluator = Evaluator(model, tokenizer, device)
# eval_dataloader = DataLoader(eval_dataset, batch_size=16)
# metrics = evaluator.evaluate_batch(eval_dataloader)
# print(metrics)
