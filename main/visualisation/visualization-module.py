import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def plot_attention(self, attention_weights, tokens, layer=0, head=0):
        """
        Plot attention weights for a specific layer and head.
        """
        att_matrix = attention_weights[layer][head].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(att_matrix, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
        plt.title(f'Attention weights (Layer {layer}, Head {head})')
        plt.tight_layout()
        plt.show()

    def plot_attention_patterns(self, attention_patterns, tokens, aspect=0):
        """
        Plot attention patterns for a specific aspect.
        """
        att_pattern = attention_patterns[aspect].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(att_pattern, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
        plt.title(f'Attention pattern (Aspect {aspect})')
        plt.tight_layout()
        plt.show()

    def visualize_explanation(self, question, answer, explanation, attention_weights, attention_patterns):
        """
        Visualize the question, answer, explanation, and attention.
        """
        tokens = self.tokenizer.tokenize(question + " " + answer + " " + explanation)
        
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Explanation: {explanation}")
        
        # Plot attention weights for the first layer and head
        self.plot_attention(attention_weights, tokens)
        
        # Plot attention patterns for the first aspect
        self.plot_attention_patterns(attention_patterns, tokens)

# Usage in main script:
# visualizer = Visualizer(tokenizer)
# question = "What is the capital of France?"
# answer, explanation, attention_weights, attention_patterns = model.generate(tokenizer.encode(question, return_tensors='pt'))
# visualizer.visualize_explanation(question, tokenizer.decode(answer[0]), tokenizer.decode(explanation[0]), attention_weights, attention_patterns)
