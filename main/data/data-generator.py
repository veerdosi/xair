import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import random

class DataGenerator:
    def __init__(self, model_name="gpt2-large", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_dataset(self, num_samples, max_length=100):
        dataset = []
        for _ in tqdm(range(num_samples), desc="Generating dataset"):
            topic = self.select_random_topic()
            question_type = self.select_random_question_type()
            question = self.generate_question(topic, question_type)
            answer, explanation = self.generate_answer_and_explanation(question)
            dataset.append({
                "topic": topic,
                "question_type": question_type,
                "question": question,
                "answer": answer,
                "explanation": explanation
            })
        return dataset

    def select_random_topic(self):
        topics = ["Science", "History", "Geography", "Literature", "Arts", "Sports", "Technology", "Current Events"]
        return random.choice(topics)

    def select_random_question_type(self):
        question_types = ["Factual", "Analytical", "Comparative", "Causal", "Hypothetical"]
        return random.choice(question_types)

    def generate_question(self, topic, question_type):
        prompt = f"Generate a {question_type} question about {topic}:\n"
        question = self.generate_text(prompt, max_length=50)
        return question.strip()

    def generate_answer_and_explanation(self, question):
        prompt = f"Q: {question}\nA:"
        answer = self.generate_text(prompt, max_length=100)
        
        explanation_prompt = f"Q: {question}\nA: {answer}\nExplanation:"
        explanation = self.generate_text(explanation_prompt, max_length=200)
        
        return answer.strip(), explanation.strip()

    def generate_text(self, prompt, max_length):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()

    def save_dataset(self, dataset, filename):
        torch.save(dataset, filename)

    @staticmethod
    def load_dataset(filename):
        return torch.load(filename)

# Usage in main script:
# data_generator = DataGenerator()
# dataset = data_generator.generate_dataset(num_samples=1000)
# data_generator.save_dataset(dataset, "diverse_qa_dataset.pth")
