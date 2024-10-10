import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import json
from tensorboardX import SummaryWriter

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Trainer:
    def __init__(self, model, tokenizer, device, lr=1e-4, checkpoint_dir='checkpoints', log_dir='logs'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    def save_checkpoint(self, epoch, train_loss, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        return epoch, train_loss, val_loss

    def train(self, train_dataset, val_dataset, batch_size, num_epochs, checkpoint_interval=5):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                question = batch['question'].to(self.device)
                answer = batch['answer'].to(self.device)
                explanation = batch['explanation'].to(self.device)
                
                self.optimizer.zero_grad()
                
                output, explanation_output, _, _ = self.model(question, explanation[:, :-1])
                
                answer_loss = self.criterion(output.view(-1, output.size(-1)), answer.view(-1))
                explanation_loss = self.criterion(explanation_output.view(-1, explanation_output.size(-1)), explanation[:, 1:].contiguous().view(-1))
                
                loss = answer_loss + explanation_loss
                loss.backward()
                
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_dataloader)
            val_loss = self.evaluate(val_dataloader)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Checkpointing
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1, train_loss, val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, train_loss, val_loss)
                print("Best model saved!")
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break

        self.writer.close()

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                question = batch['question'].to(self.device)
                answer = batch['answer'].to(self.device)
                explanation = batch['explanation'].to(self.device)
                
                output, explanation_output, _, _ = self.model(question, explanation[:, :-1])
                
                answer_loss = self.criterion(output.view(-1, output.size(-1)), answer.view(-1))
                explanation_loss = self.criterion(explanation_output.view(-1, explanation_output.size(-1)), explanation[:, 1:].contiguous().view(-1))
                
                loss = answer_loss + explanation_loss
                total_loss += loss.item()
        
        return total_loss / len(dataloader)

    def generate_qa_with_explanation(self, question, max_length=100):
        self.model.eval()
        with torch.no_grad():
            question_ids = self.tokenizer.encode(question, return_tensors='pt').to(self.device)
            output, explanation, attention_weights, attention_patterns = self.model.generate(question_ids, max_length=max_length)
            
            answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
            explanation = self.tokenizer.decode(explanation[0], skip_special_tokens=True)
            
        return answer, explanation, attention_weights, attention_patterns

# Usage in main script:
# trainer = Trainer(model, tokenizer, device, checkpoint_dir='checkpoints', log_dir='logs')
# trainer.train(train_dataset, val_dataset, batch_size=16, num_epochs=5, checkpoint_interval=1)
