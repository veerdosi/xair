import torch
import torch.nn as nn
from .explainable_transformer import ExplainableTransformerLayer
from .explanation_decoder import ExplanationDecoder

class ExplainableLLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(ExplainableLLM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.transformer_layers = nn.ModuleList([
            ExplainableTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.explanation_decoder = ExplanationDecoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        attention_weights = []
        attention_patterns = []
        
        for layer in self.transformer_layers:
            src, attn_weights, attn_patterns = layer(src, src_mask, src_key_padding_mask)
            attention_weights.append(attn_weights)
            attention_patterns.append(attn_patterns)
        
        output = self.output_projection(src)
        
        if tgt is not None:
            explanation = self.explanation_decoder(tgt, src, tgt_mask, None, tgt_key_padding_mask, src_key_padding_mask)
        else:
            explanation = None
        
        return output, explanation, attention_weights, attention_patterns

    def generate(self, src, max_length=50, temperature=1.0):
        self.eval()
        device = src.device
        
        with torch.no_grad():
            src_mask = self.generate_square_subsequent_mask(src.size(0)).to(device)
            output, _, attention_weights, attention_patterns = self.forward(src, src_mask=src_mask)
            
            # Generate primary output
            primary_output = self._generate_sequence(output, max_length, temperature)
            
            # Generate explanation
            explanation = self.explanation_decoder.generate_explanation(output, max_length, temperature)
        
        return primary_output, explanation, attention_weights, attention_patterns

    def _generate_sequence(self, output, max_length, temperature):
        device = output.device
        batch_size = output.size(1)
        
        generated_sequence = []
        
        for _ in range(max_length):
            next_token_logits = output[-1, :, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            generated_sequence.append(next_token.unsqueeze(0))
            
            if next_token.item() == self.eos_token_id:
                break
            
            next_input = next_token.unsqueeze(0)
            next_input_emb = self.embedding(next_input) * math.sqrt(self.d_model)
            next_input_emb = self.pos_encoder(next_input_emb)
            
            for layer in self.transformer_layers:
                next_input_emb, _, _ = layer(next_input_emb)
            
            output = torch.cat([output, self.output_projection(next_input_emb)], dim=0)
        
        return torch.cat(generated_sequence, dim=0)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
