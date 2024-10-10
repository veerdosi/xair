import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class ExplanationDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(ExplanationDecoder, self).__init__()
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.embedding = nn.Embedding(vocab_size, d_model)  # We'll need to define vocab_size
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        
        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask,
                                          tgt_key_padding_mask, memory_key_padding_mask)
        
        return self.output_projection(output)

    def generate_explanation(self, memory, max_length=50, temperature=1.0):
        """
        Generate an explanation given the encoder's output (memory).
        
        Args:
            memory: the output from the encoder.
            max_length: maximum length of the generated explanation.
            temperature: sampling temperature.
        """
        device = memory.device
        batch_size = memory.size(1)
        
        # Start with BOS token
        input_seq = torch.full((1, batch_size), self.bos_token_id, device=device)
        
        for _ in range(max_length):
            tgt_mask = self.generate_square_subsequent_mask(input_seq.size(0)).to(device)
            
            out = self.forward(input_seq, memory, tgt_mask=tgt_mask)
            next_token_logits = out[-1, :, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            
            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=0)
            
            if next_token.item() == self.eos_token_id:
                break
        
        return input_seq

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
