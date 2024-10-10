import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplainableMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super(ExplainableMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Additional projections for interpretable attention patterns
        self.aspect_projs = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(3)])

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(query.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(key.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(value.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output, attn_weights = self._attention(q, k, v, attn_mask, key_padding_mask)
        attn_patterns = self._compute_interpretable_patterns(query, key, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(query.shape[0], -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, attn_patterns

    def _attention(self, q, k, v, attn_mask=None, key_padding_mask=None):
        d_k = q.size(-1)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

        if attn_mask is not None:
            attn_logits += attn_mask

        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights

    def _compute_interpretable_patterns(self, query, key, value):
        patterns = []
        for proj in self.aspect_projs:
            q_asp = proj(query)
            k_asp = proj(key)
            pattern = torch.matmul(q_asp, k_asp.transpose(-2, -1)) / (self.head_dim ** 0.5)
            patterns.append(F.softmax(pattern, dim=-1))
        return patterns
