"""
Transformer Decoder-only base model for text generation
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class Attention(nn.Module):
    def __init__(self, d_model, head_size, context_length, dropout):
        super().__init__()
        self.head_size = head_size
        self.Wq = nn.Linear(d_model, head_size, bias=False)
        self.Wk = nn.Linear(d_model, head_size, bias=False)
        self.Wv = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, head_size, context_length, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            Attention(d_model, head_size, context_length, dropout)
            for _ in range(num_heads)
        ])
        self.projection_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection_layer(head_outputs))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, head_size, context_length, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, head_size, context_length, dropout)
        self.ffn = FeedForwardNetwork(d_model, dropout)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Model(nn.Module):
    def __init__(self, h_params):
        super().__init__()
        self.context_length = h_params['context_length']
        self.d_model = h_params['d_model']
        self.num_blocks = h_params['num_blocks']
        self.num_heads = h_params['num_heads']
        self.head_size = self.d_model // self.num_heads
        self.dropout = h_params['dropout']
        self.device = h_params['device']
        self.max_token_value = h_params['max_token_value']

        self.token_embedding_lookup_table = nn.Embedding(self.max_token_value, self.d_model)
        self.transformer_blocks = nn.Sequential(*(
            [TransformerBlock(self.d_model, self.num_heads, self.head_size,
                              self.context_length, self.dropout)
             for _ in range(self.num_blocks)] +
            [nn.LayerNorm(self.d_model)]
        ))
        self.model_out_linear_layer = nn.Linear(self.d_model, self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Positional Encoding
        position_encoding = torch.zeros(self.context_length, self.d_model, device=self.device)
        position = torch.arange(0, self.context_length, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=self.device) * (-math.log(10000.0) / self.d_model))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)

        x = self.token_embedding_lookup_table(idx) + position_encoding[:T, :].to(self.device)
        x = self.transformer_blocks(x)
        logits = self.model_out_linear_layer(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.context_length:]
            logits, _ = self.forward(idx_crop)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx