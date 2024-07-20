import torch
import torch.nn as nn
import torch.nn.functional as F
from.attention import SelfAttention


class ClipLayer(nn.Module):

    """
    A simple Transformers Decoder
    """

    def __init__(self, n_heads, n_embd, dropout = 0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_heads, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x), mask=True)
        out = x + self.ffn(self.ln2(x))
        return out


class CLIP(nn.Module):

    """
    A simple Decoder only Transformer architecture (GPT like)
    """

    def __init__(self, vocab_size = 49408, n_embd = 768, seq_len = 77, n_clip_layers = 12, n_clip_heads = 12) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.layers = nn.ModuleList(
            [ClipLayer(n_clip_heads, n_embd) for i in range(n_clip_layers)]
        )
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        B, T = x.size()
        x = x.type(torch.long)
        tok_x = self.tok_emb(x)
        pos_x = self.pos_emb(torch.arange(T).to(x.device))

        x = tok_x + pos_x

        for layer in self.layers:
            x = layer(x)

        return x
