import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.2):
        super().__init__()
        d_head = d_model // heads
        self.q = nn.Linear(d_head, d_head)
        self.k = nn.Linear(d_head, d_head)
        self.v = nn.Linear(d_head, d_head)

        self.proj = nn.Linear(d_model, d_model)

        self.d_head = d_head
        self.heads = heads

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = False):
        B, T, C = x.size()

        x = x.view(B, T, self.heads, self.d_head)

        q = self.q(x).transpose(1, 2)
        k = self.k(x).transpose(1, 2)
        v = self.v(x).transpose(1, 2)

        att = q @ k.transpose(-1, -2) / torch.sqrt(C)
        if mask:
            mask = torch.ones_like(att, dtype = torch.bool).triu(1)
            att.masked_fill(mask, -torch.inf)

        att = F.softmax(att, dim=-1)
        out = att @ v

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out
