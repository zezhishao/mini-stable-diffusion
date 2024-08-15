import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Config


config = Config()


class SelfAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = config.DROPOUT):
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
        B, T, C = x.shape

        x = x.view(B, T, self.heads, self.d_head)

        q = self.q(x).transpose(1, 2)
        k = self.k(x).transpose(1, 2)
        v = self.v(x).transpose(1, 2)

        att = q @ k.transpose(-1, -2) / torch.sqrt(torch.tensor(C).to(x.device))
        if mask:
            mask = torch.ones_like(att, dtype = torch.bool).triu(1)
            att.masked_fill(mask, -torch.inf)

        att = F.softmax(att, dim=-1)
        out = att @ v

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class CrossAttention(nn.Module):
    def __init__(self, heads, d_model, d_prompt, dropout = 0.2):
        super().__init__()
        d_head = d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_prompt, d_model)
        self.v = nn.Linear(d_prompt, d_model)

        self.proj = nn.Linear(d_model, d_model)

        self.d_head = d_head
        self.heads = heads

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prompt):
        Bq, Tq, Cq = x.shape
        Bkv, Tkv, Ckv = prompt.shape

        q = self.q(x).transpose(1, 2)
        k = self.k(prompt).transpose(1, 2)
        v = self.v(prompt).transpose(1, 2)

        shape = (Bq, -1, self.heads, self.d_head)
        q = q.reshape(shape).transpose(1, 2)
        k = k.reshape(shape).transpose(1, 2)
        v = v.reshape(shape).transpose(1, 2)

        att = q @ k.transpose(-1, -2) / torch.sqrt(torch.tensor(Ckv).to(x.device))
        att = F.softmax(att, dim=-1)
        out = att @ v

        out = out.transpose(1, 2).reshape(Bq, Tq, Cq)
        out = self.proj(out)
        out = self.dropout(out)

        return out
