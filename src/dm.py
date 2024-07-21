import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionModel(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.SiLU(),
            nn.Linear(n_embd * 4, n_embd * 4),
            nn.SiLU()
        )
