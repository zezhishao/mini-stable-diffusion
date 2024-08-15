import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNet
from .vae import VAE_Encoder, VAE_Decoder


class DiffusionModel(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.SiLU(),
            nn.Linear(n_embd * 4, n_embd * 4),
            nn.SiLU()
        )
        self.unet = UNet()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()

    def forward(self, x, noise, prompt, time):
        time = self.time_emb(time)

        x = self.encoder(x, noise)
        x = self.unet(x, prompt, time)
        out = self.decoder(x)
        
        return out
