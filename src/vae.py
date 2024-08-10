import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttention


class VAE_Residual(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.gn1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.res_layer = nn.Identity()
        else:
            self.res_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, in, H, W) -> x:(B, out, H, W)

        resd_x = self.res_layer(x)

        x = self.gn1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.gn(x)
        x = F.silu(x)
        x = self.conv2(x)

        x = x + resd_x

        return x


class VAE_Attention(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(32, channels)
        self.att = SelfAttention(1, channels)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        resd = x

        x = self.gn(x)
        x = x.view(B, C, H * W).transpose(-2, -1) # (B, H * W, C)
        x = self.att(x)
        x = x.transpose(-2, -1).view(B, C, H, W)
        x = x + resd

        return x


class VAE_Encoder(nn.Module):
    def __init__(self, channels = 128) -> None:
        super().__init__()

        self.net = nn.Sequential(

        # (B, 3[R, G, B], H, W) -> (B, channels, H, W)
        nn.Conv2d(3, channels, kernel_size=3, padding=1),
        VAE_Residual(channels, channels), # same dimensions

        # (B, channels, H, W) -> (B, channels, H/2, W / 2)
        nn.Conv2d(channels, channels, kernel_size=3, stride=2),
        # (B, channels, H/2, W/2) -> (B, channels * 2, H/2, W/2)
        VAE_Residual(channels, channels * 2), # increase by 2x

        # (B, channels * 2, H/2, W/2) -> (B, channels * 2, H/4, W / 4)
        nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=2),
        # (B, channels * 2, H/4, W/4) -> (B, channels * 4, H/4, W/4)
        VAE_Residual(channels * 2, channels * 4), # increase by 4x

        # (B, channels * 4, H/4, W/4) -> (B, channels * 4, H/8, W / 8)
        nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=2),
        # (B, channels * 4, H/8, W/8) -> (B, channels * 4, H/8, W/8)
        VAE_Residual(channels * 4, channels * 4), # increase by 4x

        VAE_Attention(channels * 4),
        # (B, channels * 4, H/8, W/8) -> (B, channels * 4, H/8, W/8)
        VAE_Residual(channels * 4, channels * 4), # increase by 4x

        nn.GroupNorm(32, channels*4),
        nn.SiLU(),

        # bottleneck
        # (B, channels * 4, H/8, W/8) -> (B, 8, H/8, W/8)
        nn.Conv2d(channels * 4, 8, kernel_size=3, padding=1),

        # (B, 8, H/8, W//8) -> (B, 8, H/8, W/8)
        nn.Conv2d(8, 8, kernel_size=1),
    )

    def forward(self, x: torch.Tensor, noise:torch.Tensor):
        # x: B, C, H, W
        # noise: N(0, I)
        for layer in self.net:
            if getattr(layer, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1)) # apply pad at bottom and right, if stride is 2
            x = layer(x)

        # (B, 8, H/8, W/8) -> 2 tensors(mean and log_var) of (B, 4, H/8, W/8)
        mean, log_var = torch.chunk(x, 2, dim=1)

        log_var = torch.clamp(log_var, -30, 20)
        var = log_var.exp()
        std = var.sqrt()

        x = mean + std * noise

        # Scaling constant
        x *= 0.18215

        # output shape of x: (B, 4, H/8, H/8) i.e (32, 4, 64, 64)
        return x


class VAE_Decoder(nn.Module):
    def __init__(self, channels = 128) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1),
            # (B, 4, H/8, W/8) -> (B, channels*4, H/8, W/8)
            nn.Conv2d(4, channels * 4, kernel_size=3, padding=1),
            VAE_Residual(channels * 4, channels * 4),

            # Same shapes throughout
            VAE_Attention(channels * 4),
            VAE_Residual(channels * 4, channels * 4),

            # (B, channels*4, H/8, W/8) -> (B, channels*4, H/4, W/4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
            VAE_Residual(channels * 4, channels * 4),

            # (B, channels * 4, H/4, W/4) -> (B, channels * 2, H/2, W/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1),
            VAE_Residual(channels * 4, channels * 2),

            # (B, channels * 2, H/2, W/2) -> (B, channels, H, W)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1),
            VAE_Residual(channels * 2, channels),

            nn.GroupNorm(32, channels),
            nn.SiLU(),

            # (B, channels, H, W) -> (B, 3[R, G, B], H, W)
            nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: (B, 4, H/8, W/8)

        # Descaling Factor
        x = (1. / 0.18215) * x

        # Forward Pass
        x = self.net(x)
        return x
