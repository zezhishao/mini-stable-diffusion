import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetEncoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        # (B, 3, H, W) -> (B, channels, H, W)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # (B, channels, H, W) -> (B, channels * 2, H/2, W/2)
        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # (B, channels * 2, H/2, W/2) -> (B, channels * 4, H/4, W/4)
        self.block3 =  nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # (B, channels * 4, H/4, W/4) -> (B, channels * 8, H/8, W/8)
        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels * 4, channels * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels * 8, channels * 8, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # (B, channels * 8, H/8, W/8) -> (B, channels * 16, H/16, W/16)
        self.block5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels * 8, channels * 16, kernel_size=3, padding=1)
        )

        self.blocks = nn.ModuleList([
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5
        ])

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i != 4:
                skips.append(x)
        skips.reverse()
        return x, skips
