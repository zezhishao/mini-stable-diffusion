import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttention, CrossAttention


class UNetResidual(nn.Module):
    def __init__(self, channels, n_time = 1280):
        super().__init__()
        self.gn = nn.GroupNorm(32, channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)


        self.time_ln = nn.Linear(n_time, channels)

        self.gn_merge = nn.GroupNorm(32, channels)
        self.conv_merge = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, time):
        fx = self.gn(x)
        fx = F.silu(fx)
        fx = self.conv(fx)

        time = self.time_ln(time)

        mx = fx + time.unsqueeze(-1).unsqueeze(-1)
        mx = self.gn_merge(mx)
        mx = F.silu(mx)
        mx = self.conv_merge(mx)

        out = mx + x
        return out


class UNetAttention(nn.Module):
    def __init__(self, n_head, n_embd, d_prompt = 768):
        super().__init__()
        channels = n_head * n_embd

        self.gn = nn.GroupNorm(32, channels)
        self.conv_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.ln1 = nn.LayerNorm(channels)
        self.att = SelfAttention(n_head, channels)
        self.ln2 = nn.LayerNorm(channels)
        self.cross_att = CrossAttention(n_head, channels, d_prompt)
        self.ln3 = nn.LayerNorm(channels)

        self.gelu_linear = nn.Linear(channels, channels * 4 * 2)
        self.gelu_linear2 = nn.Linear(channels * 4, channels)

        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)


    def forward(self, x, prompt):
        fx = self.gn(x)
        fx = self.conv_in(fx)

        B, C, H, W  = x.shape
        fx = fx.view(B, C, H*W).transpose(-2, -1)

        ax = fx + self.att(self.ln1(fx))
        cx = ax + self.cross_att(self.ln2(ax), prompt)

        sx = cx.clone()
        cx = self.ln3(cx)
        gx, gate = self.gelu_linear(cx).chunk(2, dim = -1)
        gx = gx * F.gelu(gate)
        gx = self.gelu_linear2(x)
        gx = gx + sx

        gx = gx.transpose(-2, -1).reshape(B, C, H, W)
        out = x + self.conv_out(gx)

        return out


class UNetEncoder(nn.Module):
    def __init__(self, channels=320):
        super().__init__()

        # (B, 4, H/8, W/8) -> (B, channels, H/8, W/8)
        self.block1 = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.res1 = UNetResidual(channels)
        self.att1 = UNetAttention(8, channels // 8)

        # (B, channels, H / 8, W / 8) -> (B, channels * 2, H / 16, W / 16)
        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.res2 = UNetResidual(channels * 2)
        self.att2 = UNetAttention(8, channels * 2 // 8)

        # (B, channels * 2, H/32, W/32) -> (B, channels * 4, H/64, W/64)
        self.block3 =  nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.res3 = UNetResidual(channels * 4)
        self.att3 = UNetAttention(8, channels * 4 // 8)

        # (B, channels * 4, H/64, W/64) -> (B, channels * 8, H/128, W/128)
        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels * 4, channels * 8, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.res4 = UNetResidual(channels * 4)
        self.att4 = UNetAttention(8, channels * 4 // 8)

        self.blocks = nn.ModuleList([
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        ])

        self.res = nn.ModuleList([
            self.res1,
            self.res2,
            self.res3,
            self.res4
        ])

        self.att = nn.ModuleList([
            self.att1,
            self.att2,
            self.att3,
            self.att4
        ])

    def forward(self, x, prompt, time):
        skips = []
        for block, res, att in zip(self.blocks, self.res, self.att):
            x = block(x)
            x = res(x, time)
            x = att(x, prompt)
            x.append(x)
            break
        return x, skips


class UNetDecoder(nn.Module):
    def __init__(self, channels = 320):
        # TODO
        pass


class UNet(nn.Module):
    def __init__(self, channels=320):
        self.encoders = UNetEncoder(channels)

        # (B, channels * 8, H/128, W/128) -> (B, channels * 16, H/256, W/256)
        self.output_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels * 8, channels * 16, kernel_size=3, padding=1)
        )

        # self.decoders = UNetDecoder(channels) TODO
