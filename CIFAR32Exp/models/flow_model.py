import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -scale)
        angles = t[:, None] * freqs[None]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        cond_bias = self.cond_proj(F.silu(cond))[:, :, None, None]
        h = h + cond_bias
        h = self.conv2(F.silu(self.norm2(h)))
        return h + residual


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        cond_dim: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.time_embed = SinusoidalTimeEmbedding(cond_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.label_embed = nn.Linear(num_classes, cond_dim)

        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, cond_dim)
        self.downsample1 = Downsample(base_channels)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, cond_dim)
        self.downsample2 = Downsample(base_channels * 2)
        self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 2, cond_dim)

        self.upsample1 = Upsample(base_channels * 2)
        self.up1 = ResidualBlock(base_channels * 4, base_channels * 2, cond_dim)
        self.upsample2 = Upsample(base_channels * 2)
        self.up2 = ResidualBlock(base_channels * 3, base_channels, cond_dim)

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def conditioning(self, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        time_cond = self.time_mlp(self.time_embed(t))
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        label_cond = self.label_embed(one_hot)
        return time_cond + label_cond

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.conditioning(t, labels)

        x0 = self.input_proj(x)
        d1 = self.down1(x0, cond)
        d2 = self.down2(self.downsample1(d1), cond)
        mid = self.bottleneck(self.downsample2(d2), cond)

        u1 = self.upsample1(mid)
        u1 = self.up1(torch.cat([u1, d2], dim=1), cond)
        u2 = self.upsample2(u1)
        u2 = self.up2(torch.cat([u2, d1], dim=1), cond)

        return self.out_conv(F.silu(self.out_norm(u2)))


def build_flow_model(config):
    model_cfg = config.get("model", {})
    return ConditionalUNet(
        in_channels=model_cfg.get("in_channels", 3),
        base_channels=model_cfg.get("base_channels", 64),
        cond_dim=model_cfg.get("cond_dim", 128),
        num_classes=model_cfg.get("num_classes", 10),
    )
