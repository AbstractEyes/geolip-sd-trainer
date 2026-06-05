"""
Native SDXL VAE — pure PyTorch AutoencoderKL, no diffusers in the path.
=======================================================================
From-scratch reimplementation of SDXL-base-1.0's VAE, module tree named
key-for-key to the diffusers checkpoint so the real weights load strict=True.
Owned, fully-controllable encode/decode for the trainer (precompute latents +
sample-time decode).

Verified against stabilityai/stable-diffusion-xl-base-1.0/vae (248 tensors):
  block_out_channels [128,256,512,512] · 2 res/encoder-block · 3/decoder-block
  latent 4 (conv_out 8 = mean|logvar) · scaling_factor 0.13025 · groups 32 eps 1e-6
  encoder downsample = asymmetric pad (0,1,0,1) then stride-2 conv pad0 (VAE convention)
  mid block = resnet -> single-head spatial attention (group_norm + biased q/k/v) -> resnet
  force_upcast: run in fp32 (the SDXL fp16 VAE is numerically unstable).

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SDXLVAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: tuple = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    scaling_factor: float = 0.13025


# ----------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------

class VAEResnetBlock(nn.Module):
    """No time embedding (VAE). norm1->silu->conv1->norm2->silu->conv2 + shortcut."""
    def __init__(self, in_ch, out_ch, groups=32, eps=1e-6):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, eps=eps)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch, eps=eps)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class VAEAttention(nn.Module):
    """Single-head spatial self-attention (diffusers VAE). Biased q/k/v/out + group_norm."""
    def __init__(self, ch, groups=32, eps=1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, ch, eps=eps)
        self.to_q = nn.Linear(ch, ch)
        self.to_k = nn.Linear(ch, ch)
        self.to_v = nn.Linear(ch, ch)
        self.to_out = nn.ModuleList([nn.Linear(ch, ch), nn.Dropout(0.0)])

    def forward(self, x):
        res = x
        B, C, H, W = x.shape
        h = self.group_norm(x).view(B, C, H * W).transpose(1, 2)        # [B,HW,C]
        q, k, v = self.to_q(h), self.to_k(h), self.to_v(h)
        q = q.unsqueeze(1); k = k.unsqueeze(1); v = v.unsqueeze(1)      # [B,1,HW,C]
        o = F.scaled_dot_product_attention(q, k, v).squeeze(1)         # [B,HW,C]
        o = self.to_out[1](self.to_out[0](o))
        o = o.transpose(1, 2).view(B, C, H, W)
        return res + o


class VAEMidBlock(nn.Module):
    def __init__(self, ch, groups, eps):
        super().__init__()
        self.attentions = nn.ModuleList([VAEAttention(ch, groups, eps)])
        self.resnets = nn.ModuleList([VAEResnetBlock(ch, ch, groups, eps),
                                      VAEResnetBlock(ch, ch, groups, eps)])

    def forward(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class VAEDownsample(nn.Module):
    """VAE convention: asymmetric pad (0,1,0,1) then stride-2 conv pad0."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class VAEUpsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class DownEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_res, groups, eps, add_down):
        super().__init__()
        self.resnets = nn.ModuleList(
            [VAEResnetBlock(in_ch if i == 0 else out_ch, out_ch, groups, eps) for i in range(n_res)])
        self.downsamplers = nn.ModuleList([VAEDownsample(out_ch)]) if add_down else None

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class UpDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_res, groups, eps, add_up):
        super().__init__()
        self.resnets = nn.ModuleList(
            [VAEResnetBlock(in_ch if i == 0 else out_ch, out_ch, groups, eps) for i in range(n_res)])
        self.upsamplers = nn.ModuleList([VAEUpsample(out_ch)]) if add_up else None

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


# ----------------------------------------------------------------------------
# encoder / decoder
# ----------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, cfg: SDXLVAEConfig):
        super().__init__()
        C = cfg.block_out_channels
        g, e = cfg.norm_num_groups, cfg.norm_eps
        self.conv_in = nn.Conv2d(cfg.in_channels, C[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        in_ch = C[0]
        for i, out_ch in enumerate(C):
            self.down_blocks.append(
                DownEncoderBlock(in_ch, out_ch, cfg.layers_per_block, g, e, add_down=(i < len(C) - 1)))
            in_ch = out_ch
        self.mid_block = VAEMidBlock(C[-1], g, e)
        self.conv_norm_out = nn.GroupNorm(g, C[-1], eps=e)
        self.conv_out = nn.Conv2d(C[-1], 2 * cfg.latent_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for b in self.down_blocks:
            x = b(x)
        x = self.mid_block(x)
        x = self.conv_out(F.silu(self.conv_norm_out(x)))
        return x                                                        # [B, 2*latent, h, w]


class Decoder(nn.Module):
    def __init__(self, cfg: SDXLVAEConfig):
        super().__init__()
        C = cfg.block_out_channels
        g, e = cfg.norm_num_groups, cfg.norm_eps
        rev = list(reversed(C))                                         # [512,512,256,128]
        self.conv_in = nn.Conv2d(cfg.latent_channels, rev[0], 3, padding=1)
        self.mid_block = VAEMidBlock(rev[0], g, e)
        self.up_blocks = nn.ModuleList()
        in_ch = rev[0]
        for i, out_ch in enumerate(rev):
            self.up_blocks.append(
                UpDecoderBlock(in_ch, out_ch, cfg.layers_per_block + 1, g, e, add_up=(i < len(rev) - 1)))
            in_ch = out_ch
        self.conv_norm_out = nn.GroupNorm(g, rev[-1], eps=e)
        self.conv_out = nn.Conv2d(rev[-1], cfg.out_channels, 3, padding=1)

    def forward(self, z):
        x = self.conv_in(z)
        x = self.mid_block(x)
        for b in self.up_blocks:
            x = b(x)
        x = self.conv_out(F.silu(self.conv_norm_out(x)))
        return x


# ----------------------------------------------------------------------------
# diagonal gaussian + the VAE
# ----------------------------------------------------------------------------

class DiagonalGaussian:
    """moments[B,2*lat,h,w] -> mean/logvar; .sample()/.mode()."""
    def __init__(self, moments: torch.Tensor):
        self.mean, self.logvar = moments.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        eps = torch.randn(self.mean.shape, generator=generator,
                          device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.std * eps

    def mode(self) -> torch.Tensor:
        return self.mean


class NativeSDXLVAE(nn.Module):
    def __init__(self, cfg: SDXLVAEConfig = SDXLVAEConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.quant_conv = nn.Conv2d(2 * cfg.latent_channels, 2 * cfg.latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(cfg.latent_channels, cfg.latent_channels, 1)

    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        return DiagonalGaussian(self.quant_conv(self.encoder(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))

    # convenience matching the trainer's usage (latent already *scaling_factor)
    def encode_latent(self, x, generator=None, scale=True):
        z = self.encode(x).sample(generator)
        return z * self.cfg.scaling_factor if scale else z

    def decode_latent(self, z, scaled=True):
        return self.decode(z / self.cfg.scaling_factor if scaled else z)