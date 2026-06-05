"""
Native SDXL UNet — pure PyTorch, no diffusers in the forward path.
====================================================================
A from-scratch reimplementation of SDXL-base-1.0's UNet2DConditionModel whose
module tree is named to match the diffusers checkpoint key-for-key, so the real
pretrained weights load with strict=True (see weights.py). This is the owned,
fully-controllable forward for the experimental trainer — every op is ours.

Architecture (verified against stabilityai/stable-diffusion-xl-base-1.0/unet):
  in/out 4ch · block_out_channels [320,640,1280] · layers_per_block 2
  down  : DownBlock2D, CrossAttnDownBlock2D(d2,h10), CrossAttnDownBlock2D(d10,h20)
  mid   : UNetMidBlock2DCrossAttn(d10,h20)
  up    : CrossAttnUpBlock2D(d10,h20), CrossAttnUpBlock2D(d2,h10), UpBlock2D
  cross_attention_dim 2048 · head_dim 64 · linear proj_in/out · GEGLU ff (mult 4)
  text_time add-embed: cat[text_embeds(1280), add_time_proj(time_ids 6)*256] -> 2816 -> 1280

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# config
# ============================================================================

@dataclass
class SDXLUNetConfig:
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: tuple = (320, 640, 1280)
    layers_per_block: int = 2
    transformer_layers_per_block: tuple = (1, 2, 10)   # index 0 unused (DownBlock2D)
    attention_head_dim: tuple = (5, 10, 20)            # = num heads per level (head_dim=64)
    cross_attention_dim: int = 2048
    norm_num_groups: int = 32
    norm_eps: float = 1e-5
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    addition_time_embed_dim: int = 256
    projection_class_embeddings_input_dim: int = 2816
    time_embed_dim: int = 1280


# ============================================================================
# timestep / sinusoidal embedding  (diffusers get_timestep_embedding parity)
# ============================================================================

def get_timestep_embedding(timesteps: torch.Tensor, dim: int,
                           flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0.0,
                           max_period: int = 10000) -> torch.Tensor:
    assert timesteps.dim() == 1
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half - downscale_freq_shift)
    emb = torch.exp(exponent)                                   # [half]
    emb = timesteps.float()[:, None] * emb[None, :]             # [B, half]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)   # [B, dim]
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1) # cos first
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    """linear_1 -> SiLU -> linear_2  (matches diffusers TimestepEmbedding)."""
    def __init__(self, in_dim: int, time_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, time_dim)
        self.linear_2 = nn.Linear(time_dim, time_dim)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


# ============================================================================
# resnet
# ============================================================================

class ResnetBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int = 1280,
                 groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, eps=eps)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch, eps=eps)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


# ============================================================================
# attention / transformer
# ============================================================================

class Attention(nn.Module):
    """SDXL cross/self attention. to_q/to_k/to_v are BIAS-FREE; to_out.0 has bias."""
    def __init__(self, query_dim: int, cross_dim: Optional[int], heads: int, dim_head: int = 64):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        cdim = cross_dim if cross_dim is not None else query_dim
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(cdim, inner, bias=False)
        self.to_v = nn.Linear(cdim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, query_dim), nn.Dropout(0.0)])

    def forward(self, x, context=None):
        ctx = x if context is None else context
        B, T, _ = x.shape
        q, k, v = self.to_q(x), self.to_k(ctx), self.to_v(ctx)
        h, d = self.heads, q.shape[-1] // self.heads
        q = q.view(B, T, h, d).transpose(1, 2)
        k = k.view(B, k.shape[1], h, d).transpose(1, 2)
        v = v.view(B, v.shape[1], h, d).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)             # scale = 1/sqrt(d)
        o = o.transpose(1, 2).reshape(B, T, h * d)
        return self.to_out[1](self.to_out[0](o))


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """net.0 GEGLU -> net.1 Dropout -> net.2 Linear  (inner = dim*mult)."""
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        self.net = nn.ModuleList([GEGLU(dim, inner), nn.Dropout(0.0), nn.Linear(inner, dim)])

    def forward(self, x):
        x = self.net[0](x)
        x = self.net[1](x)
        return self.net[2](x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, cross_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, None, heads)               # self
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(dim, cross_dim, heads)          # cross
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x, context):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class Transformer2DModel(nn.Module):
    """GroupNorm -> linear proj_in -> N basic blocks -> linear proj_out (+residual)."""
    def __init__(self, channels: int, heads: int, depth: int, cross_dim: int,
                 groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=eps)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, heads, cross_dim) for _ in range(depth)])
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x, context):
        res = x
        B, C, H, W = x.shape
        h = self.norm(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        h = self.proj_in(h)
        for blk in self.transformer_blocks:
            h = blk(h, context)
        h = self.proj_out(h)
        h = h.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return h + res


# ============================================================================
# sampling blocks
# ============================================================================

class Downsample2D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class DownBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, n_res, temb_dim, groups, eps, add_down):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, temb_dim, groups, eps)
             for i in range(n_res)])
        self.downsamplers = nn.ModuleList([Downsample2D(out_ch)]) if add_down else None

    def forward(self, x, temb, context, res_samples):
        for r in self.resnets:
            x = r(x, temb); res_samples.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x); res_samples.append(x)
        return x


class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, n_res, depth, heads, cross_dim, temb_dim, groups, eps, add_down):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, temb_dim, groups, eps)
             for i in range(n_res)])
        self.attentions = nn.ModuleList(
            [Transformer2DModel(out_ch, heads, depth, cross_dim, groups, eps) for _ in range(n_res)])
        self.downsamplers = nn.ModuleList([Downsample2D(out_ch)]) if add_down else None

    def forward(self, x, temb, context, res_samples):
        for r, a in zip(self.resnets, self.attentions):
            x = r(x, temb); x = a(x, context); res_samples.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x); res_samples.append(x)
        return x


class UpBlock2D(nn.Module):
    def __init__(self, prev_ch, out_ch, skip_chs, temb_dim, groups, eps, add_up):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock2D((prev_ch if i == 0 else out_ch) + skip_chs[i], out_ch, temb_dim, groups, eps)
             for i in range(len(skip_chs))])
        self.upsamplers = nn.ModuleList([Upsample2D(out_ch)]) if add_up else None

    def forward(self, x, temb, context, res_samples):
        for r in self.resnets:
            x = torch.cat([x, res_samples.pop()], dim=1); x = r(x, temb)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, prev_ch, out_ch, skip_chs, depth, heads, cross_dim, temb_dim, groups, eps, add_up):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock2D((prev_ch if i == 0 else out_ch) + skip_chs[i], out_ch, temb_dim, groups, eps)
             for i in range(len(skip_chs))])
        self.attentions = nn.ModuleList(
            [Transformer2DModel(out_ch, heads, depth, cross_dim, groups, eps) for _ in range(len(skip_chs))])
        self.upsamplers = nn.ModuleList([Upsample2D(out_ch)]) if add_up else None

    def forward(self, x, temb, context, res_samples):
        for r, a in zip(self.resnets, self.attentions):
            x = torch.cat([x, res_samples.pop()], dim=1); x = r(x, temb); x = a(x, context)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, ch, depth, heads, cross_dim, temb_dim, groups, eps):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(ch, ch, temb_dim, groups, eps),
                                      ResnetBlock2D(ch, ch, temb_dim, groups, eps)])
        self.attentions = nn.ModuleList([Transformer2DModel(ch, heads, depth, cross_dim, groups, eps)])

    def forward(self, x, temb, context):
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        x = self.resnets[1](x, temb)
        return x


# ============================================================================
# the UNet
# ============================================================================

class NativeSDXLUNet(nn.Module):
    def __init__(self, cfg: SDXLUNetConfig = SDXLUNetConfig()):
        super().__init__()
        self.cfg = cfg
        C0, C1, C2 = cfg.block_out_channels
        g, eps, td = cfg.norm_num_groups, cfg.norm_eps, cfg.time_embed_dim
        cd = cfg.cross_attention_dim
        h = cfg.attention_head_dim                              # (5,10,20) heads
        depth = cfg.transformer_layers_per_block

        # stem
        self.conv_in = nn.Conv2d(cfg.in_channels, C0, 3, padding=1)

        # time + text_time additional embedding
        self.time_embedding = TimestepEmbedding(C0, td)
        self.add_embedding = TimestepEmbedding(cfg.projection_class_embeddings_input_dim, td)

        # down
        self.down_blocks = nn.ModuleList([
            DownBlock2D(C0, C0, cfg.layers_per_block, td, g, eps, add_down=True),
            CrossAttnDownBlock2D(C0, C1, cfg.layers_per_block, depth[1], h[1], cd, td, g, eps, add_down=True),
            CrossAttnDownBlock2D(C1, C2, cfg.layers_per_block, depth[2], h[2], cd, td, g, eps, add_down=False),
        ])
        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(C2, depth[2], h[2], cd, td, g, eps)
        # up  (skip-channel lists verified against the checkpoint)
        self.up_blocks = nn.ModuleList([
            CrossAttnUpBlock2D(C2, C2, [C2, C2, C1], depth[2], h[2], cd, td, g, eps, add_up=True),
            CrossAttnUpBlock2D(C2, C1, [C1, C1, C0], depth[1], h[1], cd, td, g, eps, add_up=True),
            UpBlock2D(C1, C0, [C0, C0, C0], td, g, eps, add_up=False),
        ])
        # head
        self.conv_norm_out = nn.GroupNorm(g, C0, eps=eps)
        self.conv_out = nn.Conv2d(C0, cfg.out_channels, 3, padding=1)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                text_embeds: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        """sample[B,4,128,128], timestep[B] (float ok), ehs[B,T,2048],
        text_embeds[B,1280], time_ids[B,6] -> velocity[B,4,128,128]."""
        cfg = self.cfg
        B = sample.shape[0]
        if timestep.dim() == 0:
            timestep = timestep.expand(B)

        # time embedding
        t_emb = get_timestep_embedding(timestep, cfg.block_out_channels[0],
                                       cfg.flip_sin_to_cos, cfg.freq_shift).to(sample.dtype)
        emb = self.time_embedding(t_emb)                                   # [B,1280]
        # text_time additional embedding
        time_emb = get_timestep_embedding(time_ids.flatten(), cfg.addition_time_embed_dim,
                                          cfg.flip_sin_to_cos, cfg.freq_shift)
        time_emb = time_emb.reshape(B, -1).to(sample.dtype)                # [B, 6*256=1536]
        add_in = torch.cat([text_embeds, time_emb], dim=-1)               # [B, 2816]
        emb = emb + self.add_embedding(add_in)

        ehs = encoder_hidden_states
        sample = self.conv_in(sample)
        res_samples: List[torch.Tensor] = [sample]
        for blk in self.down_blocks:
            sample = blk(sample, emb, ehs, res_samples)
        sample = self.mid_block(sample, emb, ehs)
        for blk in self.up_blocks:
            sample = blk(sample, emb, ehs, res_samples)
        sample = self.conv_out(F.silu(self.conv_norm_out(sample)))
        return sample