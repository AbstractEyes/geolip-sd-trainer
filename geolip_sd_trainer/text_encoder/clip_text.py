"""
Native CLIP text encoder — pure PyTorch, no transformers in the path.
=====================================================================
One generic CLIPTextModel reimplementation that serves BOTH SDXL text encoders
(they are the same architecture, differing only in width/depth/act/projection):
  text_encoder_1 = CLIP-L  : 768 / 12L / 12H, quick_gelu, no projection
  text_encoder_2 = CLIP-G  : 1280 / 32L / 20H, gelu, + text_projection (pooled)

Module tree named key-for-key to the HF CLIPTextModel / CLIPTextModelWithProjection
checkpoint, so the real weights load strict=True (see *_test.py loaders).

SDXL usage (per encoder): the cross-attn sequence is the PENULTIMATE hidden
state hidden_states[-2] (no final LN); CLIP-G additionally yields the pooled
text_embeds = text_projection(last_hidden[eos]).  Causal attention; EOS pooling
uses argmax(input_ids) (the eos_token_id==2 legacy path, eos=49407 is the max id).

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CLIPTextConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int = 77
    vocab_size: int = 49408
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    eos_token_id: int = 2
    with_projection: bool = False
    projection_dim: int = 768


CLIP_L_CONFIG = CLIPTextConfig(768, 3072, 12, 12, hidden_act="quick_gelu",
                               with_projection=False, projection_dim=768)
CLIP_G_CONFIG = CLIPTextConfig(1280, 5120, 32, 20, hidden_act="gelu",
                               with_projection=True, projection_dim=1280)


def _act(name: str):
    if name == "quick_gelu":
        return lambda x: x * torch.sigmoid(1.702 * x)
    if name == "gelu":
        return lambda x: F.gelu(x)
    if name in ("gelu_new", "gelu_pytorch_tanh"):
        return lambda x: F.gelu(x, approximate="tanh")
    raise ValueError(f"unknown CLIP hidden_act '{name}'")


class CLIPAttention(nn.Module):
    """Causal multi-head self-attention with biased q/k/v/out (CLIP convention)."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        h, d = self.heads, C // self.heads
        q = self.q_proj(x).view(B, T, h, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, d).transpose(1, 2)
        # causal mask; an optional additive padding mask can be combined in
        if attn_mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(o)


class CLIPMLP(nn.Module):
    def __init__(self, dim: int, inter: int, act: str):
        super().__init__()
        self.fc1 = nn.Linear(dim, inter)
        self.fc2 = nn.Linear(inter, dim)
        self.act = _act(act)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.self_attn = CLIPAttention(cfg.hidden_size, cfg.num_attention_heads)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = CLIPMLP(cfg.hidden_size, cfg.intermediate_size, cfg.hidden_act)

    def forward(self, x, attn_mask=None):
        x = x + self.self_attn(self.layer_norm1(x), attn_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPEmbeddings(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)

    def forward(self, input_ids):
        T = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(pos)[None]


class CLIPEncoder(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(self, x, attn_mask=None):
        hidden_states: List[torch.Tensor] = [x]            # [0] = embeddings output
        for layer in self.layers:
            x = layer(x, attn_mask)
            hidden_states.append(x)                        # pre-final-LN per layer
        return x, hidden_states


class CLIPTextTransformer(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = CLIPEmbeddings(cfg)
        self.encoder = CLIPEncoder(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, input_ids, attn_mask=None):
        x = self.embeddings(input_ids)
        x, hidden_states = self.encoder(x, attn_mask)
        last = self.final_layer_norm(x)
        # EOS pooling — eos_token_id==2 legacy path: argmax of input_ids (eos=49407 is max)
        idx = input_ids.to(torch.int).argmax(dim=-1)
        pooled = last[torch.arange(last.shape[0], device=last.device), idx]
        return last, hidden_states, pooled


class CLIPOutput(NamedTuple):
    last_hidden_state: torch.Tensor
    hidden_states: tuple
    pooler_output: torch.Tensor
    text_embeds: Optional[torch.Tensor]


class NativeCLIPTextEncoder(nn.Module):
    """Generic CLIP text encoder. with_projection=True -> CLIP-G (adds text_projection)."""
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.cfg = cfg
        self.text_model = CLIPTextTransformer(cfg)
        if cfg.with_projection:
            self.text_projection = nn.Linear(cfg.hidden_size, cfg.projection_dim, bias=False)

    def forward(self, input_ids, attn_mask=None) -> CLIPOutput:
        last, hidden_states, pooled = self.text_model(input_ids, attn_mask)
        text_embeds = self.text_projection(pooled) if self.cfg.with_projection else None
        return CLIPOutput(last, tuple(hidden_states), pooled, text_embeds)

    @torch.no_grad()
    def encode_sdxl(self, input_ids, attn_mask=None):
        """SDXL conditioning extract: (penultimate sequence hidden_states[-2],
        pooled text_embeds or None)."""
        out = self.forward(input_ids, attn_mask)
        return out.hidden_states[-2], out.text_embeds