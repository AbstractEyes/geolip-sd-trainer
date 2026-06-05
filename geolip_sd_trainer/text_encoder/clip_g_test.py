"""
Native CLIP-G loader (text_encoder_2 = CLIP-G, CLIPTextModelWithProjection).
============================================================================
CLIP-G is the same architecture as CLIP-L (text_encoder_1), so the model code
is shared (one source of truth in text_encoder_1/clip_text.py). This file is the
CLIP-G-specific preset + loader.

  from geolip_sd_trainer.text_encoder_2.clip_g_test import load_clip_g
  clip_g = load_clip_g(device="cuda", dtype=torch.bfloat16)
  seq, pooled = clip_g.encode_sdxl(input_ids)   # penultimate [B,77,1280], text_embeds [B,1280]
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .clip_text import NativeCLIPTextEncoder, CLIP_G_CONFIG
from .clip_text_test import load_clip_text, SDXL_REPO


def load_clip_g(repo_or_path: str = SDXL_REPO, device: str = "cuda",
                dtype: torch.dtype = torch.bfloat16, variant: str = "fp16",
                token: Optional[str] = None) -> NativeCLIPTextEncoder:
    return load_clip_text(CLIP_G_CONFIG, repo_or_path, "text_encoder_2", variant, device, dtype, token)


def assert_parity(reference_keys_shapes: Optional[dict] = None) -> Tuple[int, bool]:
    with torch.device("meta"):
        m = NativeCLIPTextEncoder(CLIP_G_CONFIG)
    mine = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    if reference_keys_shapes is None:
        return len(mine), True
    ref = {k: tuple(v) for k, v in reference_keys_shapes.items()}
    ok = set(mine) == set(ref) and all(mine[k] == ref[k] for k in mine)
    return len(mine), ok