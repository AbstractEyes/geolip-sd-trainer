"""
Native CLIP text-encoder weight loading (text_encoder_1 = CLIP-L).
==================================================================
Module names match the HF CLIPTextModel checkpoint key-for-key, so loading the
real weights is a plain strict load_state_dict. The generic loader here also
serves CLIP-G (text_encoder_2/clip_g_test.py wraps it).

  from geolip_sd_trainer.text_encoder_1.clip_text_test import load_clip_l
  clip_l = load_clip_l(device="cuda", dtype=torch.bfloat16)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from .clip_text import NativeCLIPTextEncoder, CLIPTextConfig, CLIP_L_CONFIG

SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"


def _load_state_dict(repo_or_path: str, subfolder: str, variant: str, token: Optional[str]) -> dict:
    from safetensors.torch import load_file
    p = Path(repo_or_path)
    primary = "model.fp16.safetensors" if variant == "fp16" else "model.safetensors"
    fallback = "model.safetensors" if variant == "fp16" else "model.fp16.safetensors"
    if p.is_file():
        return load_file(str(p))
    if p.is_dir():
        for fn in (primary, fallback):
            if (p / fn).exists():
                return load_file(str(p / fn))
        raise FileNotFoundError(f"no CLIP safetensors in {p}")
    from huggingface_hub import hf_hub_download
    for fn in (primary, fallback):
        try:
            return load_file(hf_hub_download(repo_or_path, fn, subfolder=subfolder, token=token))
        except Exception:
            continue
    raise FileNotFoundError(f"could not fetch CLIP weights from {repo_or_path}/{subfolder}")


def load_clip_text(cfg: CLIPTextConfig, repo_or_path: str = SDXL_REPO,
                   subfolder: str = "text_encoder", variant: str = "fp16",
                   device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                   token: Optional[str] = None) -> NativeCLIPTextEncoder:
    sd = _load_state_dict(repo_or_path, subfolder, variant, token)
    # meta-device build avoids random-initializing CLIP on CPU just to overwrite it
    # (see load_native_sdxl_unet); assign=True swaps the checkpoint tensors straight in.
    with torch.device("meta"):
        model = NativeCLIPTextEncoder(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"CLIP load mismatch ({subfolder}) — missing={len(missing)} unexpected={len(unexpected)}.\n"
            f"  first missing:    {missing[:4]}\n  first unexpected: {unexpected[:4]}")
    model.to(device=device, dtype=dtype)
    return model


def load_clip_l(repo_or_path: str = SDXL_REPO, device: str = "cuda",
                dtype: torch.dtype = torch.bfloat16, variant: str = "fp16",
                token: Optional[str] = None) -> NativeCLIPTextEncoder:
    return load_clip_text(CLIP_L_CONFIG, repo_or_path, "text_encoder", variant, device, dtype, token)


def assert_parity(cfg: CLIPTextConfig = CLIP_L_CONFIG,
                  reference_keys_shapes: Optional[dict] = None) -> Tuple[int, bool]:
    with torch.device("meta"):
        m = NativeCLIPTextEncoder(cfg)
    mine = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    if reference_keys_shapes is None:
        return len(mine), True
    ref = {k: tuple(v) for k, v in reference_keys_shapes.items()}
    ok = set(mine) == set(ref) and all(mine[k] == ref[k] for k in mine)
    return len(mine), ok