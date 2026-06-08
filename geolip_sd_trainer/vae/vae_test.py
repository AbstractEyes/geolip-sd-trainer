"""
Native SDXL VAE weight loading (mirrors transformer/unet_test.py).
==================================================================
Module names match the diffusers VAE checkpoint key-for-key, so loading the
real weights is a plain strict load_state_dict.

  from geolip_sd_trainer.vae.vae_test import load_native_sdxl_vae
  vae = load_native_sdxl_vae(device="cuda", dtype=torch.float32)   # run VAE in fp32

Note: SDXL's fp16 VAE is numerically unstable (force_upcast). Decode/encode in
fp32 (or bf16); the checkpoint ships fp32 weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch

from .vae import NativeSDXLVAE, SDXLVAEConfig

SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"


def config_from_hf(repo: str = SDXL_REPO, token: Optional[str] = None) -> SDXLVAEConfig:
    from huggingface_hub import hf_hub_download
    c = json.loads(Path(hf_hub_download(repo, "config.json", subfolder="vae", token=token)).read_text())
    return SDXLVAEConfig(
        in_channels=c["in_channels"], out_channels=c["out_channels"],
        latent_channels=c["latent_channels"],
        block_out_channels=tuple(c["block_out_channels"]),
        layers_per_block=c["layers_per_block"],
        norm_num_groups=c["norm_num_groups"],
        scaling_factor=c.get("scaling_factor", 0.13025),
    )


def _load_state_dict(repo_or_path: str, variant: str, token: Optional[str]) -> dict:
    from safetensors.torch import load_file
    p = Path(repo_or_path)
    if p.is_file():
        return load_file(str(p))
    fname = ("diffusion_pytorch_model.fp16.safetensors" if variant == "fp16"
             else "diffusion_pytorch_model.safetensors")
    if p.is_dir():
        return load_file(str(p / fname))
    from huggingface_hub import hf_hub_download
    return load_file(hf_hub_download(repo_or_path, fname, subfolder="vae", token=token))


def load_native_sdxl_vae(repo_or_path: str = SDXL_REPO, variant: str = "fp32",
                         device: str = "cuda", dtype: torch.dtype = torch.float32,
                         cfg: Optional[SDXLVAEConfig] = None, token: Optional[str] = None,
                         read_config_from_hub: bool = False) -> NativeSDXLVAE:
    if cfg is None:
        cfg = config_from_hf(repo_or_path, token) if read_config_from_hub else SDXLVAEConfig()
    sd = _load_state_dict(repo_or_path, variant, token)
    # meta-device build avoids random-initializing the VAE on CPU just to overwrite it
    # (see load_native_sdxl_unet); assign=True swaps the checkpoint tensors straight in.
    with torch.device("meta"):
        model = NativeSDXLVAE(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"SDXL VAE load mismatch — missing={len(missing)} unexpected={len(unexpected)}.\n"
            f"  first missing:    {missing[:4]}\n  first unexpected: {unexpected[:4]}")
    model.to(device=device, dtype=dtype)
    return model


def assert_parity(cfg: Optional[SDXLVAEConfig] = None,
                  reference_keys_shapes: Optional[dict] = None) -> Tuple[int, bool]:
    cfg = cfg or SDXLVAEConfig()
    with torch.device("meta"):
        m = NativeSDXLVAE(cfg)
    mine = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    if reference_keys_shapes is None:
        return len(mine), True
    ref = {k: tuple(v) for k, v in reference_keys_shapes.items()}
    ok = set(mine) == set(ref) and all(mine[k] == ref[k] for k in mine)
    return len(mine), ok