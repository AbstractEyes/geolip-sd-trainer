"""
Native SDXL UNet weight loading.
================================
The module tree in unet.py is named key-for-key to the diffusers SDXL
checkpoint, so loading the real pretrained weights is a plain strict
load_state_dict — no remapping table. This file just fetches the safetensors
(local path or HF) and asserts a clean strict load.

  from geolip_sd_trainer.backbone.native.weights import load_native_sdxl_unet
  unet = load_native_sdxl_unet(device="cuda", dtype=torch.bfloat16)   # real SDXL weights
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch

from .unet import NativeSDXLUNet, SDXLUNetConfig

SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"


def config_from_hf(repo: str = SDXL_REPO, token: Optional[str] = None) -> SDXLUNetConfig:
    """Build SDXLUNetConfig from the checkpoint's unet/config.json (so any future
    SDXL-family variant with different depths/heads is honored, not assumed)."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "config.json", subfolder="unet", token=token)
    c = json.loads(Path(p).read_text())
    return SDXLUNetConfig(
        in_channels=c["in_channels"], out_channels=c["out_channels"],
        block_out_channels=tuple(c["block_out_channels"]),
        layers_per_block=c["layers_per_block"],
        transformer_layers_per_block=tuple(c["transformer_layers_per_block"]),
        attention_head_dim=tuple(c["attention_head_dim"]),
        cross_attention_dim=c["cross_attention_dim"],
        norm_num_groups=c["norm_num_groups"], norm_eps=c["norm_eps"],
        flip_sin_to_cos=c["flip_sin_to_cos"], freq_shift=c["freq_shift"],
        addition_time_embed_dim=c["addition_time_embed_dim"],
        projection_class_embeddings_input_dim=c["projection_class_embeddings_input_dim"],
    )


def _load_state_dict(repo_or_path: str, variant: str, token: Optional[str]) -> dict:
    from safetensors.torch import load_file
    p = Path(repo_or_path)
    if p.is_file():
        return load_file(str(p))
    if p.is_dir():
        cand = p / ("diffusion_pytorch_model.fp16.safetensors" if variant == "fp16"
                    else "diffusion_pytorch_model.safetensors")
        return load_file(str(cand))
    # HF repo id
    from huggingface_hub import hf_hub_download
    fname = ("diffusion_pytorch_model.fp16.safetensors" if variant == "fp16"
             else "diffusion_pytorch_model.safetensors")
    local = hf_hub_download(repo_or_path, fname, subfolder="unet", token=token)
    return load_file(local)


def load_native_sdxl_unet(repo_or_path: str = SDXL_REPO, variant: str = "fp16",
                          device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                          cfg: Optional[SDXLUNetConfig] = None,
                          token: Optional[str] = None,
                          read_config_from_hub: bool = False) -> NativeSDXLUNet:
    """Instantiate NativeSDXLUNet and strict-load the real SDXL weights.

    repo_or_path : HF repo id, a local dir containing unet weights, or a direct
                   .safetensors path.
    variant      : 'fp16' or 'fp32' (which checkpoint file to pull from HF).
    read_config_from_hub : pull unet/config.json to build cfg (else use SDXL-base defaults).
    """
    if cfg is None:
        cfg = config_from_hf(repo_or_path, token) if read_config_from_hub else SDXLUNetConfig()
    sd = _load_state_dict(repo_or_path, variant, token)
    # Build on the meta device so the ~2.5B params are never random-initialized on CPU
    # (that init + the immediate overwrite is pure waste and doubles peak host memory).
    # assign=True swaps the checkpoint tensors straight in; the native tree is parameter-
    # only (no registered buffers), so the strict check below also guarantees nothing is
    # left stranded on meta.
    with torch.device("meta"):
        model = NativeSDXLUNet(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)   # strict checks below (clearer error)
    if missing or unexpected:
        raise RuntimeError(
            f"SDXL UNet load mismatch — missing={len(missing)} unexpected={len(unexpected)}.\n"
            f"  first missing:    {missing[:4]}\n  first unexpected: {unexpected[:4]}")
    model.to(device=device, dtype=dtype)
    return model


def assert_parity(cfg: Optional[SDXLUNetConfig] = None,
                  reference_keys_shapes: Optional[dict] = None) -> Tuple[int, bool]:
    """Offline check: NativeSDXLUNet(cfg).state_dict() matches a reference
    {key: shape} map exactly (built on meta device, no allocation). Returns
    (n_tensors, ok)."""
    cfg = cfg or SDXLUNetConfig()
    with torch.device("meta"):
        m = NativeSDXLUNet(cfg)
    mine = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    if reference_keys_shapes is None:
        return len(mine), True
    ref = {k: tuple(v) for k, v in reference_keys_shapes.items()}
    ok = set(mine) == set(ref) and all(mine[k] == ref[k] for k in mine)
    return len(mine), ok