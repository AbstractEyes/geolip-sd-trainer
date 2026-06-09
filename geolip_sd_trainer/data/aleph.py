"""
data/aleph.py — geolip-aleph-void address encoder (caption -> [32,128]).
========================================================================
Thin wrapper over the `geolip-svae` package (code) + the `AbstractPhil/geolip-aleph-void`
HF repo (weights), producing the byte-trigram aleph address that phase0 stores in its
`aleph_address` column.

Recipe (confirmed against geolip-svae aleph_model.py + the package CLAUDE.md):
  caption --text_to_image--> byte-trigram image (C,H,W)
          --AlephModel.eval()--> out['svd']['aleph_logits']  (B, N, V=32, 2K=128)
          --mean over patches N--> [B, 32, 128]

`aleph_logits = cat([cos, -cos], dim=-1)` where `cos = M_rows @ A.t()` are cosine
similarities between sphere-normalized rows and the codebook axes — so values lie in
[-1, 1], which matches phase0's observed aleph_address range (~[-0.95, 0.92]). No extra
nonlinearity is applied (`post="none"`), and the package's default patch aggregation is
`mean`. These are therefore the defaults below; `source`/`aggregate`/`post` remain
configurable in case a future cache uses a different checkpoint or aggregation.

Install:  pip install "git+https://github.com/AbstractEyes/geolip-svae.git"

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class AlephConfig:
    repo_id: str = "AbstractPhil/geolip-aleph-void"
    hf_version: str = "aleph_byte_trigram_tied_hard_K64"   # byte-trigram, K=64 -> 2K=128=addr_vd
    img_size: int = 64
    patch_size: int = 2
    pad: str = "space"
    channels: int = 3
    source: str = "logits"            # CONFIRMED: aleph_logits=cat([cos,-cos]) -> [V=32, 2K=128]
    aggregate: str = "mean"           # CONFIRMED: package default = mean over all N patches
    post: str = "none"                # cos is already in [-1,1]; "tanh" only if a cache differs
    device: str = "cuda"


class AlephEncoder:
    """Loads the frozen aleph-void model and maps captions -> [B,32,128] fp16 addresses."""

    def __init__(self, cfg: Optional[AlephConfig] = None, device: Optional[str] = None,
                 token: Optional[str] = None):
        self.cfg = cfg or AlephConfig()
        dev = device or self.cfg.device
        try:
            from geolip_svae import load_model
        except ImportError as e:
            raise SystemExit(
                "geolip-svae is required for aleph addresses.\n"
                "  pip install \"git+https://github.com/AbstractEyes/geolip-svae.git\"\n"
                f"  (import error: {e})")
        self.model, self.mcfg = load_model(hf_version=self.cfg.hf_version,
                                           repo_id=self.cfg.repo_id, device=dev)
        self.model.eval()                                  # eval -> aleph_logits emitted
        try:
            self.model.requires_grad_(False)
        except Exception:
            pass
        self.device = dev
        self._pdtype = next(self.model.parameters()).dtype
        # geometry: model attributes are authoritative (patch_size/channels are set at build);
        # img_size is the byte-trigram render size from the checkpoint config (else default).
        g = self.mcfg if isinstance(self.mcfg, dict) else {}
        self.patch_size = int(getattr(self.model, "patch_size", g.get("patch_size", self.cfg.patch_size)))
        self.channels = int(getattr(self.model, "channels", g.get("channels", self.cfg.channels)))
        self.img_size = int(g.get("img_size", self.cfg.img_size))

    @torch.no_grad()
    def caption_to_aleph(self, captions: List[str]) -> np.ndarray:
        """list[str] -> [B, 32, 128] fp16 aleph addresses."""
        from geolip_svae.inference.text import text_to_image
        imgs = torch.stack([
            text_to_image(c, self.img_size, self.patch_size, self.cfg.pad, self.channels)
            for c in captions]).to(self.device, self._pdtype)
        out = self.model(imgs)
        svd = out["svd"]
        if self.cfg.source == "logits" and "aleph_logits" in svd:
            t = svd["aleph_logits"]                         # (B, N, V, 2K)
        else:
            m = svd["M_hat"]                                # (B, N, V, D)
            t = m.reshape(m.shape[0], m.shape[1], m.shape[2], -1)
        agg = t[:, 0] if self.cfg.aggregate == "first" else t.mean(dim=1)   # (B, V, last)
        if self.cfg.post == "tanh":
            agg = torch.tanh(agg)
        return agg.float().cpu().numpy().astype(np.float16)
