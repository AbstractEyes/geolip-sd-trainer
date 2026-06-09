"""
data/aleph.py — geolip-aleph-void address encoder (caption -> [32,128]).
========================================================================
Thin wrapper over the `geolip-svae` package (code) + the `AbstractPhil/geolip-aleph-void`
HF repo (weights), producing the byte-trigram aleph address that phase0 stores in its
`aleph_address` column.

Pipeline (matches the package's text path):
  caption --text_to_image--> byte-trigram image (C,H,W)
          --AlephModel(eval)--> out['svd']['aleph_logits']  (B, N, V=32, 2K=128)
          --aggregate over patches N--> [B, 32, 128]

Install:  pip install "git+https://github.com/AbstractEyes/geolip-svae.git"

The exact tensor + aggregation that produced the existing 86k phase0 addresses is
configurable (`source`, `aggregate`, `post`) so you can pin it to match; the defaults
(logits + patch-mean) yield the [32,128] shape. If a generated batch's value distribution
doesn't match phase0 (~[-1, 1]), try `post="tanh"` or `source="m_hat"`.

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
    source: str = "logits"            # "logits" (V,2K)->[32,128] | "m_hat" (V,D) flattened
    aggregate: str = "mean"           # over patches N: "mean" | "first"
    post: str = "none"                # "none" | "tanh" (bound to ~[-1,1] like phase0)
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
        # geometry from the checkpoint config (fall back to defaults from text.py)
        g = self.mcfg if isinstance(self.mcfg, dict) else {}
        self.img_size = int(g.get("img_size", self.cfg.img_size))
        self.patch_size = int(g.get("patch_size", self.cfg.patch_size))
        self.channels = int(g.get("channels", self.cfg.channels))

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
