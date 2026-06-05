"""
model.py — assembled SDXL with the Qwen + aleph conditioning front-end.
=======================================================================
Cobbles the owned native components into one config-driven model:

  transformer/unet.py        -> NativeSDXLUNet        (denoiser, the trainable core)
  vae/vae.py                 -> NativeSDXLVAE          (latents + sample decode)
  text_encoder/clip_text.py  -> NativeCLIPTextEncoder  (CLIP-L + CLIP-G, owned)
  vlm/qwen.py                -> QwenPooledEncoder       (rich-pooled third encoder)
  + SDXLQwenFrontEnd         (the only new trainable module besides the UNet)

Two orthogonal config axes:
  * ComponentConfig    — which IMPLEMENTATION fills each slot (native | ... | hf for qwen),
                         swappable behind a uniform interface.
  * ConditioningConfig — which SDXL SLOT each encoder feeds (the four testing flags);
                         CONDITIONING_PRESETS holds the nine proto configs, PHASE1_RECIPE
                         is the locked rank-#3 `aleph_clipl_clipg_pooled`.

The front-end conditioning (verbatim with the phase-0 trainer):
  ehs[B, 77(+n_addr), 2048] = cat[ CLIP-L-half(768) | CLIP-G-half(1280) ] (+ aleph anchor)
  text_embeds[B, 1280]      = real CLIP-G pooled, or pool_proj(qwen) when swap_clip_g_pooled
  HARD RULE: never swap the CLIP-G *sequence* (swap_clip_g_seq stays False in phase 1).

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .vlm.qwen import QwenConfig


_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _dtype(name) -> torch.dtype:
    return name if isinstance(name, torch.dtype) else _DTYPES[name]


# ============================================================================
# conditioning config — the testing configurations (which encoder feeds which slot)
# ============================================================================

@dataclass
class ConditioningConfig:
    # the four flags (source of truth; the preset name is just a label)
    swap_clip_l: bool = True          # CLIP-L 768 seq slot  : real CLIP-L vs clipl_proj(qwen) broadcast
    swap_clip_g_seq: bool = False     # CLIP-G 1280 seq slot : real CLIP-G vs pool_proj(qwen)  (KEEP REAL)
    swap_clip_g_pooled: bool = True   # CLIP-G pooled embeds : real vs pool_proj(qwen)
    use_aleph: bool = True            # append the aleph anchor tokens
    # dims / geometry
    clip_g_dim: int = 1280
    clip_l_dim: int = 768
    addr_vd: int = 128                # V*D of the aleph address rows
    n_addr: int = 32                  # address rows (must match the dataset Array2D)
    pos_emb_init: float = 0.02
    addr_init: float = 1e-4           # near-zero -> anchor starts ~no-op (load-bearing)


# name -> (swap_clip_l, swap_clip_g_seq, swap_clip_g_pooled, use_aleph)
CONDITIONING_PRESETS: Dict[str, Tuple[bool, bool, bool, bool]] = {
    "base":                         (False, False, False, False),
    "clipg_pooled":                 (False, False, True,  False),
    "clipg_seq_pooled":             (False, True,  True,  False),   # poison tier — never ship
    "aleph_clipg_pooled":           (False, False, True,  True),    # measured #1
    "aleph_clipg_seq_pooled":       (False, True,  True,  True),
    "clipl":                        (True,  False, False, False),
    "aleph_clipl":                  (True,  False, False, True),
    "aleph_clipl_clipg_pooled":     (True,  False, True,  True),    # rank #3 — LOCKED phase-1 recipe
    "aleph_clipl_clipg_seq_pooled": (True,  True,  True,  True),
}
PHASE1_RECIPE = "aleph_clipl_clipg_pooled"


def conditioning_from_preset(name: str, **overrides) -> ConditioningConfig:
    if name not in CONDITIONING_PRESETS:
        raise KeyError(f"unknown conditioning preset '{name}'. options: {list(CONDITIONING_PRESETS)}")
    sl, sgs, sgp, ua = CONDITIONING_PRESETS[name]
    return ConditioningConfig(swap_clip_l=sl, swap_clip_g_seq=sgs,
                              swap_clip_g_pooled=sgp, use_aleph=ua, **overrides)


# ============================================================================
# component config — swappable implementations behind the uniform interface
# ============================================================================

@dataclass
class ComponentConfig:
    sdxl_repo: str = "stabilityai/stable-diffusion-xl-base-1.0"
    unet_impl: str = "native"
    vae_impl: str = "native"
    clip_l_impl: str = "native"
    clip_g_impl: str = "native"
    qwen_impl: str = "hf"                       # frozen HF wrapper (vlm/qwen.py)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    variant: str = "fp16"                       # checkpoint file variant to pull
    dtype: str = "bf16"                         # UNet / CLIP / Qwen compute dtype
    vae_dtype: str = "fp32"                     # VAE unstable in fp16 -> fp32
    device: str = "cuda"
    token: Optional[str] = None


@dataclass
class SDXLModelConfig:
    components: ComponentConfig = field(default_factory=ComponentConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    qwen_hidden: int = 1024                     # Qwen3.5-0.8B hidden (trainer overrides from cache)
    image_size: int = 1024                      # latent 128x128
    vae_scale: float = 0.13025


# ============================================================================
# front-end — the only new trainable module besides the UNet (verbatim w/ phase-0)
# ============================================================================

class SDXLQwenFrontEnd(nn.Module):
    """Config-driven SDXL conditioning. Each slot is filled by the REAL CLIP output
    or a qwen projection per the flags; the aleph anchor is optionally appended.
    All submodules are ALWAYS constructed (flags gate the forward, not the parameter
    set), so the state-dict is identical across every conditioning config."""

    def __init__(self, cond: ConditioningConfig, qwen_hidden: int):
        super().__init__()
        self.cond = cond
        self.pool_proj = nn.Linear(qwen_hidden, cond.clip_g_dim)
        self.pos_emb = nn.Parameter(torch.randn(77, cond.clip_g_dim) * cond.pos_emb_init)
        self.clipl_proj = nn.Linear(qwen_hidden, cond.clip_l_dim)
        self.clipl_pos_emb = nn.Parameter(torch.randn(77, cond.clip_l_dim) * cond.pos_emb_init)
        self.addr_adapter = nn.Linear(cond.addr_vd, cond.clip_g_dim)
        nn.init.normal_(self.addr_adapter.weight, std=cond.addr_init)
        nn.init.zeros_(self.addr_adapter.bias)

    def forward(self, qwen_pool: torch.Tensor, clip_l_seq: torch.Tensor,
                clip_g_seq: torch.Tensor, clip_g_pool: torch.Tensor,
                address: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """qwen_pool[B,qh], clip_l_seq[B,77,768], clip_g_seq[B,77,1280],
        clip_g_pool[B,1280], address[B,n_addr,128] -> ehs[B,77(+n_addr),2048], text_embeds[B,1280]."""
        c = self.cond
        B = qwen_pool.shape[0]
        qg = self.pool_proj(qwen_pool)                                   # [B,1280]

        if c.swap_clip_l:
            ql = self.clipl_proj(qwen_pool)                              # [B,768]
            seq_l = ql.unsqueeze(1) + self.clipl_pos_emb.unsqueeze(0)    # [B,77,768]
        else:
            seq_l = clip_l_seq

        if c.swap_clip_g_seq:
            seq_g = qg.unsqueeze(1) + self.pos_emb.unsqueeze(0)          # [B,77,1280]
        else:
            seq_g = clip_g_seq

        main = torch.cat([seq_l, seq_g], dim=-1)                         # [B,77,2048]
        text_embeds = qg if c.swap_clip_g_pooled else clip_g_pool        # [B,1280]

        if c.use_aleph:
            addr_g = self.addr_adapter(address)                          # [B,n_addr,1280]
            addr_l = torch.zeros(B, address.shape[1], c.clip_l_dim,
                                 device=qg.device, dtype=main.dtype)
            addr_tok = torch.cat([addr_l, addr_g], dim=-1)               # [B,n_addr,2048]
            ehs = torch.cat([main, addr_tok], dim=1)
        else:
            ehs = main
        return ehs, text_embeds


# ============================================================================
# component loaders (registry) — swap native <-> other behind the interface
# ============================================================================

def _load_unet(comp: ComponentConfig):
    from .transformer.unet_test import load_native_sdxl_unet
    return load_native_sdxl_unet(comp.sdxl_repo, variant=comp.variant,
                                 device=comp.device, dtype=_dtype(comp.dtype), token=comp.token)


def _load_vae(comp: ComponentConfig):
    from .vae.vae_test import load_native_sdxl_vae
    return load_native_sdxl_vae(comp.sdxl_repo, variant="fp32",
                                device=comp.device, dtype=_dtype(comp.vae_dtype), token=comp.token)


def _load_clip_l(comp: ComponentConfig):
    from .text_encoder.clip_text_test import load_clip_l
    return load_clip_l(comp.sdxl_repo, device=comp.device, dtype=_dtype(comp.dtype),
                       variant=comp.variant, token=comp.token)


def _load_clip_g(comp: ComponentConfig):
    from .text_encoder.clip_g_test import load_clip_g
    return load_clip_g(comp.sdxl_repo, device=comp.device, dtype=_dtype(comp.dtype),
                       variant=comp.variant, token=comp.token)


def _load_qwen(comp: ComponentConfig):
    from .vlm.qwen import build_qwen
    return build_qwen(comp.qwen, device=comp.device, dtype=_dtype(comp.dtype), token=comp.token)


_LOADERS = {"unet": {"native": _load_unet}, "vae": {"native": _load_vae},
            "clip_l": {"native": _load_clip_l}, "clip_g": {"native": _load_clip_g},
            "qwen": {"hf": _load_qwen}}
_IMPL_KEY = {"unet": "unet_impl", "vae": "vae_impl", "clip_l": "clip_l_impl",
             "clip_g": "clip_g_impl", "qwen": "qwen_impl"}

ALL_COMPONENTS = ("unet", "vae", "clip_l", "clip_g", "qwen")
TRAIN_COMPONENTS = ("unet", "vae")               # features are cached -> encoders not needed at train
ENCODER_COMPONENTS = ("vae", "clip_l", "clip_g", "qwen")   # precompute set


# ============================================================================
# assembled model
# ============================================================================

class GeolipSDXL:
    """Owned SDXL stack behind one config. Components load selectively (cached
    features mean the encoders aren't needed at train time). Exposes the owned
    interface the trainer/sampler use; the trainable surface is `frontend` (+ the
    UNet / its LoRA)."""

    def __init__(self, cfg: SDXLModelConfig,
                 load: Sequence[str] = TRAIN_COMPONENTS, build_frontend: bool = True):
        self.cfg = cfg
        self.comp = cfg.components
        self.unet = self.vae = self.clip_l = self.clip_g = self.qwen = None
        self._tok_l = self._tok_g = None
        self.frontend = SDXLQwenFrontEnd(cfg.conditioning, cfg.qwen_hidden) if build_frontend else None
        for name in load:
            self.load_component(name)

    def load_component(self, name: str):
        impl = getattr(self.comp, _IMPL_KEY[name])
        table = _LOADERS[name]
        if impl not in table:
            raise KeyError(f"no '{impl}' implementation registered for '{name}' (have {list(table)})")
        setattr(self, name, table[impl](self.comp))
        return getattr(self, name)

    # -- tokenizers (preprocessing only; deterministic BPE, not part of the owned forward) --
    def _tokenizers(self):
        if self._tok_l is None:
            from transformers import CLIPTokenizer
            self._tok_l = CLIPTokenizer.from_pretrained(self.comp.sdxl_repo, subfolder="tokenizer",
                                                        token=self.comp.token)
            self._tok_g = CLIPTokenizer.from_pretrained(self.comp.sdxl_repo, subfolder="tokenizer_2",
                                                        token=self.comp.token)
        return self._tok_l, self._tok_g

    def tokenize(self, captions, which="l"):
        tok_l, tok_g = self._tokenizers()
        tok = tok_l if which == "l" else tok_g
        ids = tok(list(captions), padding="max_length", max_length=77, truncation=True,
                  return_tensors="pt").input_ids
        return ids.to(self.comp.device)

    # -- owned interface --
    @torch.no_grad()
    def vae_encode_latent(self, images: torch.Tensor, generator=None) -> torch.Tensor:
        return self.vae.encode(images).sample(generator) * self.cfg.vae_scale

    @torch.no_grad()
    def vae_decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self.cfg.vae_scale)

    @torch.no_grad()
    def encode_clip_l(self, captions) -> torch.Tensor:
        seq, _ = self.clip_l.encode_sdxl(self.tokenize(captions, "l"))
        return seq                                                       # [B,77,768]

    @torch.no_grad()
    def encode_clip_g(self, captions):
        seq, pooled = self.clip_g.encode_sdxl(self.tokenize(captions, "g"))
        return seq, pooled                                              # [B,77,1280], [B,1280]

    @torch.no_grad()
    def encode_qwen(self, captions) -> torch.Tensor:
        return self.qwen.encode(list(captions))                         # [B, qwen_hidden] cpu fp32

    def build_conditioning(self, qwen_pool, clip_l_seq, clip_g_seq, clip_g_pool, address):
        return self.frontend(qwen_pool, clip_l_seq, clip_g_seq, clip_g_pool, address)

    def unet_velocity(self, x_t, t, ehs, text_embeds, time_ids):
        return self.unet(x_t, t, encoder_hidden_states=ehs, text_embeds=text_embeds, time_ids=time_ids)

    def build_time_ids(self, B: int, device=None, dtype=None) -> torch.Tensor:
        s = self.cfg.image_size
        tid = torch.tensor([s, s, 0, 0, s, s], device=device or self.comp.device,
                           dtype=dtype or _dtype(self.comp.dtype))
        return tid.unsqueeze(0).repeat(B, 1)

    def to(self, device=None, dtype=None):
        if self.frontend is not None:
            self.frontend.to(device=device, dtype=dtype)
        return self


def build_sdxl(conditioning_preset: str = PHASE1_RECIPE,
               components: Optional[ComponentConfig] = None,
               load: Sequence[str] = TRAIN_COMPONENTS, qwen_hidden: int = 1024,
               **cond_overrides) -> GeolipSDXL:
    """Convenience: assemble a GeolipSDXL from a named testing preset."""
    cfg = SDXLModelConfig(components=components or ComponentConfig(),
                          conditioning=conditioning_from_preset(conditioning_preset, **cond_overrides),
                          qwen_hidden=qwen_hidden)
    return GeolipSDXL(cfg, load=load)