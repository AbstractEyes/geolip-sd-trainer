"""
trainer.py — phase-1 SDXL→Qwen+aleph Lune trainer (geolip-sd-trainer).
======================================================================
Trains the locked rank-#3 recipe (`aleph_clipl_clipg_pooled`) on the owned native
backbone with the Lune rectified-flow objective. Phase-1 specifics over phase-0:

  * STAGED WARMUP   — Stage A: UNet frozen, only the front-end calibrates the
                      swapped slots + anchor; Stage B: unfreeze the UNet (mode:
                      full | selective_attn2 | frozen) at a LOW LR with a ramp,
                      front-end continues at a higher LR. (the "feed it carefully".)
  * SHIFT SCHEDULE  — per-run constant; phase-1 STARTS at 2.0 (Lune's native
                      pretrain regime), not 2.5. Across-runs progression is human-
                      gated: run2 2.0-2.25, run3+ 2.5-3.0 (RUN_SHIFTS).
  * DROPOUT SCHEDULE— cfg_dropout starts HIGH (~0.22) and decays toward ~0.10, so
                      the unconditional is well-learned early (CFG headroom).
  * SAFEGUARDS      — every epoch: FID+KID (>=100 imgs/cfg, clean-fid) at each eval
                      cfg; 4x/epoch: an 8-prompt grid rendered at every eval cfg.
                      (mission-critical; counts/cadence are configurable.)

Objective (Lune, verbatim): s~U(0,1); s'=shift*s/(1+(shift-1)s); t=s'*1000 (FLOAT);
x_t=(1-s')x0+s'*noise; v=noise-x0; loss=MSE(unet(x_t,t,ehs,text_embeds,time_ids), v).
CFG dropout zeros the FULL conditioning (ehs + text_embeds) on a fraction; time_ids kept.

Run precompute (build_cache) once first; the trainer reads the cached fp16 features.

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (GeolipSDXL, SDXLModelConfig, ComponentConfig, conditioning_from_preset,
                    PHASE1_RECIPE, TRAIN_COMPONENTS, ENCODER_COMPONENTS)
from .checkpoint import (save_checkpoint, load_checkpoint, find_latest_checkpoint,
                         rotate_checkpoints, resolve_hf_token, hf_whoami, HubUploader,
                         export_unet_safetensors)

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

# across-runs shift progression (human-gated "depending how it fared")
RUN_SHIFTS = {1: 2.0, 2: 2.25, 3: 2.75}


# ============================================================================
# config
# ============================================================================

@dataclass
class Phase1Config:
    # data
    dataset_repo: str = "AbstractPhil/sdxl-qwen-phase0"
    cache_dir: str = "./phase1_cache"
    n_images: int = 0                       # 0 = full (~86k); >0 caps for smoke tests
    n_addr: int = 32

    # recipe / components
    conditioning_preset: str = PHASE1_RECIPE
    components: ComponentConfig = field(default_factory=ComponentConfig)
    image_size: int = 1024
    vae_scale: float = 0.13025

    # Lune objective
    shift: float = 2.0                      # phase-1 STARTS at 2.0 (not 2.5); raise across runs
    # dropout schedule (start high -> decay)
    cfg_dropout_start: float = 0.22
    cfg_dropout_end: float = 0.10
    cfg_dropout_mode: str = "cosine"        # "cosine" | "linear"
    cfg_dropout_hold_frac: float = 0.0      # fraction of training to hold at start before decaying

    # UNet movement (staged warmup)
    unet_mode: str = "full"                 # "full" | "selective_attn2" | "frozen"
    stage_a_steps: int = 400                # front-end-only pre-settle (UNet frozen)
    unet_warmup_steps: int = 1000           # UNet LR ramp after Stage A
    unet_lr: float = 1e-5                   # low (full finetune)
    frontend_lr: float = 1e-4               # higher (front-end from scratch)
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0

    # schedule / run
    num_epochs: int = 60
    batch_size: int = 4
    train_dtype: str = "bf16"               # bf16 on Blackwell
    seed: int = 42
    num_workers: int = 2

    # safeguards — FID/KID (mission-critical, configurable)
    eval_cfgs: Tuple[float, ...] = (1.0, 3.0, 5.0)
    fid_every_epochs: int = 1
    fid_images: int = 100                   # >=100 generated per cfg
    fid_ref_dir: Optional[str] = None       # real reference images (built by build_cache)
    fid_sample_steps: int = 28
    # safeguards — prompt grid
    prompt_grid_per_epoch: int = 4          # at 25/50/75/100% of the epoch
    prompt_grid_n: int = 8                  # fixed prompts (first N cached rows)
    prompt_grid_steps: int = 28
    sample_seed: int = 1234                 # fixed noise -> only the model varies

    # checkpoint / upload
    out_dir: str = "./phase1_runs"
    run_name: str = "phase1_aleph_clipl_clipg_pooled"
    hf_repo_id: str = "AbstractPhil/geolip-sdxl-aleph"
    hf_phase: str = "phase_1"
    upload_to_hub: bool = True
    upload_every_epochs: int = 5
    keep_last: int = 2
    device: str = "cuda"


# ============================================================================
# schedules
# ============================================================================

class DropoutSchedule:
    """cfg_dropout: hold at `start`, then decay to `end` over the run."""
    def __init__(self, start: float, end: float, total_steps: int,
                 mode: str = "cosine", hold_frac: float = 0.0):
        self.start, self.end = start, end
        self.total = max(1, total_steps)
        self.mode = mode
        self.hold = int(hold_frac * self.total)

    def value(self, step: int) -> float:
        if step <= self.hold:
            return self.start
        prog = min(1.0, (step - self.hold) / max(1, self.total - self.hold))
        if self.mode == "cosine":
            return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * prog))
        return self.start + (self.end - self.start) * prog


def lr_factors(step: int, cfg: Phase1Config) -> Tuple[float, float]:
    """(frontend_factor, unet_factor) in [0,1]. Front-end on from step 0; UNet stays
    0 through Stage A, then ramps over unet_warmup_steps."""
    fe = 1.0
    if step < cfg.stage_a_steps:
        return fe, 0.0
    ramp = (step - cfg.stage_a_steps) / max(1, cfg.unet_warmup_steps)
    return fe, float(min(1.0, ramp))


# ============================================================================
# Lune objective + float-t Euler sampler
# ============================================================================

def fm_targets(x0: torch.Tensor, shift: float):
    B = x0.shape[0]
    s = torch.rand(B, device=x0.device, dtype=x0.dtype)
    s = (shift * s) / (1 + (shift - 1) * s)
    t = s * 1000.0                                          # FLOAT timestep
    s4 = s.view(B, 1, 1, 1)
    noise = torch.randn_like(x0)
    x_t = (1 - s4) * x0 + s4 * noise
    v = noise - x0
    return x_t, t, v


@torch.no_grad()
def euler_sample(model: GeolipSDXL, feat_batch, n_steps: int, shift: float,
                 guidance: float = 1.0, seed: Optional[int] = None, dtype=torch.bfloat16):
    """Integrate dz=-v dσ from σ=1→0 with the front-end conditioning, decode → PIL.
    feat_batch = (lat, clipl, qpool, addr, clipg, clipgp); zeroed-cond CFG."""
    device = model.comp.device
    lat, clipl, qpool, addr, clipg, clipgp = [b.to(device, dtype) for b in feat_batch]
    B = lat.shape[0]
    ehs, txt = model.build_conditioning(qpool, clipl, clipg, clipgp, addr)
    tids = model.build_time_ids(B, device, dtype)
    use_cfg = guidance is not None and guidance != 1.0
    if use_cfg:
        ehs_u, txt_u = torch.zeros_like(ehs), torch.zeros_like(txt)

    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(lat.shape, generator=g, dtype=torch.float32).to(device, dtype)
    else:
        z = torch.randn_like(lat)
    sig = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=dtype)
    sig = (shift * sig) / (1 + (shift - 1) * sig)
    for i in range(n_steps):
        s, s_next = sig[i], sig[i + 1]
        t = (s * 1000.0).repeat(B)
        v = model.unet_velocity(z, t, ehs, txt, tids)
        if use_cfg:
            v_u = model.unet_velocity(z, t, ehs_u, txt_u, tids)
            v = v_u + guidance * (v - v_u)
        z = z + (s_next - s) * v
    img = model.vae_decode_latent(z)
    img = ((img.float() / 2 + 0.5).clamp(0, 1) * 255).round().byte().cpu().numpy()
    from PIL import Image
    return [Image.fromarray(img[k].transpose(1, 2, 0)) for k in range(B)]


# ============================================================================
# cached dataset (reads the fp16 npy cache from build_cache)
# ============================================================================

def _cache_paths(cache_dir, rid):
    d = Path(cache_dir)
    return (d / f"{rid}_lat.npy", d / f"{rid}_clipl.npy", d / f"{rid}_qpool.npy",
            d / f"{rid}_clipg.npy", d / f"{rid}_clipgp.npy", d / f"{rid}_addr.npy")


class CachedDS(torch.utils.data.Dataset):
    """Serves (lat, clipl, qpool, addr, clipg, clipgp) from disk, indexed by the
    precompute id-manifest. No HF dataset object or parquet access at train time."""
    def __init__(self, cache_dir, ids):
        self.cache_dir, self.ids = cache_dir, list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pl, pc, pq, pg, pgp, pa = _cache_paths(self.cache_dir, self.ids[i])
        load = lambda p: torch.from_numpy(np.load(p)).float()
        return load(pl), load(pc), load(pq), load(pa), load(pg), load(pgp)


def manifest_path(cache_dir, n_images):
    return Path(cache_dir) / f"ids_{n_images or 'all'}.json"


# ============================================================================
# precompute — stream the dataset columnar, cache features via the OWNED encoders
# ============================================================================

@torch.no_grad()
def build_cache(cfg: Phase1Config) -> List[str]:
    """Stream rows columnar (datasets streaming + .iter), cache per row:
    latent / CLIP-L seq / CLIP-G seq+pooled / Qwen pooled / aleph address — using
    GeolipSDXL's OWNED native encoders. Idempotent: returns ids without streaming if
    the manifest + all files exist. Also caches the FID reference images once."""
    import datasets as hfds
    import torchvision.transforms as T

    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    man = manifest_path(cfg.cache_dir, cfg.n_images)
    if man.exists():
        ids = json.loads(man.read_text())
        if ids and all(all(p.exists() for p in _cache_paths(cfg.cache_dir, r)) for r in ids):
            print(f"✓ cache complete ({len(ids)} rows) — skipping precompute")
            return ids

    dtype = _DTYPES[cfg.train_dtype]
    mcfg = SDXLModelConfig(components=cfg.components,
                           conditioning=conditioning_from_preset(cfg.conditioning_preset, n_addr=cfg.n_addr),
                           image_size=cfg.image_size, vae_scale=cfg.vae_scale)
    enc = GeolipSDXL(mcfg, load=ENCODER_COMPONENTS, build_frontend=False)   # vae+clip_l+clip_g+qwen
    to_tensor = T.Compose([T.Resize(cfg.image_size), T.CenterCrop(cfg.image_size), T.ToTensor()])

    ref_dir = Path(cfg.fid_ref_dir or (Path(cfg.cache_dir) / "fid_ref"))
    ref_dir.mkdir(parents=True, exist_ok=True)

    stream = hfds.load_dataset(cfg.dataset_repo, split="train", streaming=True)
    if cfg.n_images:
        stream = stream.take(cfg.n_images)
    print(f"Streaming + precomputing {cfg.n_images or 'all'} rows via owned encoders ...")

    from tqdm.auto import tqdm
    ids, gpos = [], 0
    for batch in tqdm(stream.iter(batch_size=16), desc="precompute"):
        caps, pil, addrs = batch["caption"], batch["image"], batch["aleph_address"]
        has_id = "id" in batch
        n = len(caps)
        rids = [str(batch["id"][j]) if has_id else f"row{gpos + j:08d}" for j in range(n)]
        gpos += n
        ids.extend(rids)
        todo = [j for j in range(n) if not all(p.exists() for p in _cache_paths(cfg.cache_dir, rids[j]))]
        if not todo:
            continue
        caps_t = [caps[j] for j in todo]
        imgs = torch.stack([to_tensor(pil[j].convert("RGB")) for j in todo])
        imgs = (imgs * 2 - 1).to(enc.comp.device, _DTYPES[cfg.components.vae_dtype])

        lat = enc.vae_encode_latent(imgs.to(_DTYPES[cfg.components.vae_dtype]))   # [b,4,128,128]
        clip_l_seq = enc.encode_clip_l(caps_t)
        clip_g_seq, clip_g_pool = enc.encode_clip_g(caps_t)
        qpool = enc.encode_qwen(caps_t)
        for jj, j in enumerate(todo):
            pl, pc, pq, pg, pgp, pa = _cache_paths(cfg.cache_dir, rids[j])
            np.save(pl, lat[jj].float().cpu().numpy().astype(np.float16))
            np.save(pc, clip_l_seq[jj].float().cpu().numpy().astype(np.float16))
            np.save(pq, qpool[jj].numpy().astype(np.float16))
            np.save(pg, clip_g_seq[jj].float().cpu().numpy().astype(np.float16))
            np.save(pgp, clip_g_pool[jj].float().cpu().numpy().astype(np.float16))
            np.save(pa, np.asarray(addrs[j], dtype=np.float16))
            if len(list(ref_dir.glob("*.png"))) < max(cfg.fid_images, 200):    # real reference set
                pil[j].convert("RGB").resize((cfg.image_size, cfg.image_size)).save(
                    ref_dir / f"{rids[j]}.png")

    man.write_text(json.dumps(ids))
    print(f"✓ precompute done ({len(ids)} rows); fid reference -> {ref_dir}")
    return ids


# ============================================================================
# safeguards — FID/KID (clean-fid) + prompt grid
# ============================================================================

@torch.no_grad()
def fid_kid_eval(model: GeolipSDXL, eval_rows: List, cfg: Phase1Config, epoch: int,
                 out_root: Path, dtype) -> Dict[float, Dict[str, float]]:
    """Generate >= fid_images per eval cfg and score FID+KID vs the real reference
    (clean-fid). KID is the trustworthy metric at this N. Returns {cfg: {fid,kid}}."""
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        print("  (FID/KID skipped: `pip install clean-fid`)")
        return {}
    ref_dir = str(cfg.fid_ref_dir or (Path(cfg.cache_dir) / "fid_ref"))
    results = {}
    for g in cfg.eval_cfgs:
        gen_dir = out_root / "fid" / f"epoch_{epoch:04d}" / f"cfg_{g}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        made = 0
        bs = 4
        idx = 0
        while made < cfg.fid_images and idx < len(eval_rows):
            batch = eval_rows[idx:idx + bs]; idx += bs
            feat = tuple(torch.stack([r[k] for r in batch]) for k in range(6))
            pics = euler_sample(model, feat, cfg.fid_sample_steps, cfg.shift,
                                guidance=g, seed=None, dtype=dtype)
            for p in pics:
                p.save(gen_dir / f"{made:05d}.png"); made += 1
        try:
            f = cleanfid.compute_fid(str(gen_dir), ref_dir, mode="clean", verbose=False)
            k = cleanfid.compute_kid(str(gen_dir), ref_dir, mode="clean", verbose=False)
            results[g] = {"fid": float(f), "kid": float(k), "n": made}
            print(f"  FID/KID @cfg {g}: FID {f:.1f}  KID {k:.4f}  (n={made})")
        except Exception as e:
            print(f"  FID/KID @cfg {g} failed: {type(e).__name__}: {e}")
    return results


@torch.no_grad()
def prompt_grid_eval(model: GeolipSDXL, sample_feat, cfg: Phase1Config,
                     out_dir: Path, tag: str, dtype):
    """Render the fixed prompts at each eval cfg (fixed seed -> only the model
    varies). out_dir/<tag>/cfg_<g>/<n>.png."""
    for g in cfg.eval_cfgs:
        d = out_dir / tag / f"cfg_{g}"; d.mkdir(parents=True, exist_ok=True)
        pics = euler_sample(model, sample_feat, cfg.prompt_grid_steps, cfg.shift,
                            guidance=g, seed=cfg.sample_seed, dtype=dtype)
        for n, p in enumerate(pics):
            p.save(d / f"{n}.png")


# ============================================================================
# trainer
# ============================================================================

class Phase1Trainer:
    def __init__(self, cfg: Phase1Config):
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = _DTYPES[cfg.train_dtype]
        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.run_dir = Path(cfg.out_dir) / cfg.run_name
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "samples").mkdir(parents=True, exist_ok=True)
        self.token = resolve_hf_token()
        if cfg.upload_to_hub and self.token:
            print(f"  HF user: {hf_whoami(self.token)}")
        self.uploader = HubUploader(cfg.hf_repo_id, cfg.hf_phase, cfg.run_name, self.token,
                                    enabled=cfg.upload_to_hub)

    # -- setup: cache -> model -> trainable set -> optimizer/schedules --
    def setup(self):
        cfg = self.cfg
        self.ids = build_cache(cfg)
        self.ds = CachedDS(cfg.cache_dir, self.ids)
        self.loader = torch.utils.data.DataLoader(
            self.ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            drop_last=True, pin_memory=True, persistent_workers=cfg.num_workers > 0)
        self.steps_per_epoch = len(self.loader)
        self.total_steps = self.steps_per_epoch * cfg.num_epochs

        qwen_hidden = int(np.load(_cache_paths(cfg.cache_dir, self.ids[0])[2]).shape[-1])
        mcfg = SDXLModelConfig(
            components=cfg.components,
            conditioning=conditioning_from_preset(cfg.conditioning_preset, n_addr=cfg.n_addr),
            qwen_hidden=qwen_hidden, image_size=cfg.image_size, vae_scale=cfg.vae_scale)
        self.model = GeolipSDXL(mcfg, load=TRAIN_COMPONENTS)          # unet + vae
        self.model.frontend.to(self.device, self.dtype)

        # trainable set by mode (mode-agnostic checkpointing keys off requires_grad)
        self.model.unet.requires_grad_(False)
        if cfg.unet_mode == "full":
            unet_params = [p for p in self.model.unet.parameters()]
        elif cfg.unet_mode == "selective_attn2":
            unet_params = [p for n, p in self.model.unet.named_parameters() if "attn2" in n]
        elif cfg.unet_mode == "frozen":
            unet_params = []
        else:
            raise ValueError(f"unknown unet_mode {cfg.unet_mode}")
        self.unet_params = unet_params
        for p in unet_params:
            p.requires_grad_(False)                                  # Stage A: UNet frozen
        self.model.frontend.requires_grad_(True)

        groups = [{"params": list(self.model.frontend.parameters()), "lr": cfg.frontend_lr, "name": "frontend"}]
        if unet_params:
            groups.append({"params": unet_params, "lr": cfg.unet_lr, "name": "unet"})
        self.opt = torch.optim.AdamW(groups, betas=cfg.betas, weight_decay=cfg.weight_decay, eps=1e-8)
        self.dropout = DropoutSchedule(cfg.cfg_dropout_start, cfg.cfg_dropout_end,
                                       self.total_steps, cfg.cfg_dropout_mode, cfg.cfg_dropout_hold_frac)
        self.gstep = 0
        self.start_epoch = 0
        self.loss_log: List[dict] = []
        self._stage_b = False

        # fixed sample prompts (first N rows) — stable across epochs
        sb = [self.ds[i] for i in range(min(cfg.prompt_grid_n, len(self.ds)))]
        self.sample_feat = tuple(torch.stack([b[k] for b in sb]) for k in range(6))
        self.eval_rows = [self.ds[i] for i in range(min(len(self.ds), cfg.fid_images + 64))]

        self._maybe_resume()
        print(f"  setup: {len(self.ds)} rows · {self.steps_per_epoch} steps/epoch × {cfg.num_epochs} "
              f"= {self.total_steps} · unet_mode={cfg.unet_mode} "
              f"(trainable unet params {sum(p.numel() for p in unet_params)/1e6:.0f}M) shift={cfg.shift}")

    def _modules(self):
        return {"unet": self.model.unet, "frontend": self.model.frontend}

    def _maybe_resume(self):
        latest = find_latest_checkpoint(self.run_dir / "checkpoints")
        if latest is None:
            return
        meta = load_checkpoint(latest, self._modules(), optimizer=self.opt, map_location="cpu")
        self.gstep = meta.get("gstep", 0)
        self.start_epoch = meta.get("epoch", 0)
        self.loss_log = meta.get("loss_log", [])
        if self.gstep >= self.cfg.stage_a_steps:
            self._enter_stage_b()
        print(f"↻ resumed from {latest.name} @ epoch {self.start_epoch} (gstep {self.gstep})")

    def _enter_stage_b(self):
        if self._stage_b:
            return
        self._stage_b = True
        if not self.unet_params:                                     # frozen mode: nothing to unfreeze
            return
        for p in self.unet_params:
            p.requires_grad_(True)
        print(f"  → Stage B: UNet unfrozen ({self.cfg.unet_mode}) at gstep {self.gstep}")

    def _set_lrs(self):
        fe_f, unet_f = lr_factors(self.gstep, self.cfg)
        for grp in self.opt.param_groups:
            if grp["name"] == "frontend":
                grp["lr"] = self.cfg.frontend_lr * fe_f
            else:
                grp["lr"] = self.cfg.unet_lr * unet_f

    def _trainable(self):
        return [p for p in self.model.frontend.parameters() if p.requires_grad] + \
               [p for p in self.unet_params if p.requires_grad]

    def fit(self):
        cfg = self.cfg
        self.model.unet.train(); self.model.frontend.train()
        grid_marks = {int(self.steps_per_epoch * f) for f in (0.25, 0.5, 0.75)}
        for epoch in range(self.start_epoch + 1, cfg.num_epochs + 1):
            ep_loss = []
            from tqdm.auto import tqdm
            pbar = tqdm(self.loader, desc=f"epoch {epoch}/{cfg.num_epochs}")
            for step_in_epoch, batch in enumerate(pbar):
                if not self._stage_b and self.gstep >= cfg.stage_a_steps:
                    self._enter_stage_b()
                self._set_lrs()
                lat, clipl, qpool, addr, clipg, clipgp = [b.to(self.device, self.dtype, non_blocking=True) for b in batch]
                B = lat.shape[0]

                ehs, txt = self.model.build_conditioning(qpool, clipl, clipg, clipgp, addr)
                tids = self.model.build_time_ids(B, self.device, self.dtype)

                p_drop = self.dropout.value(self.gstep)
                drop = torch.rand(B, device=self.device) < p_drop
                if drop.any():
                    ehs = ehs.clone(); txt = txt.clone()
                    ehs[drop] = 0; txt[drop] = 0

                x_t, t, v = fm_targets(lat, cfg.shift)
                pred = self.model.unet_velocity(x_t, t, ehs, txt, tids)
                loss = F.mse_loss(pred.float(), v.float())

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(self._trainable(), cfg.grad_clip)
                self.opt.step()
                self.gstep += 1
                ep_loss.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "drop": f"{p_drop:.2f}",
                                  "gnorm": f"{float(gnorm):.2f}", "stage": "B" if self._stage_b else "A"})

                if step_in_epoch in grid_marks and cfg.prompt_grid_per_epoch > 1:
                    self._prompt_grid(epoch, f"e{epoch:04d}_{int(100*step_in_epoch/self.steps_per_epoch):03d}pct")

            mean_loss = float(np.mean(ep_loss)) if ep_loss else float("nan")
            self.loss_log.append({"epoch": epoch, "loss": mean_loss, "gstep": self.gstep})
            (self.run_dir / "loss_log.json").write_text(json.dumps(self.loss_log, indent=2))
            print(f"  epoch {epoch} mean loss {mean_loss:.4f}")

            self._prompt_grid(epoch, f"e{epoch:04d}_100pct")            # 4th grid (end of epoch)
            if epoch % cfg.fid_every_epochs == 0:
                self._fid(epoch)
            if epoch % cfg.upload_every_epochs == 0 or epoch == cfg.num_epochs:
                self._checkpoint(epoch)

        self._finalize()

    # -- safeguards / io --
    def _prompt_grid(self, epoch, tag):
        try:
            self.model.unet.eval(); self.model.frontend.eval()
            prompt_grid_eval(self.model, self.sample_feat, self.cfg,
                             self.run_dir / "samples", tag, self.dtype)
        except Exception as e:
            print(f"  (prompt grid {tag} skipped: {type(e).__name__}: {e})")
        finally:
            self.model.unet.train(); self.model.frontend.train()

    def _fid(self, epoch):
        try:
            self.model.unet.eval(); self.model.frontend.eval()
            res = fid_kid_eval(self.model, self.eval_rows, self.cfg, epoch, self.run_dir, self.dtype)
            with (self.run_dir / "fid_log.json").open("a") as fh:
                fh.write(json.dumps({"epoch": epoch, **{str(k): v for k, v in res.items()}}) + "\n")
        except Exception as e:
            print(f"  (FID/KID skipped: {type(e).__name__}: {e})")
        finally:
            self.model.unet.train(); self.model.frontend.train()

    def _checkpoint(self, epoch):
        ckpt = self.run_dir / "checkpoints" / f"ckpt_e{epoch:04d}.pt"
        save_checkpoint(ckpt, self._modules(), optimizer=self.opt,
                        meta={"epoch": epoch, "gstep": self.gstep, "loss_log": self.loss_log,
                              "config": asdict(self.cfg)})
        rotate_checkpoints(self.run_dir / "checkpoints", self.cfg.keep_last)
        print(f"  ✓ checkpoint {ckpt.name}")
        self.uploader.upload_checkpoint(ckpt, samples_root=str(self.run_dir / "samples"))

    def _finalize(self):
        out = self.run_dir / "checkpoints" / "unet_final.safetensors"
        export_unet_safetensors(out, self.model.unet)
        self.uploader.upload_folder(str(self.run_dir / "checkpoints"), "checkpoints",
                                    allow_patterns=["unet_final.safetensors"])
        print(f"\n✅ phase-1 run '{self.cfg.run_name}' complete.")


def train(cfg: Optional[Phase1Config] = None):
    cfg = cfg or Phase1Config()
    tr = Phase1Trainer(cfg)
    tr.setup()
    tr.fit()
    return tr


if __name__ == "__main__":
    train()