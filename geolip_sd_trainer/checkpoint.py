"""
checkpoint.py — save / load / resume + HF upload for geolip-sd-trainer.
=======================================================================
Mode-agnostic by design. Phase-1 lets the UNet move (full / selective / LoRA),
so instead of special-casing each, the checkpoint saves whatever currently
REQUIRES GRAD, keyed by parameter name:

  * full finetune     -> the trainable set is the whole UNet
  * selective unfreeze -> only the unfrozen subset (e.g. attn2)
  * LoRA               -> only the adapter params (lora_A/lora_B)

Resume = reload the frozen base (strict, from the SDXL checkpoint) + apply the
saved trainable params over it + restore optimizer/scheduler/scaler. The frozen
giants are never stored, so checkpoints scale with what actually trains.

Upload mirrors phase-0: sparse checkpoints, fixed-seed samples, resumable batched
uploads with backoff, repo layout <phase>/<run>/{checkpoints,samples}. Token
handling is the hard-won version: OVERWRITE the env (not setdefault — a stale env
token silently wins), .strip(), and a whoami() preflight that fails fast (a 401
"invalid username/password" is an INVALID TOKEN, not a scope problem; public reads
can succeed with a bad token, so a working download does NOT prove validity).

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ============================================================================
# trainable-state extraction (the mode-agnostic core)
# ============================================================================

def trainable_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """All params that require grad, by name, on CPU. Full-finetune -> whole UNet;
    selective -> the subset; LoRA -> the adapter params. Buffers excluded (they
    come from the frozen base)."""
    return {n: p.detach().to("cpu") for n, p in module.named_parameters() if p.requires_grad}


def load_trainable_into(module: nn.Module, state: Dict[str, torch.Tensor],
                        strict: bool = False) -> List[str]:
    """Copy saved trainable params back by name (dtype/device-cast in place).
    Returns the saved keys that had no match. strict=True raises on any."""
    own = dict(module.named_parameters())
    missing = []
    with torch.no_grad():
        for n, v in state.items():
            if n in own:
                own[n].data.copy_(v.to(own[n].device, own[n].dtype))
            else:
                missing.append(n)
    if strict and missing:
        raise RuntimeError(f"load_trainable_into: {len(missing)} saved keys unmatched, e.g. {missing[:4]}")
    return missing


# ============================================================================
# save / load / resume
# ============================================================================

def save_checkpoint(path, modules: Dict[str, nn.Module], optimizer=None, scheduler=None,
                    scaler=None, meta: Optional[dict] = None) -> str:
    """Save trainable params of each named module + training state + meta.
    modules e.g. {'unet': unet, 'frontend': frontend}."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "modules": {k: trainable_state_dict(m) for k, m in modules.items()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "meta": meta or {},
    }
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp); os.replace(tmp, path)        # atomic-ish write
    return str(path)


def load_checkpoint(path, modules: Dict[str, nn.Module], optimizer=None, scheduler=None,
                    scaler=None, map_location="cpu", strict: bool = False) -> dict:
    """Restore trainable params into each named module + training state. Returns meta."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    for k, m in modules.items():
        if k in ck["modules"]:
            load_trainable_into(m, ck["modules"][k], strict=strict)
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return ck.get("meta", {})


def find_latest_checkpoint(ckpt_dir, pattern: str = "ckpt_e*.pt") -> Optional[Path]:
    files = sorted(Path(ckpt_dir).glob(pattern))
    return files[-1] if files else None


def rotate_checkpoints(ckpt_dir, keep_last: int, pattern: str = "ckpt_e*.pt"):
    """Delete oldest local checkpoints beyond keep_last (storage hygiene)."""
    files = sorted(Path(ckpt_dir).glob(pattern))
    for f in files[:-keep_last] if keep_last > 0 else []:
        f.unlink(missing_ok=True)


def export_unet_safetensors(path, unet: nn.Module):
    """Full UNet weights as safetensors for standalone inference / sharing
    (independent of the resume checkpoint). Use after a finetune run."""
    from safetensors.torch import save_file
    sd = {n: p.detach().to("cpu") for n, p in unet.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path))
    return str(path)


# ============================================================================
# HF token
# ============================================================================

def resolve_hf_token(env_var: str = "HF_TOKEN") -> Optional[str]:
    """env first, then Colab userdata. OVERWRITE the env (a stale env token would
    otherwise silently win) and .strip()."""
    tok = os.environ.get(env_var)
    if not tok:
        try:
            from google.colab import userdata
            tok = userdata.get(env_var)
        except Exception:
            tok = None
    if tok:
        tok = tok.strip()
        os.environ[env_var] = tok
        return tok
    return None


def hf_whoami(token: Optional[str]) -> str:
    """Fast-fail preflight. Raises with a clear message on an invalid token.
    NOTE: a working public download does NOT prove the token is valid — only
    whoami() does."""
    from huggingface_hub import HfApi
    try:
        who = HfApi(token=token).whoami()
    except Exception as e:
        raise SystemExit(
            f"\nHF token preflight failed: {type(e).__name__}: {e}\n"
            "A 401 'invalid username/password' means an INVALID TOKEN (not a missing scope).\n"
            "Set a valid write token in env HF_TOKEN (or Colab userdata) and retry.")
    return who.get("name", "?")


# ============================================================================
# uploader (resumable, backoff)
# ============================================================================

class HubUploader:
    """Uploads to <repo_id>/<phase>/<run_name>/. HF skips files it already has by
    hash, so a batched upload also carries every not-yet-uploaded epoch below it."""

    def __init__(self, repo_id: str, phase: str, run_name: str, token: Optional[str],
                 enabled: bool = True):
        self.repo_id = repo_id
        self.sub = f"{phase}/{run_name}"
        self.token = token
        self.enabled = enabled and bool(token)

    def _api(self):
        from huggingface_hub import HfApi
        return HfApi(token=self.token)

    def ensure_repo(self):
        from huggingface_hub import create_repo
        create_repo(self.repo_id, repo_type="model", exist_ok=True, token=self.token)

    def upload_checkpoint(self, ckpt_path, samples_root: Optional[str] = None,
                          retries: int = 4, large_samples: bool = False,
                          sample_patterns=("*.png",)) -> bool:
        """Upload one checkpoint file + (optionally) the whole local samples tree,
        with exponential backoff. Returns True on success."""
        if not self.enabled:
            return False
        ckpt_path = Path(ckpt_path)
        for attempt in range(retries):
            try:
                self.ensure_repo()
                api = self._api()
                api.upload_file(path_or_fileobj=str(ckpt_path), repo_id=self.repo_id,
                                repo_type="model",
                                path_in_repo=f"{self.sub}/checkpoints/{ckpt_path.name}")
                if samples_root and Path(samples_root).exists():
                    if large_samples:
                        self._upload_large(samples_root, f"{self.sub}/samples")
                    else:
                        api.upload_folder(folder_path=str(samples_root), repo_id=self.repo_id,
                                          repo_type="model", path_in_repo=f"{self.sub}/samples",
                                          allow_patterns=list(sample_patterns))
                print(f"  ↑ uploaded {ckpt_path.name}"
                      f"{' + samples' if samples_root else ''} → {self.repo_id}/{self.sub}")
                return True
            except Exception as e:
                wait = 2 ** attempt
                print(f"  ↑ upload {attempt + 1}/{retries} failed ({type(e).__name__}: {e}); retry {wait}s")
                time.sleep(wait)
        print(f"  ↑ upload FAILED after {retries} tries ({ckpt_path.name}) — kept locally")
        return False

    def upload_folder(self, local_dir, repo_subpath: str, retries: int = 4,
                      large: bool = False, allow_patterns=None) -> bool:
        """Upload an arbitrary local dir (e.g. a final UNet export or adapter)."""
        if not self.enabled:
            return False
        for attempt in range(retries):
            try:
                self.ensure_repo()
                if large:
                    self._upload_large(local_dir, f"{self.sub}/{repo_subpath}")
                else:
                    self._api().upload_folder(folder_path=str(local_dir), repo_id=self.repo_id,
                                              repo_type="model",
                                              path_in_repo=f"{self.sub}/{repo_subpath}",
                                              allow_patterns=allow_patterns)
                print(f"  ↑ {repo_subpath} → {self.repo_id}/{self.sub}")
                return True
            except Exception as e:
                wait = 2 ** attempt
                print(f"  ↑ {repo_subpath} {attempt + 1}/{retries} failed ({e}); retry {wait}s")
                time.sleep(wait)
        return False

    def _upload_large(self, local_dir, repo_subtree: str):
        """upload_large_folder is resumable but has NO path_in_repo — it mirrors the
        folder you point it at to the repo root. So stage a mirror whose top dirs ARE
        the desired subtree, then upload that staging root."""
        import shutil, tempfile
        staging = Path(tempfile.mkdtemp(prefix="hfstage_"))
        dest = staging / repo_subtree
        dest.parent.mkdir(parents=True, exist_ok=True)
        # symlink the content under the mirrored subtree (cheap; copy if symlink unsupported)
        try:
            dest.symlink_to(Path(local_dir).resolve(), target_is_directory=True)
        except OSError:
            shutil.copytree(local_dir, dest)
        try:
            self._api().upload_large_folder(folder_path=str(staging), repo_id=self.repo_id,
                                            repo_type="model")
        finally:
            shutil.rmtree(staging, ignore_errors=True)