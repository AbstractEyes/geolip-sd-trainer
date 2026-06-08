"""
test_optimizations.py — fast, dependency-light checks for the perf refactor.
==========================================================================
These validate the two highest-risk *mechanical* changes WITHOUT a GPU, model
download, or HF dataset:

  1. the single-.npz cache round-trips through CachedDS with the right shapes/order
  2. the new safetensors checkpoint round-trips AND the legacy .pt still loads
  3. the dist seam degrades to the single-process answer with no env vars

Run directly:   python tests/test_optimizations.py
Or with pytest: pytest tests/test_optimizations.py

The GPU/parity-gated checks (VAE bf16 latent parity, Qwen generate=False
equivalence, batched-CFG image parity, FID custom-stats, a WORLD_SIZE=2 dry run)
need CUDA + weights and are listed at the bottom as commands to run on a pod.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from geolip_sd_trainer.trainer import _cache_path, _save_row, CachedDS, _CACHE_KEYS
from geolip_sd_trainer.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from geolip_sd_trainer import dist


def test_cache_npz_roundtrip():
    """A row written by _save_row reads back through CachedDS with matching values and
    the trainer's expected unpack order (lat, clipl, qpool, addr, clipg, clipgp)."""
    with tempfile.TemporaryDirectory() as d:
        rid = "row00000000"
        arrays = {
            "lat":    np.random.randn(4, 128, 128).astype(np.float16),
            "clipl":  np.random.randn(77, 768).astype(np.float16),
            "qpool":  np.random.randn(1024).astype(np.float16),
            "clipg":  np.random.randn(77, 1280).astype(np.float16),
            "clipgp": np.random.randn(1280).astype(np.float16),
            "addr":   np.random.randn(32, 128).astype(np.float16),
        }
        _save_row(_cache_path(d, rid), arrays)
        assert _cache_path(d, rid).exists() and not Path(str(_cache_path(d, rid)) + ".tmp").exists()

        lat, clipl, qpool, addr, clipg, clipgp = CachedDS(d, [rid])[0]
        assert lat.dtype == torch.float16 and lat.shape == (4, 128, 128)
        assert clipl.shape == (77, 768) and clipg.shape == (77, 1280)
        assert qpool.shape == (1024,) and clipgp.shape == (1280,) and addr.shape == (32, 128)
        # order check: addr (returned 4th) must equal the saved addr, not clipg
        assert np.allclose(addr.numpy(), arrays["addr"])
        assert np.allclose(clipg.numpy(), arrays["clipg"])
        assert set(_CACHE_KEYS) == set(arrays)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)


def _roundtrip(suffix: str):
    with tempfile.TemporaryDirectory() as d:
        src = _Tiny()
        for p in src.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            src.lin.weight.add_(1.234)
        ckpt = Path(d) / f"ckpt_e0001{suffix}"
        save_checkpoint(ckpt, {"net": src}, optimizer=None,
                        meta={"epoch": 1, "gstep": 42, "loss_log": [{"epoch": 1}]},
                        save_optimizer=False)
        found = find_latest_checkpoint(Path(d))
        assert found is not None and found.name == ckpt.name

        dst = _Tiny()
        meta = load_checkpoint(found, {"net": dst})
        assert meta.get("gstep") == 42 and meta.get("epoch") == 1
        assert torch.allclose(src.lin.weight, dst.lin.weight)
        assert torch.allclose(src.lin.bias, dst.lin.bias)


def test_checkpoint_safetensors_roundtrip():
    _roundtrip(".safetensors")


def test_checkpoint_legacy_pt_roundtrip():
    _roundtrip(".pt")                           # backward compatibility


def test_optimizer_state_opt_in():
    """save_optimizer=False must NOT leave an optimizer sidecar; True must."""
    with tempfile.TemporaryDirectory() as d:
        net = _Tiny()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        opt.step()  # populate state
        off = Path(d) / "off.safetensors"
        save_checkpoint(off, {"net": net}, optimizer=opt, meta={}, save_optimizer=False)
        assert not Path(str(off) + ".opt.pt").exists()
        on = Path(d) / "on.safetensors"
        save_checkpoint(on, {"net": net}, optimizer=opt, meta={}, save_optimizer=True)
        assert Path(str(on) + ".opt.pt").exists()


def test_dist_single_process_defaults():
    assert dist.world_size() == 1
    assert dist.rank() == 0
    assert dist.is_main() is True
    assert dist.is_distributed() is False
    dist.barrier()                              # no-op, must not raise


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall fast checks passed.\n"
          "GPU/parity-gated checks to run on a pod (need CUDA + weights):\n"
          "  • VAE bf16 latent parity:   compare vae_encode_latent fp32 vs bf16 (< ~1e-2 after *vae_scale)\n"
          "  • Qwen generate=False:      ComponentConfig(qwen=QwenConfig(generate=False)) determinism + quality delta\n"
          "  • batched-CFG image parity: euler_sample(guidance=3.0) before/after, same seed, ~equal\n"
          "  • smoke train:              train(Phase1Config(n_images=64, num_epochs=1, upload_to_hub=False))\n"
          "  • multi-pod dry run:        WORLD_SIZE=2 torchrun --nproc_per_node=2 -m geolip_sd_trainer.trainer\n"
          "                              -> disjoint cache shards, rank-0 merge == single-pod id set, only rank0 writes")
