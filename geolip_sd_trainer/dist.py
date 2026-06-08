"""
dist.py — minimal torch.distributed seam for multi-RunPod runs.
================================================================
Single-pod runs (no env vars / WORLD_SIZE=1) behave EXACTLY as before: every helper
degrades to the trivial single-process answer and no process group is created. Multi-
pod runs are driven by the standard env vars a launcher (torchrun / a RunPod entry
script) sets: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT.

Storage-agnostic by design: cross-pod coordination uses torch.distributed.barrier()
when a process group exists, plus filesystem MARKER FILES that also work for
data-prep-only pods that never form a group. Any shared substrate (a RunPod network
volume now; S3 / HF shards later) plugs in behind the plain `cache_dir` path without
touching the sharding logic.

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch


def _grp_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    return torch.distributed.get_rank() if _grp_ready() else int(os.environ.get("RANK", 0))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def world_size() -> int:
    return torch.distributed.get_world_size() if _grp_ready() else int(os.environ.get("WORLD_SIZE", 1))


def is_distributed() -> bool:
    return world_size() > 1


def is_main() -> bool:
    return rank() == 0


def init_distributed(device: str = "cuda") -> int:
    """Init the process group if WORLD_SIZE>1 and it isn't already up. Returns the local
    rank so the caller can pin its GPU. No-op (returns 0) for single-pod runs."""
    if not is_distributed():
        return 0
    if not _grp_ready():
        backend = "nccl" if device.startswith("cuda") and torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    lr = local_rank()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(lr)
    return lr


def barrier():
    """Block until all ranks arrive (no-op without a process group)."""
    if _grp_ready():
        torch.distributed.barrier()


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Average a tensor across ranks (logging only). Unchanged for single-pod runs."""
    if _grp_ready() and world_size() > 1:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        t = t / world_size()
    return t


def wait_for_marker(path, timeout: float = 86_400.0, poll: float = 3.0):
    """Poll for a filesystem marker file — cross-pod coordination that also works when
    there is no process group (e.g. independent data-prep pods). Raises on timeout."""
    path = Path(path)
    waited = 0.0
    while not path.exists():
        if waited >= timeout:
            raise TimeoutError(f"timed out after {timeout:.0f}s waiting for marker {path}")
        time.sleep(poll)
        waited += poll
