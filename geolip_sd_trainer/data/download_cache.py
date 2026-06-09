"""
data/download_cache.py — SCRIPT B: consume the prepared columnar cache from HF.
===============================================================================
Two modes, both multi-GPU-sharded by RANK/WORLD_SIZE:

  * MATERIALIZE — download/stream the columnar cache, decode the 6 trainer columns, and
    write the exact per-row `.npz` (+ merged manifest) that trainer.CachedDS already reads.
    Each rank handles a disjoint dataset shard (.shard(num_shards, index)); rank 0 merges
    the per-rank manifests. After this, Phase1Trainer.setup() short-circuits build_cache
    (the manifest + files already exist) — zero re-encode. Best for a shared RunPod volume.

  * STREAM-DIRECT — `HFColumnarDS`, a map-style Dataset that decodes the 6 binary columns
    on the fly. No ~130 GB local copy; the training DistributedSampler shards it per rank.

Because every cache row carries an explicit `rid`, dataset sharding is safe (ids come from
the column, not row position), so `.shard()` cleanly splits the parquet files across pods.

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .. import dist
from ..checkpoint import resolve_hf_token
from ..trainer import (_cache_path, _save_row, manifest_path, _rank_manifest_path,
                       _atomic_write_text)
from .cache_format import TRAINER_KEYS, decode_trainer_npz


@dataclass
class DownloadCacheConfig:
    repo: str = "AbstractPhil/sdxl-qwen-phase1-cache"
    cache_dir: str = "./phase1_cache"
    n_images: int = 0                       # 0 = full; >0 caps (smoke test)
    device: str = "cuda"
    token: Optional[str] = None


# ---------------------------------------------------------------------------
# materialize -> local per-row .npz (drop-in for trainer's CachedDS)
# ---------------------------------------------------------------------------

def materialize(cfg: Optional[DownloadCacheConfig] = None) -> List[str]:
    """Download the columnar cache and write trainer-format .npz under cfg.cache_dir.
    Idempotent + rank-sharded; rank 0 merges manifests. Returns the full id list."""
    cfg = cfg or DownloadCacheConfig()
    import datasets as hfds

    dist.init_distributed(cfg.device)
    rank, ws = dist.rank(), dist.world_size()
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    tag = f"ids_{cfg.n_images or 'all'}"
    man = manifest_path(cfg.cache_dir, cfg.n_images)
    if man.exists():
        ids = json.loads(man.read_text())
        if ids and all(_cache_path(cfg.cache_dir, r).exists() for r in ids):
            print(f"✓ local cache already complete ({len(ids)} rows) — skipping download")
            return ids

    token = cfg.token or resolve_hf_token()
    stream = hfds.load_dataset(cfg.repo, split="train", streaming=True, token=token)
    if cfg.n_images:
        stream = stream.take(cfg.n_images)
    if ws > 1:
        stream = stream.shard(num_shards=ws, index=rank)     # disjoint parquet files per rank

    from tqdm.auto import tqdm
    shard_note = f" [rank {rank}/{ws}]" if ws > 1 else ""
    ids: List[str] = []
    for row in tqdm(stream, desc=f"download-cache{shard_note}"):
        rid = row["rid"]
        ids.append(rid)
        path = _cache_path(cfg.cache_dir, rid)
        if not path.exists():
            _save_row(path, decode_trainer_npz(row))         # 6 fp16 keys -> .npz

    if ws == 1:
        _atomic_write_text(man, json.dumps(ids))
        print(f"✓ materialized {len(ids)} rows -> {cfg.cache_dir}")
        return ids

    # multi-rank: per-rank manifest + rank-0 merge (same marker dance as build_cache)
    _atomic_write_text(_rank_manifest_path(cfg.cache_dir, cfg.n_images, rank), json.dumps(ids))
    (Path(cfg.cache_dir) / f"{tag}.rank{rank}.done").write_text("ok")
    merged_marker = Path(cfg.cache_dir) / f"{tag}.merged"
    if dist.is_main():
        for r in range(ws):
            dist.wait_for_marker(Path(cfg.cache_dir) / f"{tag}.rank{r}.done")
        merged: List[str] = []
        for r in range(ws):
            merged.extend(json.loads(_rank_manifest_path(cfg.cache_dir, cfg.n_images, r).read_text()))
        _atomic_write_text(man, json.dumps(merged))
        merged_marker.write_text("ok")
        print(f"✓ materialized — merged {len(merged)} rows from {ws} ranks -> {cfg.cache_dir}")
        ids = merged
    else:
        dist.wait_for_marker(merged_marker)
        ids = json.loads(man.read_text())
    dist.barrier()
    return ids


# ---------------------------------------------------------------------------
# stream-direct -> map-style Dataset (no local copy)
# ---------------------------------------------------------------------------

class HFColumnarDS(torch.utils.data.Dataset):
    """Map-style dataset over the HF columnar cache; decodes the 6 trainer columns in
    __getitem__ and returns them in CachedDS order (lat, clipl, qpool, addr, clipg, clipgp)
    as fp16 tensors. Works with a DistributedSampler for multi-GPU training."""

    def __init__(self, repo: str = "AbstractPhil/sdxl-qwen-phase1-cache", split: str = "train",
                 n_images: int = 0, token: Optional[str] = None):
        from datasets import load_dataset
        ds = load_dataset(repo, split=split, token=token or resolve_hf_token())
        if n_images:
            ds = ds.select(range(min(n_images, len(ds))))
        self.ds = ds.with_format(None)                       # raw python (binary bytes intact)
        self.qwen_hidden = int(len(self.ds[0]["qpool"]) // 2)   # fp16 -> 2 bytes/elem

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = decode_trainer_npz(self.ds[i])
        t = {k: torch.from_numpy(d[k]) for k in TRAINER_KEYS}
        return t["lat"], t["clipl"], t["qpool"], t["addr"], t["clipg"], t["clipgp"]


def _cfg_from_env() -> DownloadCacheConfig:
    e = os.environ.get
    return DownloadCacheConfig(
        repo=e("GEOLIP_CACHE_REPO", "AbstractPhil/sdxl-qwen-phase1-cache"),
        cache_dir=e("GEOLIP_CACHE_DIR", "./phase1_cache"),
        n_images=int(e("GEOLIP_N_IMAGES", "0")),
    )


if __name__ == "__main__":
    materialize(_cfg_from_env())
