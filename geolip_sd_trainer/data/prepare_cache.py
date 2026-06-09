"""
data/prepare_cache.py — SCRIPT A: phase0 -> columnar training cache (HF parquet).
=================================================================================
Reproduces AbstractPhil/sdxl-qwen-phase1-cache from a phase0-format source
(image + caption + aleph_address[32,128]): streams the source, encodes each row with
the OWNED native encoders (VAE latent, CLIP-L/-G seq + pooled) plus the rich Qwen pass
(pooled + full left-padded sequence `qseq` + `seq_len` + the `gen_text` re-description),
copies the pre-computed `aleph_address`, and writes the 11-column zstd parquet shards
(256 rows each) to a target HF *dataset* repo.

This is the columnar analogue of trainer.build_cache (which writes local per-row .npz);
it mirrors the reference builder qwen_cache_dataset.py MODE="prepare" output exactly.

Multi-GPU: each rank ENCODES only its modulo shard and writes its own rank-namespaced
parquet shards (disjoint rids), so N RunPods append in parallel with no collisions. The
global row index keeps `rid = row{rid_offset+index:08d}` unique; append continues the
index past the target's existing rows.

Run (single GPU):     python -m geolip_sd_trainer.data.prepare_cache
Run (multi-GPU/pod):  RANK=<r> WORLD_SIZE=<N> python -m geolip_sd_trainer.data.prepare_cache
                      (or torchrun … -m geolip_sd_trainer.data.prepare_cache)

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .. import dist
from ..checkpoint import resolve_hf_token, hf_whoami
from ..model import (GeolipSDXL, SDXLModelConfig, ComponentConfig, conditioning_from_preset,
                     PHASE1_RECIPE, ENCODER_COMPONENTS)
from ..trainer import _threaded_prefetch
from .cache_format import (SHARD_ROWS, PAD_T, VAE_SCALE, cache_features, cache_meta, encode_f16)
from .hf_io import DatasetShardWriter, count_remote_rows

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class PrepareCacheConfig:
    source_repo: str = "AbstractPhil/sdxl-qwen-phase0"          # phase0-format input
    target_repo: str = "AbstractPhil/sdxl-qwen-phase1-cache"    # columnar cache output
    n_images: int = 0                                          # 0 = full; >0 caps (smoke test)
    n_addr: int = 32
    image_size: int = 1024
    vae_scale: float = VAE_SCALE
    pad_T: int = PAD_T
    batch_size: int = 16                                       # encode/stream batch
    components: ComponentConfig = field(default_factory=ComponentConfig)
    conditioning_preset: str = PHASE1_RECIPE
    append: bool = True                                        # continue rid past existing rows
    rid_offset: int = -1                                       # -1 = auto (count target rows)
    upload_meta: bool = True                                   # rank 0 writes meta.json + README
    private: bool = False
    device: str = "cuda"
    token: Optional[str] = None


def _readme(cfg: PrepareCacheConfig, qwen_hidden: int, meta: dict) -> str:
    rows = "\n".join(f"| `{k}` | binary fp16 | {v['shape']} |" for k, v in meta["arrays"].items())
    return (
        f"# {cfg.target_repo.split('/')[-1]}\n\n"
        f"Precomputed SDXL + Qwen3.5 conditioning cache built from `{cfg.source_repo}` by "
        f"`geolip_sd_trainer.data.prepare_cache`.\n\n"
        f"11 columns; tensor columns are raw little-endian float16 bytes in parquet `binary` "
        f"columns (no shape metadata — see `meta.json`). Shards: `data/shard_*.parquet`, "
        f"{SHARD_ROWS} rows each, zstd.\n\n"
        f"## Tensor columns\n\n| column | type | shape |\n|---|---|---|\n{rows}\n\n"
        f"Plus `rid`(string), `caption`(string), `gen_text`(string), `seq_len`(int64).\n\n"
        f"## Decode\n\n```python\nimport numpy as np\nfrom datasets import load_dataset\n"
        f"ds = load_dataset(\"{cfg.target_repo}\", split=\"train\")\n"
        f"row = ds[0]\nlat = np.frombuffer(row[\"lat\"], dtype=\"<f2\").reshape(4,128,128).copy()\n"
        f"qseq = np.frombuffer(row[\"qseq\"], dtype=\"<f2\").reshape({cfg.pad_T},{qwen_hidden})\n"
        f"real = qseq[{cfg.pad_T} - row[\"seq_len\"]:]   # strip left padding\n```\n"
    )


@torch.no_grad()
def prepare_cache(cfg: Optional[PrepareCacheConfig] = None):
    cfg = cfg or PrepareCacheConfig()
    import datasets as hfds
    import torchvision.transforms as T

    local_rank = dist.init_distributed(cfg.device)
    rank, ws = dist.rank(), dist.world_size()
    device = cfg.device
    if ws > 1 and cfg.device.startswith("cuda") and torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        cfg.components.device = device

    token = cfg.token or resolve_hf_token()
    if dist.is_main() and token:
        print(f"  HF user: {hf_whoami(token)}")

    rid_offset = cfg.rid_offset
    if rid_offset < 0:
        rid_offset = count_remote_rows(cfg.target_repo, token) if cfg.append else 0
    if dist.is_main():
        print(f"  prepare_cache: {cfg.source_repo} -> {cfg.target_repo} "
              f"(rid_offset={rid_offset}, world_size={ws})")

    # owned native encoders (vae + clip_l + clip_g + qwen); no front-end needed
    mcfg = SDXLModelConfig(
        components=cfg.components,
        conditioning=conditioning_from_preset(cfg.conditioning_preset, n_addr=cfg.n_addr),
        image_size=cfg.image_size, vae_scale=cfg.vae_scale)
    enc = GeolipSDXL(mcfg, load=ENCODER_COMPONENTS, build_frontend=False)
    qwen_hidden = enc.qwen.hidden
    vae_dt = _DTYPES[cfg.components.vae_dtype]
    to_tensor = T.Compose([T.Resize(cfg.image_size), T.CenterCrop(cfg.image_size), T.ToTensor()])

    stream = hfds.load_dataset(cfg.source_repo, split="train", streaming=True)
    if cfg.n_images:
        stream = stream.take(cfg.n_images)

    writer = DatasetShardWriter(cfg.target_repo, cache_features(qwen_hidden, cfg.pad_T), token,
                                rank=rank, shard_rows=SHARD_ROWS, private=cfg.private)

    def producer():                                            # background thread: decode + shard select
        gpos = 0
        for batch in stream.iter(batch_size=cfg.batch_size):
            caps, pil, addrs = batch["caption"], batch["image"], batch["aleph_address"]
            has_id = "id" in batch
            n = len(caps); base = gpos; gpos += n
            owned = range(n) if ws == 1 else [j for j in range(n) if (base + j) % ws == rank]
            idxs = list(owned)
            if not idxs:
                yield None
                continue
            rids = [str(batch["id"][j]) if has_id else f"row{rid_offset + base + j:08d}" for j in idxs]
            imgs = torch.stack([to_tensor(pil[j].convert("RGB")) for j in idxs])
            yield (rids, [caps[j] for j in idxs], imgs,
                   [np.asarray(addrs[j], dtype=np.float16) for j in idxs])

    from tqdm.auto import tqdm
    shard_note = f" [rank {rank}/{ws}]" if ws > 1 else ""
    for item in tqdm(_threaded_prefetch(producer), desc=f"prepare-cache{shard_note}"):
        if item is None:
            continue
        rids, caps_o, imgs, addr_o = item
        imgs = (imgs * 2 - 1).to(device, vae_dt)
        lat = enc.vae_encode_latent(imgs)
        clipl = enc.encode_clip_l(caps_o)
        clipg, clipgp = enc.encode_clip_g(caps_o)
        qf = enc.encode_qwen_full(caps_o, pad_T=cfg.pad_T)
        for k, rid in enumerate(rids):
            writer.add({
                "rid": rid,
                "caption": caps_o[k],
                "gen_text": qf["gen_text"][k],
                "seq_len": int(qf["seq_len"][k]),
                "lat":    encode_f16(lat[k].float().cpu().numpy()),
                "clipl":  encode_f16(clipl[k].float().cpu().numpy()),
                "qpool":  encode_f16(qf["pooled"][k].numpy()),
                "clipg":  encode_f16(clipg[k].float().cpu().numpy()),
                "clipgp": encode_f16(clipgp[k].float().cpu().numpy()),
                "addr":   encode_f16(addr_o[k]),
                "qseq":   encode_f16(qf["qseq"][k].numpy()),
            })
    writer.close()

    if dist.is_main() and cfg.upload_meta:
        meta = cache_meta(cfg.source_repo, qwen_hidden, cfg.pad_T, cfg.vae_scale,
                          generate=cfg.components.qwen.generate,
                          max_new_tokens=cfg.components.qwen.max_new_tokens,
                          layer=cfg.components.qwen.layer)
        writer.upload_bytes(json.dumps(meta, indent=2).encode("utf-8"), "meta.json")
        writer.upload_bytes(_readme(cfg, qwen_hidden, meta).encode("utf-8"), "README.md")

    dist.barrier()
    print(f"  ✓ rank {rank}: {writer.rows_written} rows in {writer.k} shards -> {cfg.target_repo}")
    if dist.is_distributed() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return writer.rows_written


def _cfg_from_env() -> PrepareCacheConfig:
    e = os.environ.get
    comp = ComponentConfig()
    if e("GEOLIP_VAE_DTYPE"):
        comp.vae_dtype = e("GEOLIP_VAE_DTYPE")
    if e("GEOLIP_QWEN_GENERATE"):
        comp.qwen.generate = e("GEOLIP_QWEN_GENERATE") == "1"
    return PrepareCacheConfig(
        source_repo=e("GEOLIP_SOURCE_REPO", "AbstractPhil/sdxl-qwen-phase0"),
        target_repo=e("GEOLIP_TARGET_REPO", "AbstractPhil/sdxl-qwen-phase1-cache"),
        n_images=int(e("GEOLIP_N_IMAGES", "0")),
        append=e("GEOLIP_APPEND", "1") == "1",
        rid_offset=int(e("GEOLIP_RID_OFFSET", "-1")),
        components=comp,
    )


if __name__ == "__main__":
    prepare_cache(_cfg_from_env())
