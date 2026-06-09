# Data pipeline — cache prep, cache consume, generation

geolip-sd-trainer works with **two dataset formats**, and this pipeline covers both, all
sharded for multi-GPU RunPod fleets via `RANK`/`WORLD_SIZE` (the `geolip_sd_trainer.dist`
seam). Single-pod usage is unchanged — leave the env vars unset.

| dataset | what | format |
|---|---|---|
| **phase0** (`AbstractPhil/sdxl-qwen-phase0`) | raw source: rendered image + caption + aleph | `image`(Image 1024² JPEG), `caption`(str), `aleph_address`(Array2D **[32,128]** f16) |
| **phase1-cache** (`AbstractPhil/sdxl-qwen-phase1-cache`) | prepared training cache (heavy compute already done) | 11 columns; 7 tensor columns are raw little-endian **fp16 bytes** in parquet `binary` cols |

The prepared-cache contract (verified from the dataset's `meta.json`) lives in
[`geolip_sd_trainer/data/cache_format.py`](../geolip_sd_trainer/data/cache_format.py):

```
rid(str) · caption(str) · gen_text(str) · seq_len(int64) ·
lat[4,128,128] · clipl[77,768] · qpool[1024] · clipg[77,1280] ·
clipgp[1280] · addr[32,128] · qseq[512,1024]            # all fp16, qseq LEFT-padded
shard_rows = 256 · zstd parquet · qwen.layer=-1 · generate=True (max_new_tokens=64)
decode: np.frombuffer(row[k], dtype="<f2").reshape(shape).copy()
qseq real tokens: qseq[512 - seq_len:]
```

The local trainer cache (`<cache_dir>/<rid>.npz`) stores the **6** keys
`lat/clipl/qpool/clipg/clipgp/addr`; the HF cache adds `qseq/seq_len/gen_text/caption/rid`.

---

## Install

```bash
pip install "git+https://github.com/AbstractEyes/geolip-sd-trainer"          # core
pip install "geolip-sd-trainer[generate]"                                    # + Script C
# generate needs: diffusers, accelerate, and geolip-svae (the aleph-void code):
pip install diffusers accelerate "git+https://github.com/AbstractEyes/geolip-svae.git"
```

---

## Script A — `prepare_cache` (phase0 → columnar cache on HF)

Streams a phase0-format source, runs the owned native encoders (VAE latent, CLIP-L/-G seq
+ pooled) plus the rich Qwen pass (pooled **and** the full left-padded `qseq`, `seq_len`,
and the `gen_text` re-description), copies the pre-stored `aleph_address`, and uploads
256-row zstd parquet shards that mirror `sdxl-qwen-phase1-cache` exactly.

```bash
# single pod
GEOLIP_TARGET_REPO=<you>/sdxl-qwen-phase1-cache HF_TOKEN=hf_xxx python prepare_cache.py

# multi-pod: each rank encodes only its 1/N shard and writes disjoint parquet shards
RANK=<r> WORLD_SIZE=<N> GEOLIP_TARGET_REPO=<you>/sdxl-qwen-phase1-cache \
  HF_TOKEN=hf_xxx python prepare_cache.py
# or: torchrun --nproc_per_node=<gpus> --nnodes=<N> --node_rank=<r> \
#              --master_addr=<host> --master_port=29500 prepare_cache.py
```

- **Append vs new:** defaults to append (`GEOLIP_APPEND=1`) — `rid` continues past the
  target's current row count (`rid_offset` auto-detected, override with `GEOLIP_RID_OFFSET`)
  and shards are rank-namespaced so they never collide with existing `shard_*.parquet`.
  Point `GEOLIP_TARGET_REPO` at a fresh repo for a clean build.
- **Throughput:** `GEOLIP_QWEN_GENERATE=0` skips the two-shot re-description (the biggest
  per-row cost — but then `gen_text` = caption and `qseq`/`qpool` encode the caption
  directly, changing the cache); `GEOLIP_VAE_DTYPE=bf16` halves VAE cost.
- Rank 0 also uploads `meta.json` + `README.md` (the decode contract).
- Programmatic: `from geolip_sd_trainer.data.prepare_cache import prepare_cache, PrepareCacheConfig`.

---

## Script B — consume the prepared cache (two modes)

### B1. Materialize → local `.npz` (recommended for a shared volume)

Downloads the columnar cache, decodes the 6 trainer columns, and writes the per-row `.npz`
the trainer already reads. After this, `Phase1Trainer.setup()` finds the manifest + files
and **skips re-encoding entirely**. Rank-sharded (`.shard(num_shards, index)` over the
parquet files — safe because every row carries an explicit `rid`); rank 0 merges manifests.

```bash
GEOLIP_CACHE_REPO=AbstractPhil/sdxl-qwen-phase1-cache \
GEOLIP_CACHE_DIR=/mnt/shared/phase1_cache python download_cache.py        # single pod
RANK=<r> WORLD_SIZE=<N> GEOLIP_CACHE_DIR=/mnt/shared/phase1_cache python download_cache.py
```

Then train against it (it short-circuits):
```python
train(Phase1Config(cache_mode="hf_materialize", hf_cache_repo="AbstractPhil/sdxl-qwen-phase1-cache",
                   cache_dir="/mnt/shared/phase1_cache"))
```

### B2. Stream-direct (no `.npz` copy)

Train straight off the HF cache — a map-style dataset decodes the 6 binary columns on the
fly; the training `DistributedSampler` shards it per rank. Downloads the parquet once
(~41 GB, memory-mapped via Arrow), no 130 GB local materialization.

```python
train(Phase1Config(cache_mode="hf_stream", hf_cache_repo="AbstractPhil/sdxl-qwen-phase1-cache"))
```

`cache_mode="local"` (default) keeps the original behavior: encode phase0 into `cache_dir`.

---

## Script C — `generate` (Qwen-Image-Lightning → new phase0 rows)

Generates new rows from a caption source and appends them to a phase0-format dataset in
chunks of 256: 4-step Qwen-Image-Lightning render → resize to 1024² JPEG → aleph address
from the caption via geolip-aleph-void.

```bash
# captions from a local file (one per line, or a .json list)
GEOLIP_TARGET_REPO=<you>/sdxl-qwen-phase0 GEOLIP_PROMPTS_PATH=captions.txt \
  HF_TOKEN=hf_xxx python generate_data.py

# captions from an HF dataset column, multi-GPU (one pipeline per GPU)
RANK=<r> WORLD_SIZE=<N> GEOLIP_TARGET_REPO=<you>/sdxl-qwen-phase0 \
  GEOLIP_PROMPTS_REPO=conceptual_captions GEOLIP_PROMPTS_COLUMN=caption \
  HF_TOKEN=hf_xxx python generate_data.py
# or torchrun … generate_data.py  (one process per GPU)
```

- **Models:** base `Qwen/Qwen-Image` + `lightx2v/Qwen-Image-Lightning`
  (`Qwen-Image-Lightning-4steps-V2.0.safetensors`), `num_inference_steps=4`,
  `true_cfg_scale=1.0`, Lightning `FlowMatchEulerDiscreteScheduler`. Render 1328² → 1024².
- **Aleph:** [`data/aleph.py`](../geolip_sd_trainer/data/aleph.py) loads
  `AbstractPhil/geolip-aleph-void` (`hf_version="aleph_byte_trigram_tied_hard_K64"`) via the
  `geolip-svae` package, renders the caption to a byte-trigram image (`text_to_image`), and
  reads the `[32,128]` address from the model. The exact tensor/aggregation is configurable
  (`AlephConfig.source` / `aggregate` / `post`) so you can pin it to match the existing 86k
  rows; defaults to `aleph_logits` + patch-mean. **If generated addresses don't match
  phase0's value distribution (~[-1,1]), set `post="tanh"`** or adjust the checkpoint.
- **Multi-GPU:** Qwen-Image has no tensor-parallel — each rank runs a full pipeline on
  `cuda:LOCAL_RANK`, takes `prompts[rank::world_size]`, and writes disjoint rank-namespaced
  shards. Set `GEOLIP_CPU_OFFLOAD=1` for smaller-VRAM GPUs.
- Programmatic: `from geolip_sd_trainer.data.generate import generate, GenerateConfig`.

---

## End-to-end on a RunPod fleet

```
prompts ──Script C (N GPUs)──▶ phase0 (image+caption+aleph)
                                   │
        ┌──────────────────────────┘
        ▼
   Script A (N GPUs) ──▶ phase1-cache (columnar, on HF)
        │
        ▼
   Script B materialize (N GPUs) ──▶ /mnt/shared/phase1_cache (.npz)
        │
        ▼
   torchrun run_phase1.py (N GPUs) ──▶ trained UNet     (see docs/MULTIPOD.md)
```

Every stage shards by `RANK`/`WORLD_SIZE`, coordinates via marker files / `dist.barrier()`,
and only rank 0 writes shared artifacts (manifests, `meta.json`, checkpoints).

## Verifying the format (no GPU)

`cache_format.encode_f16` / `decode_f16` round-trip and the `Features`/`meta.json` match the
reference contract. Spot-check against the live dataset:

```python
import numpy as np
from datasets import load_dataset
from geolip_sd_trainer.data.cache_format import decode_trainer_npz
row = load_dataset("AbstractPhil/sdxl-qwen-phase1-cache", split="train").with_format(None)[0]
six = decode_trainer_npz(row)              # {lat, clipl, qpool, clipg, clipgp, addr}
assert six["lat"].shape == (4, 128, 128) and six["addr"].shape == (32, 128)
real = np.frombuffer(row["qseq"], "<f2").reshape(512, -1)[512 - row["seq_len"]:]
```
