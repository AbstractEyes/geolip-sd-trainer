# Multi-RunPod guide

How to run `geolip-sd-trainer` across **multiple pods simultaneously** for the two
scalable workloads:

1. **Feature extraction / prep** — `build_cache` shards the dataset across pods so the
   dominant Qwen/VAE encode cost splits ~`world_size` ways.
2. **Distributed training** — every pod trains one shared model, gradients averaged each
   step (so N pods converge to **one** model, not N divergent ones).

Single-pod runs behave exactly as before — none of this is engaged unless `WORLD_SIZE>1`.

---

## 1. Concepts

The system is driven by the standard env vars a launcher sets. `torchrun` sets them for
training; for standalone data-prep you set them yourself.

| var | meaning |
|-----|---------|
| `RANK` | global index of this process (0 … `WORLD_SIZE`-1) |
| `WORLD_SIZE` | total number of processes across all pods |
| `LOCAL_RANK` | index of this process on its node (which GPU to pin) |
| `MASTER_ADDR` / `MASTER_PORT` | rendezvous host/port (training only) |

The seam lives in `geolip_sd_trainer/dist.py` (`rank()`, `world_size()`, `is_main()`,
`barrier()`, `init_distributed()`, `wait_for_marker()`). It degrades to the trivial
single-process answer when `WORLD_SIZE` is unset.

### Shared storage is required

Multi-pod correctness depends on every pod seeing the **same** `cache_dir` and `out_dir`.
The code keeps these as plain paths, so any shared substrate works:

- **RunPod network volume** (recommended, simplest): mount the same volume on every pod
  and point `cache_dir` / `out_dir` at it. No code changes.
- **S3 / object store or HF dataset shards**: the file I/O sits behind a thin seam; a
  future backend can replace the local path without touching the sharding logic. For now,
  prefer a network volume.

If pods do **not** share storage, each must run its own full precompute and they cannot
collaborate on one model — you lose the scaling benefit.

### How coordination works

- **Data-prep**: each rank encodes only rows where `global_index % world_size == rank`,
  writes a per-rank manifest `ids_<n>_rank<r>.json`, then drops a `ids_<n>.rank<r>.done`
  marker. Rank 0 waits for all `.done` markers, merges the per-rank manifests (in rank
  order) into the canonical `ids_<n>.json`, and writes a `ids_<n>.merged` marker. Other
  ranks wait for `.merged`, then read the full manifest. Every shared-file write is atomic
  (`tmp` + `os.replace`).
- **Training**: a `DistributedSampler` partitions rows; gradients of the currently-trainable
  params are averaged across ranks after each `backward` (one coalesced all-reduce). All
  ranks start identical (same seed → identical front-end init + identical frozen UNet load,
  so no broadcast is needed). Only **rank 0** writes checkpoints / samples / FID and uploads
  to the Hub; other ranks wait at an end-of-epoch barrier.

---

## 2. Recipe A — parallel data extraction / prep (no training)

Run `build_cache` on each pod with `RANK`/`WORLD_SIZE` set. No `torchrun` and no process
group are needed; the pods coordinate purely through the marker files on the shared volume.

On pod *r* of *N* (e.g. N=8):

```bash
export WORLD_SIZE=8
export RANK=<r>                       # 0,1,2,...,7 — one per pod
export HF_TOKEN=hf_xxx                # if the dataset/repo is gated

python -c "from geolip_sd_trainer import Phase1Config, build_cache; \
build_cache(Phase1Config(dataset_repo='AbstractPhil/sdxl-qwen-phase0', \
cache_dir='/mnt/shared/phase1_cache'))"
```

- Each pod encodes 1/N of the rows; rank 0 merges and writes the canonical manifest.
- Idempotent: re-running skips rows already cached (per-row existence check). If the
  merged manifest already exists and every file is present, all pods short-circuit.
- **Throughput knobs** (set on the `Phase1Config` / its `components`):
  - `ComponentConfig(qwen=QwenConfig(generate=False))` — skips the two-shot Qwen
    `generate`, the single biggest per-row cost. Changes the cached features, so use a
    fresh `cache_dir`.
  - `ComponentConfig(vae_dtype="bf16")` — ~2× faster VAE encode (run the latent-parity
    check first; cache stores fp16 regardless).
  - `GEOLIP_LOAD_WORKERS=1` — force sequential encoder loading if concurrent CUDA
    placement misbehaves on your image.

> Note: in this modulo-sharded mode every pod still *streams* the full dataset (it only
> *encodes* its shard). The compute — not the stream — is the bottleneck, so you still get
> ~Nx speedup. A dataset with an `id` column enables a future `.shard()` to also skip the
> redundant stream download.

---

## 3. Recipe B — distributed training with torchrun

Use the bundled launcher [`run_phase1.py`](../run_phase1.py). Point cache/out at the
shared volume.

```bash
export GEOLIP_CACHE_DIR=/mnt/shared/phase1_cache
export GEOLIP_OUT_DIR=/mnt/shared/phase1_runs
export HF_TOKEN=hf_xxx

# on each node (node_rank 0..N-1); torchrun fans out nproc_per_node procs per node
torchrun --nproc_per_node=<gpus_per_node> --nnodes=<N> --node_rank=<r> \
         --master_addr=<head_ip> --master_port=29500 run_phase1.py
```

What happens:
1. Every rank runs `build_cache` first (sharded as in Recipe A) and barriers until the
   merged cache is ready — so a fresh cluster can prep **and** train in one command.
2. Each rank builds the model on its pinned GPU (`cuda:LOCAL_RANK`), wraps training with a
   `DistributedSampler`, and averages gradients each step.
3. Rank 0 evaluates (FID/KID + prompt grid), checkpoints, and uploads in the background;
   other ranks wait at the epoch barrier.

Effective batch size is `batch_size × WORLD_SIZE`. Resume is automatic and consistent:
every rank reloads the same checkpoint from the shared `out_dir`.

---

## 4. Recipe C — separate prep cluster, then train

1. Run **Recipe A** on a cheap/large pod fleet to fill the shared cache.
2. Run **Recipe B** (training) on a GPU fleet pointed at the same `cache_dir`. Because the
   merged manifest + files already exist, `build_cache` short-circuits instantly and
   training starts immediately.

---

## 5. Correctness notes

- **Identical start, no broadcast** — `torch.manual_seed(seed)` is the same on every rank,
  so the front-end init and the (deterministic) frozen UNet load match across ranks.
- **Manual gradient averaging** instead of `DistributedDataParallel` — this is robust to
  the Stage-A→B staged unfreeze (auto-DDP binds gradient hooks at construction and can't
  track a later unfreeze). All ranks transition stages at the same `gstep`, so the synced
  parameter set stays consistent.
- **Gradient clipping is post-sync**, so the clip factor is identical on every rank.
- **Determinism** — keep `WORLD_SIZE` fixed across a cache's lifetime if the dataset has no
  `id` column (the fallback row ids are global-index based and stable only at fixed
  `world_size`). A dataset with an `id` column is fully stable.

---

## 6. Troubleshooting

| symptom | fix |
|---------|-----|
| Training pods can't see prepped features | `cache_dir` isn't actually shared — mount the same network volume on every pod and set `GEOLIP_CACHE_DIR` to it. |
| Checkpoints overwrite / collide | Only rank 0 writes; ensure all ranks share one `out_dir` and run under one `WORLD_SIZE`. |
| NCCL timeout during an epoch | Rank 0's eval/checkpoint exceeded the collective timeout; raise `fid_start_epoch`, lower `fid_images`/`fid_sample_steps`, or increase the NCCL timeout (`init_process_group(timeout=...)`). |
| `Using device_map requires accelerate` | `pip install accelerate` (it's a dependency); without it the Qwen load falls back to the slower CPU→GPU path automatically. |
| Hang waiting for a `.merged` / `.done` marker | A prep rank died before writing its marker. Check that every `RANK` in `0..WORLD_SIZE-1` actually ran; re-launch the missing rank (idempotent). |
| Concurrent component load errors | `export GEOLIP_LOAD_WORKERS=1` to load encoders sequentially. |

---

## 7. Quick reference

```bash
# parallel data-prep, pod r of N
WORLD_SIZE=N RANK=r python -c "from geolip_sd_trainer import *; \
build_cache(Phase1Config(cache_dir='/mnt/shared/phase1_cache'))"

# distributed training, node r of N
GEOLIP_CACHE_DIR=/mnt/shared/phase1_cache GEOLIP_OUT_DIR=/mnt/shared/phase1_runs \
torchrun --nproc_per_node=G --nnodes=N --node_rank=r \
         --master_addr=HEAD --master_port=29500 run_phase1.py
```
