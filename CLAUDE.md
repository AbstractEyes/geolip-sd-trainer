# CLAUDE.md

Operational guide for running `geolip-sd-trainer` in **Google Colab / Jupyter** and on
**RunPod** (single pod and multiple pods). Read this before changing training, caching,
or distributed code.

## What this is

A pure-PyTorch SDXL trainer. UNet / VAE / CLIP-L / CLIP-G are reimplemented natively
(no diffusers at runtime) and load real SDXL weights via `load_state_dict(strict=True)`.
Qwen3.5 is a **frozen** HF encoder used for rich pooled text features. The trainable
surface is a small front-end (`SDXLQwenFrontEnd`, ~2.4M params) plus the UNet. The
objective is Lune rectified-flow. Primary doc: [README.md](README.md); multi-pod:
[docs/MULTIPOD.md](docs/MULTIPOD.md).

## Layout

```
geolip_sd_trainer/
  transformer/unet.py  unet_test.py       native SDXL denoiser + loader/parity
  vae/vae.py           vae_test.py         native SDXL VAE + loader/parity
  text_encoder/clip_text.py  *_test.py     CLIP-L + CLIP-G (shared module)
  vlm/qwen.py                              frozen Qwen3.5 pooled encoder (HF)
  model.py                                 GeolipSDXL + SDXLQwenFrontEnd + prefetch_models
  trainer.py                               Phase1 trainer: build_cache, fit, eval, sampler
  checkpoint.py                            save/load/resume (safetensors) + HF upload
  dist.py                                  multi-pod seam (rank/world_size/barrier/...)
run_phase1.py                              single/multi-pod launcher (env-configurable)
tests/test_optimizations.py               fast CPU checks (cache + checkpoint round-trips)
```

## Environment setup (both Colab and RunPod)

- **transformers v5 is mandatory** for Qwen3.5 (4.x cannot load it). The preflight
  (`QwenConfig.min_transformers >= 5.2.0`) fails fast with the upgrade command. On Colab,
  after upgrading transformers you **must restart the runtime**.
- **Blackwell (sm_120)** GPUs need a torch build for **CUDA 12.8 (`cu128`)**.
- `accelerate` is a dependency — it enables `device_map` so Qwen loads straight onto the
  GPU (no CPU round-trip / RAM spike). Without it, the loader falls back automatically.
- Auth: set `HF_TOKEN` in the environment (or Colab `userdata`). `resolve_hf_token()`
  reads env then Colab userdata; `hf_whoami()` is a fail-fast preflight (a 401 means an
  **invalid token**, not a missing scope).

## Operating in Colab / Jupyter (single GPU)

The disk is wiped on restart, so the first run otherwise pays a long serial cold
download. Do this at the top of the notebook:

```python
from geolip_sd_trainer import prefetch_models
prefetch_models()                       # parallel HF download of all components
```

Then:

```python
from geolip_sd_trainer import build_sdxl, PHASE1_RECIPE
model = build_sdxl(PHASE1_RECIPE)       # meta-device load, components load concurrently
```

Train (precompute runs first, idempotently):

```python
from geolip_sd_trainer import Phase1Config, train
train(Phase1Config(num_epochs=60, batch_size=4, upload_to_hub=True))
```

Colab tips:
- **Memory** — `vae_dtype` defaults to `fp32` (safe). The cold-start path already avoids
  the CPU random-init double-allocation (meta load) and the Qwen CPU→GPU spike
  (`device_map`). For very tight boxes, set `GEOLIP_LOAD_WORKERS=1`.
- **Smoke test** — `Phase1Config(n_images=64, num_epochs=1, upload_to_hub=False)`.
- **Persist the cache** — point `cache_dir` at a mounted Drive folder so precompute
  survives a restart, or pre-build it once and reuse.
- **Quick iteration** — raise `fid_start_epoch` and lower `fid_sample_steps` /
  `prompt_grid_steps` to spend less time in eval.

## Operating on RunPod

Single pod is identical to Colab but with persistent disk (HF cache survives, so
`prefetch_models` is a one-time cost). For **multiple pods**, see
[docs/MULTIPOD.md](docs/MULTIPOD.md). Essentials:

- Mount a **shared network volume** and point `cache_dir` and `out_dir` at it
  (`GEOLIP_CACHE_DIR`, `GEOLIP_OUT_DIR`). All pods must see the same paths.
- **Parallel data-prep**: set `RANK`/`WORLD_SIZE` per pod and call `build_cache(...)`
  (no `torchrun` needed; pods coordinate via marker files).
- **Distributed training**:
  ```bash
  GEOLIP_CACHE_DIR=/mnt/shared/cache GEOLIP_OUT_DIR=/mnt/shared/runs \
  torchrun --nproc_per_node=<gpus> --nnodes=<N> --node_rank=<r> \
           --master_addr=<head> --master_port=29500 run_phase1.py
  ```
- Only **rank 0** checkpoints / samples / uploads; effective batch is
  `batch_size × WORLD_SIZE`.

## Key activation points

- `prefetch_models(components=ALL_COMPONENTS)` — warm the HF cache in parallel.
- `build_sdxl(preset, load=...)` / `GeolipSDXL(cfg, load=...)` — assemble a model.
- `build_cache(cfg)` — stream + cache features (one `.npz` per row); rank-sharded.
- `train(cfg)` / `Phase1Trainer` — precompute → fit, automatic resume.
- `save_checkpoint` / `load_checkpoint` / `HubUploader` / `export_unet_safetensors`.
- `geolip_sd_trainer.dist` — `rank()/world_size()/is_main()/barrier()`.

## Hard constraints — do NOT break

- **CLIP-G sequence stays real** in phase 1 (`swap_clip_g_seq=False`,
  `geolip_sd_trainer/model.py`). Never "optimize" it to the Qwen projection.
- **VAE dtype** — only *fp16* is unstable (SDXL `force_upcast`); `fp32` is the verified
  default. `bf16` is safe for frozen inference but **gate any default change behind the
  latent-parity check** (see `tests/` + README). Don't flip it blind.
- **Qwen stays deterministic** (`do_sample=False`, greedy) and reproducible — required
  for cache consistency across pods.
- **Resume backward compatibility** — the checkpoint format is now `*.safetensors` +
  `.meta.json` (+ opt `.opt.pt`); the loader must keep reading legacy `.pt`.
- **Same seed across ranks** — multi-pod relies on identical init without a broadcast;
  don't make per-rank-seeded changes to model construction.

## Formats & behavior to know

- **Cache**: one `.npz` per row at `<cache_dir>/<row_id>.npz` (fp16, keys
  `lat/clipl/qpool/clipg/clipgp/addr`). Older builds used six `.npy` per row — wipe the
  old cache dir when upgrading. `CachedDS` returns tensors as **fp16**; the train step
  casts to bf16 on-GPU.
- **Checkpoints**: `ckpt_e####.safetensors` (+ `.meta.json`, + `.opt.pt` only when
  `save_optimizer_state=True`). Optimizer state is **opt-out by default** — full-finetune
  resume re-warms AdamW cheaply.
- **Throughput levers** (defaults are safe): `compile_unet`, `use_channels_last`,
  `num_workers`, `fid_start_epoch`, `fid_sample_batch_size`, `prompt_grid_steps`,
  `ComponentConfig(qwen=QwenConfig(generate=False))`, `ComponentConfig(vae_dtype="bf16")`.

## Verifying changes

- Fast, no GPU/network: `python tests/test_optimizations.py` (cache + checkpoint
  round-trips, dist defaults).
- GPU/parity-gated (run on a pod): VAE bf16 latent parity, Qwen `generate=False`
  equivalence, batched-CFG image parity, a `n_images=64` smoke train, and a
  `WORLD_SIZE=2` `torchrun` dry run (disjoint shards, rank-0 merge == single-pod id set,
  rank-0-only writes). Commands are printed by the test script.

## Gotchas

- This is a CUDA training package; there is no working CPU-only path for full runs.
- `build_cache` streams the whole dataset on every pod (modulo-shards the *encode*, not
  the *download*) — the encode is the bottleneck, so scaling still holds.
- A long rank-0 eval can trip the NCCL collective timeout during distributed training —
  gate eval (`fid_start_epoch`) or raise the timeout.
- Don't add per-step `.item()`/`.cpu()` syncs in the hot loop; the current loop syncs
  once per step on purpose.
