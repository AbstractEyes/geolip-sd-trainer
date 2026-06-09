"""
data/generate.py — SCRIPT C: Qwen-Image-Lightning 4-step generation -> phase0 format.
=====================================================================================
Generates new phase0 rows (image + caption + aleph_address[32,128]) from a list of
captions and appends them to a phase0-format HF dataset in chunks of 256.

  caption --Qwen-Image + Lightning 4-step LoRA--> 1328² image -> resize 1024² JPEG
          --geolip-aleph-void (data/aleph.py)----> aleph_address [32,128] fp16

Multi-GPU: data-parallel, ONE full pipeline per GPU (Qwen-Image has no tensor-parallel).
Each rank pins cuda:{local_rank}, takes prompts[rank::world_size], and writes its own
rank-namespaced parquet shards (append-only, disjoint) — so N RunPods extend the dataset
in parallel with no collisions. Seeds are global+reproducible.

Install:  pip install diffusers "git+https://github.com/AbstractEyes/geolip-svae.git"
Run:      RANK=<r> WORLD_SIZE=<N> python -m geolip_sd_trainer.data.generate
          (or torchrun … -m geolip_sd_trainer.data.generate)

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

from .. import dist
from ..checkpoint import resolve_hf_token, hf_whoami
from .aleph import AlephConfig, AlephEncoder
from .cache_format import SHARD_ROWS, phase0_features
from .hf_io import DatasetShardWriter

# Lightning 4-step scheduler config (lightx2v / ModelTC Qwen-Image-Lightning)
_LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
    "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000,
    "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
    "time_shift_type": "exponential", "use_beta_sigmas": False, "use_dynamic_shifting": True,
    "use_exponential_sigmas": False, "use_karras_sigmas": False,
}


@dataclass
class GenerateConfig:
    target_repo: str = "AbstractPhil/sdxl-qwen-phase0"     # phase0-format output (append)
    # prompt source (one of): a local file (.txt one-per-line / .json list), or an HF dataset
    prompts_path: Optional[str] = None
    prompts_repo: Optional[str] = None
    prompts_column: str = "caption"
    n_prompts: int = 0                                     # 0 = all
    # generation
    base_model: str = "Qwen/Qwen-Image"
    lora_repo: str = "lightx2v/Qwen-Image-Lightning"
    lora_weight: str = "Qwen-Image-Lightning-4steps-V2.0.safetensors"
    num_inference_steps: int = 4
    true_cfg_scale: float = 1.0
    gen_width: int = 1328
    gen_height: int = 1328
    out_size: int = 1024                                   # phase0 images are 1024² JPEG
    positive_magic: str = ", Ultra HD, 4K, cinematic composition."
    negative_prompt: str = " "
    seed_base: int = 0
    jpeg_quality: int = 95
    cpu_offload: bool = False                              # enable for smaller-VRAM GPUs
    # aleph
    aleph: AlephConfig = field(default_factory=AlephConfig)
    # io
    private: bool = False
    device: str = "cuda"
    token: Optional[str] = None


def _load_prompts(cfg: GenerateConfig, token: Optional[str]) -> List[str]:
    if cfg.prompts_path:
        p = Path(cfg.prompts_path)
        text = p.read_text(encoding="utf-8")
        prompts = json.loads(text) if p.suffix == ".json" else [ln.strip() for ln in text.splitlines()]
        prompts = [s for s in prompts if s]
    elif cfg.prompts_repo:
        from datasets import load_dataset
        ds = load_dataset(cfg.prompts_repo, split="train", streaming=True, token=token)
        prompts = []
        for row in ds:
            prompts.append(row[cfg.prompts_column])
            if cfg.n_prompts and len(prompts) >= cfg.n_prompts:
                break
    else:
        raise SystemExit("generate: set prompts_path (file) or prompts_repo (HF dataset).")
    if cfg.n_prompts:
        prompts = prompts[:cfg.n_prompts]
    return prompts


def _build_pipe(cfg: GenerateConfig, device: str):
    from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(_LIGHTNING_SCHEDULER)
    pipe = DiffusionPipeline.from_pretrained(cfg.base_model, scheduler=scheduler,
                                             torch_dtype=torch.bfloat16, token=cfg.token)
    pipe.load_lora_weights(cfg.lora_repo, weight_name=cfg.lora_weight)
    if cfg.cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=int(device.split(":")[-1]) if ":" in device else 0)
    else:
        pipe = pipe.to(device)
    return pipe


def _jpeg_dict(img, quality: int) -> dict:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return {"bytes": buf.getvalue(), "path": None}         # datasets.Image() encoding


def generate(cfg: Optional[GenerateConfig] = None) -> int:
    cfg = cfg or GenerateConfig()
    local_rank = dist.init_distributed(cfg.device)
    rank, ws = dist.rank(), dist.world_size()
    device = cfg.device
    if ws > 1 and cfg.device.startswith("cuda") and torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    cfg.aleph.device = device

    token = cfg.token or resolve_hf_token()
    cfg.token = token
    if dist.is_main() and token:
        print(f"  HF user: {hf_whoami(token)}")

    prompts = _load_prompts(cfg, token)
    mine = prompts[rank::ws]                                # disjoint per-rank shard
    if dist.is_main():
        print(f"  generate: {len(prompts)} prompts -> {cfg.target_repo} (world_size={ws})")

    pipe = _build_pipe(cfg, device)
    aleph = AlephEncoder(cfg.aleph, device=device, token=token)
    writer = DatasetShardWriter(cfg.target_repo, phase0_features(), token, rank=rank,
                                shard_prefix="train-shard", shard_rows=SHARD_ROWS, private=cfg.private)

    from tqdm.auto import tqdm
    shard_note = f" [rank {rank}/{ws}]" if ws > 1 else ""
    for i, caption in enumerate(tqdm(mine, desc=f"generate{shard_note}", disable=not dist.is_main())):
        seed = cfg.seed_base + (rank + i * ws)              # global, reproducible per prompt
        img = pipe(prompt=caption + cfg.positive_magic, negative_prompt=cfg.negative_prompt,
                   width=cfg.gen_width, height=cfg.gen_height,
                   num_inference_steps=cfg.num_inference_steps, true_cfg_scale=cfg.true_cfg_scale,
                   generator=torch.Generator(device=device).manual_seed(seed)).images[0]
        if cfg.out_size and img.size != (cfg.out_size, cfg.out_size):
            img = img.resize((cfg.out_size, cfg.out_size))
        addr = aleph.caption_to_aleph([caption])[0]        # [32,128] fp16
        writer.add({"image": _jpeg_dict(img, cfg.jpeg_quality),
                    "caption": caption, "aleph_address": addr})
    writer.close()
    dist.barrier()
    print(f"  ✓ rank {rank}: generated {writer.rows_written} rows in {writer.k} shards -> {cfg.target_repo}")
    if dist.is_distributed() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return writer.rows_written


def _cfg_from_env() -> GenerateConfig:
    e = os.environ.get
    return GenerateConfig(
        target_repo=e("GEOLIP_TARGET_REPO", "AbstractPhil/sdxl-qwen-phase0"),
        prompts_path=e("GEOLIP_PROMPTS_PATH"),
        prompts_repo=e("GEOLIP_PROMPTS_REPO"),
        prompts_column=e("GEOLIP_PROMPTS_COLUMN", "caption"),
        n_prompts=int(e("GEOLIP_N_PROMPTS", "0")),
        cpu_offload=e("GEOLIP_CPU_OFFLOAD", "0") == "1",
    )


if __name__ == "__main__":
    generate(_cfg_from_env())
