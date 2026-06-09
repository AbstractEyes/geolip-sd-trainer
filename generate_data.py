"""
generate_data.py — launcher for SCRIPT C (Qwen-Image-Lightning -> phase0 rows on HF).
=====================================================================================
Generates new image+caption+aleph rows from a caption source and appends them to a
phase0-format dataset, multi-GPU, in chunks of 256.

Single pod:   GEOLIP_PROMPTS_PATH=captions.txt python generate_data.py
Multi pod:    RANK=<r> WORLD_SIZE=<N> GEOLIP_PROMPTS_REPO=<ds> python generate_data.py
              (or torchrun … generate_data.py — one process per GPU)

Configure via env: GEOLIP_TARGET_REPO, GEOLIP_PROMPTS_PATH | GEOLIP_PROMPTS_REPO,
GEOLIP_PROMPTS_COLUMN, GEOLIP_N_PROMPTS, GEOLIP_CPU_OFFLOAD, HF_TOKEN.

Requires:  pip install diffusers "git+https://github.com/AbstractEyes/geolip-svae.git"
See docs/DATA_PIPELINE.md.
"""
from geolip_sd_trainer.data.generate import generate, _cfg_from_env

if __name__ == "__main__":
    generate(_cfg_from_env())
