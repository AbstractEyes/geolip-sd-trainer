"""
prepare_cache.py — launcher for SCRIPT A (phase0 -> columnar training cache on HF).
==================================================================================
Single pod:   python prepare_cache.py
Multi pod:    torchrun --nproc_per_node=<gpus> --nnodes=<N> --node_rank=<r> \
                       --master_addr=<host> --master_port=<port> prepare_cache.py
              (or: RANK=<r> WORLD_SIZE=<N> python prepare_cache.py  — no torchrun needed)

Configure via env: GEOLIP_SOURCE_REPO, GEOLIP_TARGET_REPO, GEOLIP_N_IMAGES,
GEOLIP_APPEND, GEOLIP_RID_OFFSET, GEOLIP_VAE_DTYPE, GEOLIP_QWEN_GENERATE, HF_TOKEN.
See docs/DATA_PIPELINE.md.
"""
from geolip_sd_trainer.data.prepare_cache import prepare_cache, _cfg_from_env

if __name__ == "__main__":
    prepare_cache(_cfg_from_env())
