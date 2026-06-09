"""
download_cache.py — launcher for SCRIPT B (materialize the HF columnar cache locally).
=====================================================================================
Downloads AbstractPhil/sdxl-qwen-phase1-cache and writes the per-row .npz the trainer
reads (so Phase1Trainer.setup() skips re-encoding). Rank-sharded for multi-pod volumes.

Single pod:   python download_cache.py
Multi pod:    RANK=<r> WORLD_SIZE=<N> python download_cache.py
              (or torchrun … download_cache.py)

Configure via env: GEOLIP_CACHE_REPO, GEOLIP_CACHE_DIR, GEOLIP_N_IMAGES, HF_TOKEN.

For STREAM-DIRECT (no local copy), don't run this — instead train with
Phase1Config(cache_mode="hf_stream", hf_cache_repo=...). See docs/DATA_PIPELINE.md.
"""
from geolip_sd_trainer.data.download_cache import materialize, _cfg_from_env

if __name__ == "__main__":
    materialize(_cfg_from_env())
