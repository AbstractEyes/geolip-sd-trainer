"""
geolip_sd_trainer.data — the data pipeline (cache prep, cache download/stream, generation).
===========================================================================================
Three RunPod-multi-GPU-friendly tools around the two dataset formats:

  * prepare_cache  — phase0 (image+caption+aleph) -> native encoders + Qwen seq -> the
                     11-column COLUMNAR training cache (mirrors AbstractPhil/sdxl-qwen-
                     phase1-cache), uploaded as zstd parquet shards of 256 rows.
  * download_cache — pull that prepared columnar cache from HF, either MATERIALIZED to the
                     local per-row .npz the trainer already reads, or STREAMED directly.
  * generate       — Qwen-Image-Lightning 4-step text->image + geolip-aleph-void address ->
                     the 3-column phase0 format, uploaded in chunks of 256.

All three shard work by RANK/WORLD_SIZE via geolip_sd_trainer.dist and coordinate through
the same marker-file dance as build_cache. The columnar contract lives in cache_format.py.
"""
from .cache_format import (
    SHARD_ROWS, PAD_T, COLUMN_ORDER, TENSOR_COLS, TRAINER_KEYS,
    make_specs, encode_f16, decode_f16, cache_features, cache_meta,
    phase0_features, decode_trainer_npz,
)

__all__ = [
    "SHARD_ROWS", "PAD_T", "COLUMN_ORDER", "TENSOR_COLS", "TRAINER_KEYS",
    "make_specs", "encode_f16", "decode_f16", "cache_features", "cache_meta",
    "phase0_features", "decode_trainer_npz",
]
