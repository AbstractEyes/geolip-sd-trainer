"""
data/cache_format.py — the columnar training-cache contract.
============================================================
Single source of truth for the on-HF format of AbstractPhil/sdxl-qwen-phase1-cache
(86k rows, zstd parquet, 256 rows/shard). Verified against the dataset's meta.json:

  shard_rows = 256 · pad_T = 512 · dtype fp16 · qwen.layer = -1 · generate = True

11 columns, in this order:

  rid       Value(string)   "row{global_index:08d}" (or the source dataset 'id')
  caption   Value(string)   the original source caption
  gen_text  Value(string)   the two-shot Qwen re-description that was actually encoded
  seq_len   Value(int64)    real (non-pad) Qwen token count
  lat       Value(binary)   VAE latent           fp16 [4,128,128]  (vae_scale pre-applied)
  clipl     Value(binary)   CLIP-L penultimate   fp16 [77,768]
  qpool     Value(binary)   Qwen last-token pool fp16 [hidden]      (hidden=1024)
  clipg     Value(binary)   CLIP-G penultimate   fp16 [77,1280]
  clipgp    Value(binary)   CLIP-G pooled        fp16 [1280]
  addr      Value(binary)   geolip aleph address fp16 [32,128]
  qseq      Value(binary)   Qwen layer-(-1) seq  fp16 [512,1024] LEFT-PADDED (real at tail)

Binary columns hold raw little-endian float16 bytes (numpy .tobytes()); they carry no
shape metadata, so the (dtype, shape) SPECS here (and the uploaded meta.json) are the only
decode contract. The 6 keys lat/clipl/qpool/clipg/clipgp/addr map 1:1 to the local trainer
.npz; qseq/seq_len/gen_text/caption/rid are the extra columnar-cache columns.

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

SHARD_ROWS = 256                 # rows per parquet shard (matches meta.json)
PAD_T = 512                      # qseq left-pad length (matches meta.json pad_T)
VAE_SCALE = 0.13025              # latent scale already applied to `lat`

# (col -> fixed shape) for everything except the hidden-sized qpool/qseq, filled by make_specs
_FIXED_SHAPES: Dict[str, Tuple[int, ...]] = {
    "lat": (4, 128, 128),
    "clipl": (77, 768),
    "clipg": (77, 1280),
    "clipgp": (1280,),
    "addr": (32, 128),
}

COLUMN_ORDER = ["rid", "caption", "gen_text", "seq_len",
                "lat", "clipl", "qpool", "clipg", "clipgp", "addr", "qseq"]
TENSOR_COLS = ["lat", "clipl", "qpool", "clipg", "clipgp", "addr", "qseq"]
TEXT_COLS = ["rid", "caption", "gen_text"]
INT_COLS = ["seq_len"]
# the 6 features the trainer's CachedDS consumes (qseq/gen_text/etc. are extra)
TRAINER_KEYS = ["lat", "clipl", "qpool", "clipg", "clipgp", "addr"]


def make_specs(qwen_hidden: int = 1024, pad_T: int = PAD_T) -> Dict[str, Tuple[type, Tuple[int, ...]]]:
    """{col: (np.float16, shape)} for every tensor column. qpool/qseq depend on qwen_hidden."""
    specs = {k: (np.float16, shp) for k, shp in _FIXED_SHAPES.items()}
    specs["qpool"] = (np.float16, (qwen_hidden,))
    specs["qseq"] = (np.float16, (pad_T, qwen_hidden))
    return specs


def encode_f16(arr) -> bytes:
    """Array -> raw little-endian fp16 bytes (the parquet binary-column encoding)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f2"))
    return a.tobytes()


def decode_f16(buf: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """Raw fp16 bytes -> fp16 ndarray of `shape` (writable copy, safe for torch.from_numpy)."""
    return np.frombuffer(buf, dtype="<f2").reshape(shape).copy()


def cache_features(qwen_hidden: int = 1024, pad_T: int = PAD_T):
    """datasets.Features for the 11-column cache (binary tensor cols + string/int cols),
    built in COLUMN_ORDER so the parquet column order matches the reference dataset."""
    from datasets import Features, Value
    feats = {}
    for col in COLUMN_ORDER:
        if col in TEXT_COLS:
            feats[col] = Value("string")
        elif col in INT_COLS:
            feats[col] = Value("int64")
        else:
            feats[col] = Value("binary")
    return Features(feats)


def cache_meta(source_dataset: str, qwen_hidden: int = 1024, pad_T: int = PAD_T,
               vae_scale: float = VAE_SCALE, generate: bool = True,
               max_new_tokens: int = 64, layer: int = -1) -> dict:
    """The meta.json written alongside the shards (the canonical decode contract)."""
    specs = make_specs(qwen_hidden, pad_T)
    return {
        "dataset": source_dataset,
        "shard_rows": SHARD_ROWS,
        "pad_T": pad_T,
        "dtype": "fp16",
        "byte_order": "little-endian",
        "vae_scale": vae_scale,
        "arrays": {k: {"dtype": "float16", "shape": list(shp)} for k, (_, shp) in specs.items()},
        "text_cols": TEXT_COLS,
        "int_cols": INT_COLS,
        "column_order": COLUMN_ORDER,
        "qwen": {"layer": layer, "generate": generate, "max_new_tokens": max_new_tokens,
                 "seq_padding": "left (real tokens at tail; slice seq[pad_T-seq_len:])"},
        "decode": "np.frombuffer(row[key], dtype='<f2').reshape(arrays[key].shape).copy()",
        "rid_scheme": "dataset 'id' or row{global_index:08d}",
    }


def decode_trainer_npz(row: dict) -> Dict[str, np.ndarray]:
    """Decode just the 6 trainer keys (lat/clipl/qpool/clipg/clipgp/addr) from a cache row
    into the fp16 arrays the local .npz / CachedDS expects. qpool's hidden dim is inferred
    from its byte length, so no out-of-band hidden size is needed."""
    out = {k: decode_f16(row[k], _FIXED_SHAPES[k]) for k in ("lat", "clipl", "clipg", "clipgp", "addr")}
    out["qpool"] = np.frombuffer(row["qpool"], dtype="<f2").copy()       # (hidden,)
    return out


def phase0_features():
    """datasets.Features for the raw phase0 format: image / caption / aleph_address[32,128]."""
    from datasets import Features, Value, Image, Array2D
    return Features({
        "image": Image(),
        "caption": Value("string"),
        "aleph_address": Array2D(shape=(32, 128), dtype="float16"),
    })
