"""
data/hf_io.py — writing/reading HF *dataset* repos (parquet shards).
====================================================================
geolip_sd_trainer.checkpoint.HubUploader targets MODEL repos with a fixed
<phase>/<run> layout, so it is not reusable here. This module provides:

  * DatasetShardWriter — buffer rows, flush a zstd parquet shard every `shard_rows`
    and upload it to a dataset repo; rank-namespaced shard names so concurrent pods
    never collide; append-friendly (writes new shards next to existing ones).
  * count_remote_rows — current row count of a dataset (for the append rid offset).
  * ensure_dataset_repo — idempotent create_repo(repo_type="dataset").

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import List, Optional

from .cache_format import SHARD_ROWS


def ensure_dataset_repo(repo_id: str, token: Optional[str] = None, private: bool = False) -> str:
    from huggingface_hub import create_repo
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token, private=private)
    return repo_id


def count_remote_rows(repo_id: str, token: Optional[str] = None) -> int:
    """Best-effort current num_examples of an HF dataset's train split (for append rid
    offsets). Returns 0 if the dataset doesn't exist yet or the query fails."""
    try:
        import requests
        url = f"https://datasets-server.huggingface.co/info?dataset={repo_id}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        info = requests.get(url, headers=headers, timeout=30).json().get("dataset_info", {})

        def _find_train(obj):                              # recursive: first splits.train.num_examples
            if isinstance(obj, dict):
                sp = obj.get("splits")
                if isinstance(sp, dict) and "train" in sp:
                    n = sp["train"].get("num_examples")
                    if n is not None:
                        return int(n)
                for v in obj.values():
                    r = _find_train(v)
                    if r is not None:
                        return r
            return None
        return _find_train(info) or 0
    except Exception:
        return 0


class DatasetShardWriter:
    """Append-only sharded parquet writer for a single HF dataset repo + split.

    Each rank writes shards named `<subdir>/<prefix>_r<rank>_<k>.parquet` so N pods append
    disjoint, non-colliding shards. Rows are buffered and flushed in groups of `shard_rows`.
    """

    def __init__(self, repo_id: str, features, token: Optional[str], *, subdir: str = "data",
                 shard_prefix: str = "shard", rank: int = 0, shard_rows: int = SHARD_ROWS,
                 retries: int = 5, private: bool = False, compression: str = "zstd"):
        self.repo_id = repo_id
        self.features = features
        self.token = token
        self.subdir = subdir
        self.prefix = shard_prefix
        self.rank = rank
        self.shard_rows = shard_rows
        self.retries = retries
        self.private = private
        self.compression = compression
        self.buf: List[dict] = []
        self.k = 0
        self.rows_written = 0
        self._repo_ready = False

    def _ensure(self):
        if not self._repo_ready:
            ensure_dataset_repo(self.repo_id, self.token, self.private)
            self._repo_ready = True

    def add(self, row: dict):
        self.buf.append(row)
        if len(self.buf) >= self.shard_rows:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        from datasets import Dataset
        rows, self.buf = self.buf, []
        name = f"{self.prefix}_r{self.rank:02d}_{self.k:06d}.parquet"
        path_in_repo = f"{self.subdir}/{name}"
        cols = {c: [r[c] for r in rows] for c in self.features}
        ds = Dataset.from_dict(cols, features=self.features)
        tmp = Path(tempfile.gettempdir()) / f"geolip_{self.repo_id.replace('/', '_')}_{name}"
        try:
            ds.to_parquet(str(tmp), compression=self.compression)    # zstd, like the reference
        except TypeError:                                            # older datasets: no passthrough
            ds.to_parquet(str(tmp))
        self._upload(str(tmp), path_in_repo)
        tmp.unlink(missing_ok=True)
        self.k += 1
        self.rows_written += len(rows)

    def upload_bytes(self, data: bytes, path_in_repo: str):
        """Upload an auxiliary file (meta.json / README.md). Rank 0 only, typically."""
        tmp = Path(tempfile.gettempdir()) / ("geolip_aux_" + path_in_repo.replace("/", "_"))
        tmp.write_bytes(data)
        self._upload(str(tmp), path_in_repo)
        tmp.unlink(missing_ok=True)

    def _upload(self, local: str, path_in_repo: str):
        from huggingface_hub import HfApi
        for attempt in range(self.retries):
            try:
                self._ensure()
                HfApi(token=self.token).upload_file(
                    path_or_fileobj=local, repo_id=self.repo_id, repo_type="dataset",
                    path_in_repo=path_in_repo)
                print(f"  ↑ {self.repo_id}:{path_in_repo}")
                return
            except Exception as e:
                wait = 2 ** attempt
                print(f"  ↑ {path_in_repo} {attempt + 1}/{self.retries} failed "
                      f"({type(e).__name__}: {e}); retry {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"upload FAILED after {self.retries} tries: {path_in_repo}")

    def close(self):
        self.flush()
