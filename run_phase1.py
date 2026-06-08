"""
run_phase1.py — launcher for single- or multi-pod phase-1 training.
===================================================================
Single pod:
    python run_phase1.py

Multi pod (torchrun sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT):
    torchrun --nproc_per_node=<gpus> --nnodes=<N> --node_rank=<r> \
             --master_addr=<host> --master_port=<port> run_phase1.py

Override paths / knobs with env vars (handy for RunPod entry scripts). For multi-pod,
GEOLIP_CACHE_DIR and GEOLIP_OUT_DIR MUST point at storage every pod shares (a network
volume, etc.) so all ranks see the merged feature cache and resume consistently.

See docs/MULTIPOD.md for the full guide.
"""
import os

from geolip_sd_trainer import Phase1Config, train


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


if __name__ == "__main__":
    cfg = Phase1Config(
        dataset_repo=_env("GEOLIP_DATASET", "AbstractPhil/sdxl-qwen-phase0"),
        cache_dir=_env("GEOLIP_CACHE_DIR", "./phase1_cache"),   # SHARE across pods for multi-pod
        out_dir=_env("GEOLIP_OUT_DIR", "./phase1_runs"),        # SHARE across pods for multi-pod
        n_images=_env("GEOLIP_N_IMAGES", 0, int),               # 0 = full dataset; >0 caps (smoke tests)
        num_epochs=_env("GEOLIP_EPOCHS", 60, int),
        batch_size=_env("GEOLIP_BATCH", 4, int),
        shift=_env("GEOLIP_SHIFT", 2.0, float),
        unet_mode=_env("GEOLIP_UNET_MODE", "full"),
        hf_repo_id=_env("GEOLIP_HF_REPO", "AbstractPhil/geolip-sdxl-aleph"),
        upload_to_hub=_env("GEOLIP_UPLOAD", "1") == "1",
    )
    train(cfg)
