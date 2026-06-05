"""
geolip-sd-trainer
=================
Pure-PyTorch SDXL trainer with Qwen-pooled + aleph conditioning on the Lune
rectified-flow objective. The SDXL UNet / VAE / CLIP-L / CLIP-G are reimplemented
natively (no diffusers at runtime); Qwen3.5 is a frozen HF encoder.

Activation points (the public surface):

  Assemble a model from a testing config
  --------------------------------------
  >>> from geolip_sd_trainer import build_sdxl, PHASE1_RECIPE
  >>> model = build_sdxl(PHASE1_RECIPE)                 # locked rank-#3 recipe

  Precompute features, then train phase 1
  ---------------------------------------
  >>> from geolip_sd_trainer import Phase1Config, train
  >>> train(Phase1Config(shift=2.0, unet_mode="full"))  # build_cache runs first

  Lower-level handles
  -------------------
  GeolipSDXL, SDXLModelConfig, ComponentConfig, ConditioningConfig,
  conditioning_from_preset, CONDITIONING_PRESETS, SDXLQwenFrontEnd,
  Phase1Trainer, build_cache, fm_targets, euler_sample, DropoutSchedule,
  save_checkpoint / load_checkpoint / HubUploader, QwenConfig.

The heavy native modules (transformer/, vae/, text_encoder/) load lazily through
the component loaders, so importing this package stays light (torch + numpy only).
"""
__version__ = "0.1.0"

from .model import (
    GeolipSDXL, SDXLModelConfig, ComponentConfig, ConditioningConfig,
    SDXLQwenFrontEnd, conditioning_from_preset, CONDITIONING_PRESETS,
    PHASE1_RECIPE, build_sdxl, TRAIN_COMPONENTS, ENCODER_COMPONENTS, ALL_COMPONENTS,
)
from .trainer import (
    Phase1Config, Phase1Trainer, train, build_cache, CachedDS,
    fm_targets, euler_sample, DropoutSchedule, RUN_SHIFTS,
)
from .checkpoint import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint, rotate_checkpoints,
    export_unet_safetensors, resolve_hf_token, hf_whoami, HubUploader,
)
from .vlm.qwen import QwenConfig

__all__ = [
    "__version__",
    # assembly
    "GeolipSDXL", "SDXLModelConfig", "ComponentConfig", "ConditioningConfig",
    "SDXLQwenFrontEnd", "conditioning_from_preset", "CONDITIONING_PRESETS",
    "PHASE1_RECIPE", "build_sdxl", "TRAIN_COMPONENTS", "ENCODER_COMPONENTS", "ALL_COMPONENTS",
    # training
    "Phase1Config", "Phase1Trainer", "train", "build_cache", "CachedDS",
    "fm_targets", "euler_sample", "DropoutSchedule", "RUN_SHIFTS",
    # checkpoint / hub
    "save_checkpoint", "load_checkpoint", "find_latest_checkpoint", "rotate_checkpoints",
    "export_unet_safetensors", "resolve_hf_token", "hf_whoami", "HubUploader",
    # qwen
    "QwenConfig",
]