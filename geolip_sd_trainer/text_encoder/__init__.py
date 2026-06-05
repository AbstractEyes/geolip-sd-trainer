"""Native CLIP text encoders (CLIP-L + CLIP-G), one source of truth."""
from .clip_text import (NativeCLIPTextEncoder, CLIPTextConfig, CLIPOutput,
                        CLIP_L_CONFIG, CLIP_G_CONFIG)
from .clip_text_test import load_clip_text, load_clip_l
from .clip_g_test import load_clip_g

__all__ = ["NativeCLIPTextEncoder", "CLIPTextConfig", "CLIPOutput",
           "CLIP_L_CONFIG", "CLIP_G_CONFIG", "load_clip_text", "load_clip_l", "load_clip_g"]