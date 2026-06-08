"""
vlm/qwen.py — Qwen3.5 rich-pooled text interface (the third encoder).
=====================================================================
A thin, owned interface over the HF Qwen3.5 causal LM used as a FROZEN
rich-pooled text encoder — NOT a native reimplementation (Qwen is a frozen,
deterministic, cacheable feature extractor; reimplementing a 0.8B LLM buys
nothing here). The proven extraction (beatrix QwenDualShot / Qwen3-Embedding):

  chat-template (instruction, not a bare string)
    -> optional two-shot generate-then-encode (Qwen re-describes the caption)
    -> last-token pooling of a chosen layer (the [EOS] aggregate), padding-aware

Output: one rich pooled vector per caption ([B, hidden], fp32, CPU) — the signal
that drives both SDXL conditioning slots via the front-end.

REQUIREMENTS: Qwen3.5 needs a recent `transformers` (newer than Colab ships).
Set QwenConfig.min_transformers to the required version and pin it in
requirements.txt; `qwen_preflight` fails fast with the upgrade command before any
costly download. Decoder-only batched generation REQUIRES left padding.

Author: AbstractPhil + Mirel | License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

# (plain caption, vivid re-description) — the two-shot exemplars that steer the
# generate pass toward concrete, single-sentence descriptions.
DEFAULT_EXAMPLES: Tuple[Tuple[str, str], ...] = (
    ("a red bicycle leaning on a brick wall",
     "A bright red bicycle with thin tires rests against a weathered red-brick wall in soft daylight."),
    ("portrait of an old fisherman",
     "A weathered elderly fisherman with a grey beard and knit cap, lit by warm side light, looking to camera."),
)

DEFAULT_SYS_PROMPT = ("You describe images in one vivid, concrete sentence — "
                      "name the subjects, their attributes, the composition, "
                      "lighting and style.")


@dataclass
class QwenConfig:
    repo: str = "Qwen/Qwen3.5-0.8B"
    generate: bool = True               # two-shot generate-then-encode (richest); False = direct encode
    max_new_tokens: int = 64
    layer: int = -1                     # hidden layer to pool (Qwen-Embedding convention: last)
    max_length: int = 1024
    sys_prompt: str = DEFAULT_SYS_PROMPT
    examples: Tuple[Tuple[str, str], ...] = DEFAULT_EXAMPLES
    # Qwen3.5 requires transformers v5 (4.x will NOT work); 0.8B floored at >=5.2.0.
    # None = skip the gate. Larger MoE variants (e.g. 27B) may want >=5.4.
    min_transformers: Optional[str] = "5.2.0"
    trust_remote_code: bool = True


# ----------------------------------------------------------------------------
# preflight (fail fast before the download)
# ----------------------------------------------------------------------------

def _ver_tuple(v: str) -> tuple:
    out = []
    for part in v.split(".")[:3]:
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def qwen_preflight(cfg: QwenConfig) -> str:
    """Verify transformers is present and (if min_transformers set) new enough for
    Qwen3.5. Returns the installed version string. Raises SystemExit with the fix."""
    try:
        import transformers
    except ImportError:
        raise SystemExit("transformers is not installed.\n  pip install -U "
                         f"'transformers{'>=' + cfg.min_transformers if cfg.min_transformers else ''}'")
    ver = getattr(transformers, "__version__", "0")
    if cfg.min_transformers and _ver_tuple(ver) < _ver_tuple(cfg.min_transformers):
        raise SystemExit(
            f"\ntransformers {ver} is too old for Qwen3.5 (need >= {cfg.min_transformers}).\n"
            f"  pip install -U 'transformers>={cfg.min_transformers}'\n"
            "  (Colab ships an older transformers — upgrade, then RESTART the runtime.)")
    return ver


# ----------------------------------------------------------------------------
# pooling utility (padding-aware last-token / [EOS] aggregate)
# ----------------------------------------------------------------------------

def last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """Hidden state of the final REAL token, handling LEFT or RIGHT padding.
    Left-padded (the generation/batch default) -> the last column is real for all
    rows; right-padded -> index by per-row length. (Qwen3-Embedding convention.)"""
    left_padded = bool((attn_mask[:, -1].sum() == attn_mask.shape[0]).item())
    if left_padded:
        return last_hidden[:, -1]
    lengths = attn_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), lengths]


# ----------------------------------------------------------------------------
# the encoder
# ----------------------------------------------------------------------------

class QwenPooledEncoder:
    """Caption -> rich POOLED Qwen vector [B, hidden]. Frozen, deterministic
    (greedy), cacheable. Construct via build_qwen(...) or directly with a loaded
    HF model + tokenizer."""

    def __init__(self, model, tokenizer, cfg: QwenConfig):
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg
        self.hidden = int(model.config.hidden_size)
        # decoder-only batched generation REQUIRES left padding (right padding makes
        # shorter captions generate after their pad tokens, corrupting output).
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    # -- chat templating --
    def _chat(self, text: str, as_generation: bool) -> str:
        msgs = [{"role": "system", "content": self.cfg.sys_prompt}]
        if as_generation:
            for a, b in self.cfg.examples:
                msgs.append({"role": "user", "content": f"Describe: {a}"})
                msgs.append({"role": "assistant", "content": b})
            msgs.append({"role": "user", "content": f"Describe: {text}"})
        else:
            msgs.append({"role": "user", "content": text})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=as_generation)

    @torch.no_grad()
    def _generate_batch(self, texts: List[str]) -> List[str]:
        prompts = [self._chat(t, as_generation=True) for t in texts]
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.cfg.max_length).to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=self.cfg.max_new_tokens,
                                  do_sample=False,
                                  pad_token_id=(self.tok.pad_token_id or self.tok.eos_token_id))
        gen = out[:, enc["input_ids"].shape[1]:]
        return [self.tok.decode(g, skip_special_tokens=True).strip() for g in gen]

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """-> [B, hidden] pooled fp32 on CPU."""
        to_encode = self._generate_batch(texts) if self.cfg.generate else list(texts)
        wrapped = [self._chat(t, as_generation=False) for t in to_encode]
        enc = self.tok(wrapped, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.cfg.max_length).to(self.model.device)
        out = self.model(**enc, output_hidden_states=True, return_dict=True)
        hid = out.hidden_states[self.cfg.layer]                     # [B, T, hidden]
        pooled = last_token_pool(hid, enc["attention_mask"])
        return pooled.float().cpu()

    @torch.no_grad()
    def encode_batched(self, texts: List[str], batch_size: int = 16) -> torch.Tensor:
        """Encode a long list in chunks -> [N, hidden] fp32 CPU. Utility for precompute."""
        chunks = [self.encode(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        return torch.cat(chunks, dim=0) if chunks else torch.empty(0, self.hidden)

    __call__ = encode


# ----------------------------------------------------------------------------
# loader
# ----------------------------------------------------------------------------

def build_qwen(cfg: Optional[QwenConfig] = None, device: str = "cuda",
               dtype: torch.dtype = torch.bfloat16,
               token: Optional[str] = None) -> QwenPooledEncoder:
    """Preflight, load the frozen HF Qwen3.5 model + tokenizer, wrap as a pooled
    encoder. Surfaces a clear upgrade hint if transformers can't build Qwen3.5."""
    cfg = cfg or QwenConfig()
    qwen_preflight(cfg)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    common = dict(trust_remote_code=cfg.trust_remote_code, torch_dtype=dtype, token=token)
    try:
        tok = AutoTokenizer.from_pretrained(cfg.repo, trust_remote_code=cfg.trust_remote_code, token=token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        # low_cpu_mem_usage loads via meta+assign (no random-init); device_map places the
        # weights straight on the GPU, skipping the CPU→GPU copy and the ~2x peak host-RAM
        # spike that OOMs free Colab. device_map needs `accelerate`; if it's missing we fall
        # back to a plain load + .to(device) (still correct, just the slower path).
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.repo, low_cpu_mem_usage=True,
                device_map=({"": device} if device != "cpu" else None), **common)
            placed = device != "cpu"
        except (ImportError, NotImplementedError):
            model, placed = AutoModelForCausalLM.from_pretrained(cfg.repo, **common), False
    except (KeyError, ValueError) as e:
        raise SystemExit(
            f"\nCould not build Qwen3.5 from '{cfg.repo}' with the installed transformers "
            f"({e}).\nThis usually means transformers is too old for the Qwen3.5 architecture.\n"
            f"  pip install -U 'transformers{'>=' + cfg.min_transformers if cfg.min_transformers else ''}'  "
            "then restart the runtime.")
    if not placed:
        model = model.to(device)
    model = model.eval()
    model.requires_grad_(False)
    return QwenPooledEncoder(model, tok, cfg)