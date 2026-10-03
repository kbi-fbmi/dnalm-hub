"""Known HyenaDNA checkpoints published by HazyResearch / LongSafari.

Tools accept short aliases (``"medium-160k"``) instead of full repo ids. Any
HuggingFace repo id can also be passed straight through (e.g. your own HyenaDNA
fine-tune), as long as it loads via ``AutoModelForCausalLM`` with the HyenaDNA
character tokenizer.

Registered checkpoints are pinned to a specific Hub commit (``revision``): the
``-hf`` repos ship ``trust_remote_code`` modeling code, and pinning means a
Hub-side edit to that code can't silently change (or break) what this server runs.

Source: https://huggingface.co/LongSafari (``hyenadna-*-seqlen-hf``; the repos
without ``-hf`` are the original, non-transformers checkpoints).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str | None
    params: str
    num_layers: int
    hidden_size: int
    # Context limit in tokens = bases: config.max_seq_len (pretraining length + 2).
    # This service adds no special tokens, so every token is one base.
    max_tokens: int
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "tiny-1k": ModelSpec(
        "LongSafari/hyenadna-tiny-1k-seqlen-hf",
        "e8c1effa8673814e257e627d2e1eda9ea5a373f6",
        "0.44M",
        2,
        128,
        1026,
        "Smallest HyenaDNA (2 layers, width 128), 1k context. Fast; good for testing.",
    ),
    "small-32k": ModelSpec(
        "LongSafari/hyenadna-small-32k-seqlen-hf",
        "8fe770c78eb13fe33bf81501612faeddf4d6f331",
        "3.3M",
        4,
        256,
        32770,
    ),
    "medium-160k": ModelSpec(
        "LongSafari/hyenadna-medium-160k-seqlen-hf",
        "7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce",
        "6.6M",
        8,
        256,
        160002,
        "Recommended default: 8 layers, 160k-base context.",
    ),
    "medium-450k": ModelSpec(
        "LongSafari/hyenadna-medium-450k-seqlen-hf",
        "42dedd4d374eac0fb8168549e546a3472fbd27ae",
        "6.6M",
        8,
        256,
        450002,
    ),
    "large-1m": ModelSpec(
        "LongSafari/hyenadna-large-1m-seqlen-hf",
        "0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446",
        "6.6M",
        8,
        256,
        1000002,
        "Longest context (1M bases). Same width/depth as the medium checkpoints; "
        "a 1M-base input needs ~11 GiB of VRAM (see HYENADNA_MAX_BASES).",
    ),
}

DEFAULT_MODEL_ALIAS = "medium-160k"


@dataclass(frozen=True)
class ResolvedModel:
    repo_id: str
    revision: str | None
    max_tokens: int | None  # None: unregistered repo, read config.max_seq_len


def resolve_model(name: str | None) -> ResolvedModel:
    """Resolve a registry alias or a raw HuggingFace repo id."""
    if not name:
        name = DEFAULT_MODEL_ALIAS
    key = name.strip().lower()
    if key in MODEL_REGISTRY:
        spec = MODEL_REGISTRY[key]
        return ResolvedModel(spec.repo_id, spec.revision, spec.max_tokens)
    if "/" in name:
        return ResolvedModel(name.strip(), None, None)
    raise ValueError(
        f"Unknown model '{name}'. Call list_available_checkpoints for available aliases, or pass a "
        "full HuggingFace repo id (e.g. 'LongSafari/hyenadna-medium-160k-seqlen-hf')."
    )
