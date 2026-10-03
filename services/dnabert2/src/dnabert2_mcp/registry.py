"""Known DNABERT-2 checkpoints on the HuggingFace Hub.

Tools accept short aliases (``"117m"``) instead of full repo ids. Any HuggingFace
repo id can also be passed straight through (e.g. your own DNABERT-2 fine-tune), as
long as it ships the same MosaicBERT-style ``trust_remote_code`` architecture with
a ``BertForMaskedLM`` head and DNABERT-2's BPE tokenizer.

Registered checkpoints are pinned to a specific Hub commit (``revision``): the repo
ships ``trust_remote_code`` modeling code (``bert_layers.py``), and pinning means a
Hub-side edit to that code can't silently change (or break) what this server runs.

Only DNABERT-2-117M is registered on purpose. DNABERT-S (a contrastively fine-tuned
sibling for species-aware embeddings) is deliberately not offered here.

Source: https://huggingface.co/zhihan1996/DNABERT-2-117M
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str | None
    params: str
    max_tokens: int  # context limit, including the [CLS] and [SEP] special tokens
    notes: str = ""


# DNABERT-2 uses ALiBi instead of position embeddings, so there is no hard limit
# baked into the weights. It was pre-trained on short sequences (the paper uses
# 128-token inputs) and the config sizes the ALiBi bias table to 512
# (`alibi_starting_size`, also `max_position_embeddings`). We cap at 512 tokens
# (~2.4 kb of random sequence at ~4.7 bp/token, more for repetitive DNA): 4x the
# training length is a sensible extrapolation bound, and staying within the
# pre-built table avoids the remote code rebuilding (mutating) it per request.
_MAX_TOKENS = 512

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "117m": ModelSpec(
        "zhihan1996/DNABERT-2-117M",
        "7bce263b15377fc15361f52cfab88f8b586abda0",
        "117M",
        _MAX_TOKENS,
        "DNABERT-2 117M: BPE tokenizer, ALiBi, multi-species masked LM (the only DNABERT-2 checkpoint).",
    ),
}

DEFAULT_MODEL_ALIAS = "117m"


@dataclass(frozen=True)
class ResolvedModel:
    repo_id: str
    revision: str | None
    max_tokens: int | None  # None: unregistered repo, read the tokenizer's model_max_length


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
        "full HuggingFace repo id (e.g. 'zhihan1996/DNABERT-2-117M')."
    )
