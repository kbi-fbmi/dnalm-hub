"""Known Nucleotide Transformer v2 (and v1 2.5B) checkpoints published by InstaDeep.

Tools accept short aliases (``"500m"``) instead of full repo ids. Any HuggingFace
repo id can also be passed straight through (e.g. your own NTv2 fine-tune), as long
as it loads via ``AutoModelForMaskedLM`` with a 6-mer ``EsmTokenizer``.

Registered checkpoints are pinned to a specific Hub commit (``revision``): the v2
repos ship ``trust_remote_code`` modeling code, and pinning means a Hub-side edit
to that code can't silently change (or break) what this server runs.

Source: https://huggingface.co/collections/InstaDeepAI/nucleotide-transformer-65099cdde13ff96230f2e592
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str | None
    params: str
    generation: str  # "v2" | "v1"
    max_tokens: int  # including the leading <cls> token
    notes: str = ""


# NTv2: rotary embeddings, 2048-token context (~12kb at 6 bases/token).
_V2_MAX_TOKENS = 2048
# NT v1: absolute position embeddings, 1000-token context (~6kb).
_V1_MAX_TOKENS = 1000

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "50m": ModelSpec(
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "81b29e5786726d891dbf929404ef20adca5b36f1",
        "50M", "v2", _V2_MAX_TOKENS,
    ),
    "100m": ModelSpec(
        "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "f34324c6fde36a4f635f0f1f06cac5d25acd6798",
        "100M", "v2", _V2_MAX_TOKENS,
    ),
    "250m": ModelSpec(
        "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        "c0f0359229f36ff6bc3a021247eefe0a9c344bd1",
        "250M", "v2", _V2_MAX_TOKENS,
    ),
    "500m": ModelSpec(
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "06615c1660c892fc199840c18123f8385b3542a8",
        "500M", "v2", _V2_MAX_TOKENS,
        "Recommended default: largest NTv2 checkpoint.",
    ),
    "2.5b-v1": ModelSpec(
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        "b746b125aacd1b0970c05a32fd71ba726754542e",
        "2.5B", "v1", _V1_MAX_TOKENS,
        "Nucleotide Transformer v1 (not v2): largest model in the collection, but only a "
        "1000-token (~6kb) context and ~5GB of weights in bf16.",
    ),
}

DEFAULT_MODEL_ALIAS = "500m"


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
        "full HuggingFace repo id (e.g. 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species')."
    )
