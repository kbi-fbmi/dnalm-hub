"""Known NTv3 checkpoints published by InstaDeep on the HuggingFace Hub.

The registry lets tools accept short, memorable aliases (``"100m-pre"``) instead of
full repo ids. Any HuggingFace repo id can also be passed straight through, so a
user's own fine-tuned NTv3-derived checkpoint works too, as long as it ships
``trust_remote_code``-compatible modeling code following the same architecture.

Source: https://huggingface.co/collections/InstaDeepAI/nucleotide-transformer-v3
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    params: str
    stage: str  # "pre" (DNA-only masked LM) | "post" (+ species-conditioned annotation/track heads)
    context: str
    pad_multiple: int
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "8m-pre": ModelSpec("InstaDeepAI/NTv3_8M_pre", "7.7M", "pre", "full (~1Mb)", 128),
    "8m-pre-8kb": ModelSpec("InstaDeepAI/NTv3_8M_pre_8kb", "7.7M", "pre", "8kb", 128),
    "100m-pre": ModelSpec(
        "InstaDeepAI/NTv3_100M_pre",
        "0.1B",
        "pre",
        "full (~1Mb)",
        128,
        "Recommended default: good quality/cost tradeoff for embeddings and variant scoring.",
    ),
    "100m-pre-8kb": ModelSpec("InstaDeepAI/NTv3_100M_pre_8kb", "0.1B", "pre", "8kb", 128),
    "100m-post": ModelSpec(
        "InstaDeepAI/NTv3_100M_post",
        "0.1B",
        "post",
        "full (~1Mb)",
        128,
        "Post-trained on functional genomics: species-conditioned (`species`, default human); adds "
        "annotate_sequence (genes, exons, promoters, ...) and predict_tracks (RNA-seq/ChIP/ATAC signal).",
    ),
    "100m-post-131kb": ModelSpec("InstaDeepAI/NTv3_100M_post_131kb", "0.1B", "post", "131kb", 128),
    "650m-pre": ModelSpec(
        "InstaDeepAI/NTv3_650M_pre",
        "0.7B",
        "pre",
        "full (~1Mb)",
        128,
        "Highest-quality pre-trained checkpoint; heavier compute and memory.",
    ),
    "650m-pre-8kb": ModelSpec("InstaDeepAI/NTv3_650M_pre_8kb", "0.7B", "pre", "8kb", 128),
    "650m-post": ModelSpec("InstaDeepAI/NTv3_650M_post", "0.7B", "post", "full (~1Mb)", 128),
    "650m-post-131kb": ModelSpec("InstaDeepAI/NTv3_650M_post_131kb", "0.7B", "post", "131kb", 128),
    "5ds-pre": ModelSpec(
        "InstaDeepAI/NTv3_5downsample_pre",
        "0.6B",
        "pre",
        "full (~1Mb)",
        32,
        "Experimental 5-downsample variant; requires length divisible by 32, not 128.",
    ),
    "5ds-pre-8kb": ModelSpec("InstaDeepAI/NTv3_5downsample_pre_8kb", "0.6B", "pre", "8kb", 32),
    "5ds-post": ModelSpec("InstaDeepAI/NTv3_5downsample_post", "0.6B", "post", "full (~1Mb)", 32),
    "5ds-post-131kb": ModelSpec(
        "InstaDeepAI/NTv3_5downsample_post_131kb", "0.6B", "post", "131kb", 32
    ),
}

DEFAULT_MODEL_ALIAS = "100m-pre"
# Default for the post-only tools (annotate_sequence, predict_tracks, list_species, list_tracks).
DEFAULT_POST_MODEL_ALIAS = "100m-post"


def resolve_model(name: str | None) -> tuple[str, int]:
    """Resolve a registry alias or a raw HuggingFace repo id to ``(repo_id, pad_multiple)``."""
    if not name:
        name = DEFAULT_MODEL_ALIAS
    key = name.strip().lower()
    if key in MODEL_REGISTRY:
        spec = MODEL_REGISTRY[key]
        return spec.repo_id, spec.pad_multiple
    if "/" in name:
        # Custom or unregistered repo id: best-effort guess at the padding requirement.
        pad_multiple = 32 if "5downsample" in name.lower() else 128
        return name, pad_multiple
    raise ValueError(
        f"Unknown model '{name}'. Call list_models for available aliases, or pass a full "
        "HuggingFace repo id (e.g. 'InstaDeepAI/NTv3_100M_pre')."
    )
