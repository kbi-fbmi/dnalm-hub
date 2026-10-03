"""Known __NAME__ checkpoints on the HuggingFace Hub.

Tools accept short aliases instead of full repo ids; any other repo id is passed
straight through. Pin `revision` to a Hub commit for every registered checkpoint
that ships `trust_remote_code` modeling code, so a Hub-side edit can't silently
change (or break) what this server runs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str | None
    params: str
    max_tokens: int  # context limit, including special tokens
    notes: str = ""


# TODO: replace with the real checkpoints (and their commit SHAs from
# https://huggingface.co/api/models/<repo_id> -> "sha").
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "small": ModelSpec("org/model-small", None, "?M", 1024, "TODO: describe."),
}

DEFAULT_MODEL_ALIAS = "small"


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
        f"Unknown model '{name}'. Call list_available_checkpoints for available aliases, "
        "or pass a full HuggingFace repo id."
    )
