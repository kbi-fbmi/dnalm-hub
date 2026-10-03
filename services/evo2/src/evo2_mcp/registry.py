"""Known Evo 2 checkpoints (community transformers port) on the HuggingFace Hub.

Arc Institute publishes Evo 2 for its own Vortex inference stack (Transformer
Engine, FP8, custom kernels). This service instead runs the community
pure-PyTorch port by Aquiles-ai (https://github.com/Aquiles-ai/Evo2-transformers),
which loads through ``AutoModelForCausalLM`` + ``trust_remote_code`` with the
weights converted unchanged from the official checkpoints. The port publishes
two of Arc's eight checkpoints:

- ``Aquiles-ai/Evo2-7B`` = ``evo2_7b`` (1M context), registered as ``7b``.
- ``Aquiles-ai/Evo2-1B-Base`` = ``evo2_1b_base`` (8k context), NOT registered:
  Arc's docs say the 1B needs FP8 for accurate results, and in this port's BF16
  it scored a real human locus at -1.35 nats/base, barely better than shuffled
  sequence (-1.39; the 7B: -1.16). Passing the repo id still works.

The 40B checkpoints (~80 GB in BF16) can't fit on one 32 GB GPU and the port
publishes no 7B base/262k or 20B conversions, so none are registered. Any
other repo id produced by the port's converter (``convert_evo2_vortex_to_hf.py``)
can still be passed as ``checkpoint``.

Every registered checkpoint is pinned to a Hub commit, because it ships
``trust_remote_code`` modeling code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str | None
    params: str
    max_tokens: int  # native context of the checkpoint (1 token = 1 base, no special tokens)
    recommended_layers: tuple[str, ...]  # get_embedding_layers(which="recommended")
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "7b": ModelSpec(
        "Aquiles-ai/Evo2-7B",
        "0838c72ff3ada8ccebebff8fab3c8af508595ca4",
        "7B",
        1_048_576,
        # blocks.28.mlp.l3: the layer Arc's README uses for evo2_7b embeddings;
        # blocks.26.mlp.l3: the layer the Evo 2 paper's sparse autoencoders read.
        ("blocks.28.mlp.l3", "blocks.26.mlp.l3"),
        "Recommended default: Evo 2 7B (evo2_7b, 1M native context; this service caps the "
        "length, see max_sequence_length). Robust in BF16.",
    ),
}

DEFAULT_MODEL_ALIAS = "7b"


@dataclass(frozen=True)
class ResolvedModel:
    repo_id: str
    revision: str | None
    max_tokens: int | None  # None: unregistered repo, read max_position_embeddings from the config
    recommended_layers: tuple[str, ...] = ()  # empty: derive from the depth (see inference)


def resolve_model(name: str | None) -> ResolvedModel:
    """Resolve a registry alias or a raw HuggingFace repo id."""
    if not name:
        name = DEFAULT_MODEL_ALIAS
    key = name.strip().lower()
    if key in MODEL_REGISTRY:
        spec = MODEL_REGISTRY[key]
        return ResolvedModel(spec.repo_id, spec.revision, spec.max_tokens, spec.recommended_layers)
    if "/" in name:
        return ResolvedModel(name.strip(), None, None)
    raise ValueError(
        f"Unknown model '{name}'. Call list_available_checkpoints for available aliases, "
        "or pass a full HuggingFace repo id (e.g. 'Aquiles-ai/Evo2-7B')."
    )
