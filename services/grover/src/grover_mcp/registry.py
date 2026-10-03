"""Known GROVER checkpoints published by PoetschLab (TU Dresden).

GROVER ("Genome Rules Obtained Via Extracted Representations", Sanabria et al.,
Nature Machine Intelligence 2024, https://doi.org/10.1038/s42256-024-00872-0) is
a BERT-base masked DNA language model trained on the human genome (hg19) with a
byte-pair-encoding vocabulary from 600 merge cycles. PoetschLab publishes a
single pre-trained checkpoint; the same weights are also on Zenodo
(https://doi.org/10.5281/zenodo.8373117).

Tools accept the short alias (``"grover"``) instead of the full repo id. Any other
HuggingFace repo id can also be passed straight through (e.g. a GROVER fine-tune),
as long as it loads via ``AutoModelForMaskedLM`` with a GROVER-style BPE tokenizer.

The registered checkpoint is pinned to a Hub commit (``revision``) even though it
ships no remote code: the Hub repo's tokenizer files have been edited after
release, and pinning keeps tokenization (and so every score) reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str | None
    params: str
    max_tokens: int  # context limit, including [CLS] and [SEP]
    notes: str = ""


# BERT absolute position embeddings: max_position_embeddings = 512.
_GROVER_MAX_TOKENS = 512

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "grover": ModelSpec(
        "PoetschLab/GROVER",
        "6b223110f0d6963e849f55bc2a2f3cff0e38c7a4",
        "87M",
        _GROVER_MAX_TOKENS,
        "GROVER: BERT-base masked LM on the human genome (hg19), BPE vocabulary (600 merges), "
        "512-token context (~1.9-2 kb, depending on the sequence).",
    ),
}

DEFAULT_MODEL_ALIAS = "grover"


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
        "full HuggingFace repo id (e.g. 'PoetschLab/GROVER')."
    )
