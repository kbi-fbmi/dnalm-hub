"""Known GENA-LM checkpoints published by AIRI Institute.

Tools accept short aliases (``"bert-large-t2t"``) instead of full repo ids. Any
HuggingFace repo id can also be passed straight through (e.g. your own GENA-LM
fine-tune), as long as it loads via ``AutoModel`` (``trust_remote_code``) into a
masked-LM head, or is a native ``BigBirdForMaskedLM`` checkpoint.

Registered checkpoints are pinned to a specific Hub commit (``revision``): the
BERT repos ship ``trust_remote_code`` modeling code, and pinning means a Hub-side
edit to that code can't silently change (or break) what this server runs.

Not registered:

- ``gena-lm-bigbird-base-sparse(-t2t)``: needs DeepSpeed sparse-attention ops
  (Triton kernels, fp16 only) -- a heavy, fragile dependency for a 110M model
  whose 4096-token context ``bigbird-base-t2t`` already covers.
- ``gena-lm-bert-base`` / ``bigbird-base-sparse``: the authors' preliminary
  models (non-augmented "T2T split v1"), superseded by the ``-t2t`` ones.

Source: https://github.com/AIRI-Institute/GENA_LM/blob/main/README_previous_generation.md
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str | None
    params: str
    architecture: str  # "bert" (remote pre-layernorm BERT) | "bigbird" (transformers' BigBird)
    max_tokens: int  # context limit, including [CLS] and [SEP]
    notes: str = ""


# BERT variants: absolute position embeddings, 512 tokens (~4.5 kb of human DNA
# with GENA's BPE vocabulary; only ~3 kb of high-entropy, e.g. random, sequence).
_BERT_MAX_TOKENS = 512
# HF BigBird (block-sparse attention): 4096 tokens (~36 kb).
_BIGBIRD_MAX_TOKENS = 4096

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "bert-base-t2t": ModelSpec(
        "AIRI-Institute/gena-lm-bert-base-t2t",
        "4f1352bd4e820f1dba341047f54ad7795e083bf2",
        "110M",
        "bert",
        _BERT_MAX_TOKENS,
        "BERT-12L, pre-layernorm without a final layernorm; trained on T2T human + 1000G SNPs.",
    ),
    "bert-base-lastln-t2t": ModelSpec(
        "AIRI-Institute/gena-lm-bert-base-lastln-t2t",
        "3177322cb31140be33a692bfb968263cbd3ef89a",
        "110M",
        "bert",
        _BERT_MAX_TOKENS,
        "BERT-12L, pre-layernorm WITH a final layernorm; trained on T2T human + 1000G SNPs.",
    ),
    "bert-base-t2t-multi": ModelSpec(
        "AIRI-Institute/gena-lm-bert-base-t2t-multi",
        "4633e5a1ada905bb7afee6877d71cc12578a95a5",
        "110M",
        "bert",
        _BERT_MAX_TOKENS,
        "BERT-12L; trained on T2T human + 1000G SNPs + multispecies genomes.",
    ),
    "bert-large-t2t": ModelSpec(
        "AIRI-Institute/gena-lm-bert-large-t2t",
        "f997a8a7c1ba5feb6d11de46354a41c51ffe9660",
        "336M",
        "bert",
        _BERT_MAX_TOKENS,
        "Recommended default: BERT-24L, the largest GENA-LM and the best one in the authors' "
        "downstream benchmarks (promoters, splice sites); trained on T2T human + 1000G SNPs.",
    ),
    "bigbird-base-t2t": ModelSpec(
        "AIRI-Institute/gena-lm-bigbird-base-t2t",
        "f155a24ef5cefd7218a1e4c6888aedb574509cc3",
        "110M",
        "bigbird",
        _BIGBIRD_MAX_TOKENS,
        "BigBird-12L (block-sparse attention), 4096-token (~36 kb) context; trained on T2T "
        "human + 1000G SNPs. Sequences are run one at a time (see README).",
    ),
}

DEFAULT_MODEL_ALIAS = "bert-large-t2t"


@dataclass(frozen=True)
class ResolvedModel:
    repo_id: str
    revision: str | None
    max_tokens: int | None  # None: unregistered repo, read it from the model config
    architecture: str | None = None  # None: unregistered repo, read it from the model config


def resolve_model(name: str | None) -> ResolvedModel:
    """Resolve a registry alias or a raw HuggingFace repo id."""
    if not name:
        name = DEFAULT_MODEL_ALIAS
    key = name.strip().lower()
    if key in MODEL_REGISTRY:
        spec = MODEL_REGISTRY[key]
        return ResolvedModel(spec.repo_id, spec.revision, spec.max_tokens, spec.architecture)
    if "/" in name:
        return ResolvedModel(name.strip(), None, None)
    raise ValueError(
        f"Unknown model '{name}'. Call list_available_checkpoints for available aliases, or pass a "
        "full HuggingFace repo id (e.g. 'AIRI-Institute/gena-lm-bert-large-t2t')."
    )
