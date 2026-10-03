"""Model loading and inference for __NAME__.

Fill in the TODOs. Keep the function signatures: `server.py` and the tests rely
on them. Reuse `dnalm_common` instead of re-implementing shared pieces:

- `LRUModelCache`, `select_device` / `select_dtype` -- loading and VRAM budget,
- `validate_sequence` -- input validation (raise `ValueError` for bad input; the
  server turns it into a readable MCP error),
- `score_snp_whole_sequence` + `masked_lm_pseudo_log_likelihood` (masked LMs) or
  an exact causal log-likelihood (causal LMs) -- tokenizer-agnostic SNP scoring.

See services/ntv2/src/ntv2_mcp/inference.py for a complete masked-LM example.
"""

from __future__ import annotations

import os

import numpy as np
from dnalm_common import LRUModelCache, score_snp_whole_sequence, select_device, select_dtype

from .registry import ResolvedModel

DEVICE = select_device("__ENV___DEVICE")
DTYPE = select_dtype("__ENV___DTYPE")

_cache = LRUModelCache(max_entries=int(os.environ.get("__ENV___MAX_RESIDENT_MODELS", "2")))


def load(m: ResolvedModel):
    """Load (and LRU-cache) `(tokenizer, model)` for `m`."""

    def _load() -> tuple:
        # TODO: e.g.
        #   tokenizer = AutoTokenizer.from_pretrained(m.repo_id, revision=m.revision, trust_remote_code=True)
        #   model = AutoModelForMaskedLM.from_pretrained(m.repo_id, revision=m.revision,
        #                                                trust_remote_code=True, dtype=DTYPE)
        #   model.to(DEVICE).eval()
        raise NotImplementedError("load() is not implemented yet for __NAME__.")

    return _cache.get_or_load(f"{m.repo_id}@{m.revision or 'main'}", _load)


def model_info(m: ResolvedModel) -> dict:
    """Return at least: repo_id, device, dtype, num_parameters, hidden_size, num_hidden_state_layers."""
    raise NotImplementedError("model_info() is not implemented yet for __NAME__.")


def compute_embeddings(
    m: ResolvedModel, sequences: list[str], layer: str | int | None, pooling: str
) -> tuple[list[np.ndarray], int, int]:
    """Return `(embeddings, layer_index, num_layers)`: one 1D vector per sequence for
    pooling="mean", one (num_tokens, hidden) matrix per sequence for "per_token".
    Reject over-long sequences with a `ValueError`; never truncate silently."""
    raise NotImplementedError("compute_embeddings() is not implemented yet for __NAME__.")


def log_likelihood(m: ResolvedModel, sequence: str) -> float:
    """Whole-sequence (pseudo-)log-likelihood, used by score_snp."""
    raise NotImplementedError("log_likelihood() is not implemented yet for __NAME__.")


def score_variant(m: ResolvedModel, sequence: str, alt_allele: str, position: int | None = None) -> dict:
    return score_snp_whole_sequence(sequence, alt_allele, lambda s: log_likelihood(m, s), position=position)


def compare_sequences(m: ResolvedModel, sequence_a: str, sequence_b: str, layer: str | int | None):
    embeddings, layer_idx, _ = compute_embeddings(m, [sequence_a, sequence_b], layer, "mean")
    a, b = embeddings
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return (float(np.dot(a, b) / denom) if denom > 0 else 0.0), layer_idx
