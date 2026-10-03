"""Model loading and inference helpers for Nucleotide Transformer v2.

NTv2 checkpoints are ESM-style masked DNA language models shipped as
``trust_remote_code`` architectures on the HuggingFace Hub, loadable through
``AutoTokenizer`` / ``AutoModelForMaskedLM`` (see e.g.
https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species).

Key differences from NTv3 (``ntv3_mcp.inference``):

- 6-mer tokenizer: one token covers up to six bases (leftover bases and every
  'N' become single-character tokens), plus a leading ``<cls>``. Token indices
  therefore don't line up with sequence offsets, so there is no single-position
  masked prediction here, and SNP scoring uses the tokenizer-agnostic
  whole-sequence pseudo-log-likelihood delta from ``dnalm_common.scoring``.
- Hard context limit (2048 tokens for v2, 1000 for v1). Over-long sequences are
  rejected with a clear error, never silently truncated.
- The remote modeling code needs ``transformers<5`` (it imports helpers removed
  in 5.x) -- pinned in this service's own pyproject.toml.
- The remote code's rotary embedding emits float32 activations even when the
  weights are bf16/fp16, which then crash a mixed-dtype matmul. Half-precision
  forward passes therefore run under ``torch.autocast`` (see ``_autocast``).
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import torch
from dnalm_common import (
    LRUModelCache,
    is_single_nucleotide_tokenizer,
    masked_lm_pseudo_log_likelihood,
    score_snp_whole_sequence,
    select_device,
    select_dtype,
    validate_sequence,
)
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .registry import ResolvedModel

DEVICE = select_device("NTV2_DEVICE")
DTYPE = select_dtype("NTV2_DTYPE")

_MAX_RESIDENT_MODELS = int(os.environ.get("NTV2_MAX_RESIDENT_MODELS", "2"))
_PLL_BATCH_SIZE = int(os.environ.get("NTV2_PLL_BATCH_SIZE", "16"))
_cache = LRUModelCache(max_entries=_MAX_RESIDENT_MODELS)

# Fallback for unregistered repos whose tokenizer doesn't declare a usable model_max_length.
_FALLBACK_MAX_TOKENS = 2048
_BASES_PER_TOKEN = 6


def _autocast():
    """Autocast to DTYPE for half-precision runs; a no-op for float32 or non-CUDA devices."""
    if DTYPE in (torch.bfloat16, torch.float16) and DEVICE.startswith("cuda"):
        return torch.autocast("cuda", dtype=DTYPE)
    return contextlib.nullcontext()


def load(m: ResolvedModel):
    """Load (and LRU-cache, `NTV2_MAX_RESIDENT_MODELS`, default 2) the tokenizer + model."""

    def _load() -> tuple:
        try:
            tokenizer = AutoTokenizer.from_pretrained(m.repo_id, revision=m.revision, trust_remote_code=True)
            model = AutoModelForMaskedLM.from_pretrained(
                m.repo_id, revision=m.revision, trust_remote_code=True, dtype=DTYPE
            )
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
            raise RuntimeError(f"Failed to load NTv2 model '{m.repo_id}' from HuggingFace: {exc}") from exc
        tokenizer.padding_side = "right"
        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    return _cache.get_or_load(f"{m.repo_id}@{m.revision or 'main'}", _load)


def max_tokens(m: ResolvedModel, tokenizer) -> int:
    """Context limit in tokens (including <cls>): the registry's value, else the tokenizer's."""
    if m.max_tokens is not None:
        return m.max_tokens
    declared = getattr(tokenizer, "model_max_length", None)
    # Tokenizers without a configured limit report a huge sentinel (int(1e30)).
    if isinstance(declared, int) and 0 < declared < 1_000_000:
        return declared
    return _FALLBACK_MAX_TOKENS


def check_token_lengths(lengths: list[int], limit: int) -> None:
    """Raise a clear `ValueError` if any tokenized sequence exceeds `limit` tokens."""
    for i, n in enumerate(lengths):
        if n > limit:
            raise ValueError(
                f"Sequence {i} tokenizes to {n} tokens, above this checkpoint's {limit}-token context "
                f"limit (roughly {(limit - 1) * _BASES_PER_TOKEN} bp of N-free sequence; each 'N' costs a "
                "whole token). Sequences are never truncated -- split it into shorter windows."
            )


def _tokenize(m: ResolvedModel, tokenizer, sequences: list[str]) -> dict:
    encoded = tokenizer(sequences, add_special_tokens=True, truncation=False)
    check_token_lengths([len(ids) for ids in encoded["input_ids"]], max_tokens(m, tokenizer))
    batch = tokenizer.pad(encoded, padding=True, return_tensors="pt", return_attention_mask=True)
    return {"input_ids": batch["input_ids"].to(DEVICE), "attention_mask": batch["attention_mask"].to(DEVICE)}


def _parse_layer(layer: str | int | None, n_layers: int) -> int:
    if layer in (None, "last", "LAST"):
        return n_layers
    try:
        idx = int(layer)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"layer must be an integer (as int or string) or 'last'; got {layer!r}") from exc
    if idx < 0:
        idx = n_layers + 1 + idx
    if not (0 <= idx <= n_layers):
        raise ValueError(f"layer must be between 0 and {n_layers} (or 'last'); got {layer!r}")
    return idx


def _probe_hidden_states(m: ResolvedModel):
    """Run a tiny dummy forward pass to inspect how many hidden-state layers a model exposes."""
    tokenizer, model = load(m)
    batch = _tokenize(m, tokenizer, ["ACGTAC" * 4])
    with torch.no_grad(), _autocast():
        out = model(**batch, output_hidden_states=True)
    return tokenizer, model, out


def model_info(m: ResolvedModel) -> dict:
    tokenizer, model, out = _probe_hidden_states(m)
    n_params = sum(p.numel() for p in model.parameters())
    raw_cfg = model.config.to_dict()
    safe_cfg = {k: v for k, v in raw_cfg.items() if isinstance(v, (int, float, str, bool)) or v is None}
    return {
        "repo_id": m.repo_id,
        "revision": m.revision,
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "num_parameters": n_params,
        "vocab_size": len(tokenizer),
        "hidden_size": int(out.hidden_states[-1].shape[-1]),
        "num_hidden_state_layers": len(out.hidden_states),
        "max_tokens": max_tokens(m, tokenizer),
        "single_nucleotide_tokenizer": is_single_nucleotide_tokenizer(tokenizer),
        "mask_token": tokenizer.mask_token,
        "config": safe_cfg,
    }


def list_embedding_layers(m: ResolvedModel, which: str) -> dict:
    """Enumerate hidden-state layer indices usable as `layer_name` in embed_sequence.

    `which="recommended"` is a heuristic (final layer and the one before it), not an
    InstaDeep recommendation; `which="all"` returns every index, 0 (input
    embeddings) through the last transformer layer.
    """
    _, _, out = _probe_hidden_states(m)
    n_layers = len(out.hidden_states) - 1
    all_layers = list(range(0, n_layers + 1))
    if which == "all":
        layers = all_layers
        info = f"All {len(all_layers)} hidden-state layers (0=input embeddings, {n_layers}=final layer/'last')."
    elif which == "recommended":
        layers = sorted({n_layers, max(n_layers - 1, 0)})
        info = (
            "Heuristic only: the final layer and the one before it. InstaDeep's NTv2 model cards use "
            "the final layer; use which='all' to explore others."
        )
    else:
        raise ValueError("which must be 'recommended' or 'all'.")
    return {"layers": layers, "num_hidden_state_layers": len(out.hidden_states), "info": info}


def compute_embeddings(
    m: ResolvedModel,
    sequences: list[str],
    layer: str | int | None,
    pooling: str,
) -> tuple[list[np.ndarray], int, int]:
    """Per-sequence embeddings: a 1D vector each for "mean", a (num_tokens, hidden) matrix for "per_token".

    Mean pooling averages over every non-padding token, `<cls>` included -- the same
    recipe as InstaDeep's NTv2 model card. Per-token rows are 6-mer tokens (with the
    leading `<cls>` first), not individual bases, trimmed to each sequence's real length.
    """
    if not sequences:
        raise ValueError("sequences must be a non-empty list of DNA sequences.")
    if pooling not in ("mean", "per_token"):
        raise ValueError("pooling must be 'mean' or 'per_token'.")
    tokenizer, model = load(m)
    seqs = [validate_sequence(s) for s in sequences]
    batch = _tokenize(m, tokenizer, seqs)
    with torch.no_grad(), _autocast():
        out = model(**batch, output_hidden_states=True)
    n_layers = len(out.hidden_states) - 1
    idx = _parse_layer(layer, n_layers)
    h = out.hidden_states[idx].float()
    mask = batch["attention_mask"]

    if pooling == "mean":
        weights = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return list(pooled.cpu().numpy()), idx, n_layers
    lengths = mask.sum(dim=1).tolist()
    return [h[i, : int(n)].cpu().numpy() for i, n in enumerate(lengths)], idx, n_layers


def compare_sequences(
    m: ResolvedModel,
    sequence_a: str,
    sequence_b: str,
    layer: str | int | None,
) -> tuple[float, int]:
    embeddings, layer_idx, _ = compute_embeddings(m, [sequence_a, sequence_b], layer, "mean")
    a, b = embeddings
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    similarity = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    return similarity, layer_idx


def pseudo_log_likelihood(m: ResolvedModel, sequence: str) -> float:
    """Masked-LM pseudo-log-likelihood of `sequence`, summed over all non-special tokens."""
    tokenizer, model = load(m)
    seq = validate_sequence(sequence)
    batch = _tokenize(m, tokenizer, [seq])
    input_ids = batch["input_ids"][0]
    special = set(tokenizer.all_special_ids)
    positions = [i for i, t in enumerate(input_ids.tolist()) if t not in special]

    def logits_fn(ids: torch.Tensor) -> torch.Tensor:
        with _autocast():
            return model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits

    return masked_lm_pseudo_log_likelihood(
        input_ids, positions, tokenizer.mask_token_id, logits_fn, batch_size=_PLL_BATCH_SIZE
    )


def score_variant(m: ResolvedModel, sequence: str, alt_allele: str, position: int | None = None) -> dict:
    """Whole-sequence PLL(mutated) - PLL(reference); see `dnalm_common.scoring`."""
    return score_snp_whole_sequence(
        sequence, alt_allele, lambda s: pseudo_log_likelihood(m, s), position=position
    )
