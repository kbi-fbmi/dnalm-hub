"""Model loading and inference helpers for GENA-LM.

GENA-LM checkpoints (https://github.com/AIRI-Institute/GENA_LM) are masked DNA
language models with a 32k-entry BPE tokenizer. Two architectures:

- ``bert-*``: a pre-layernorm BERT shipped as ``trust_remote_code``
  ``modeling_bert.py``. Its ``auto_map`` only registers ``AutoModel`` -- which
  maps to ``BertForMaskedLM`` -- so it is loaded through ``AutoModel``, not
  ``AutoModelForMaskedLM``. The remote code imports helpers removed in
  transformers 5.x, hence ``transformers<5`` in this service's pyproject.toml.
- ``bigbird-base-t2t``: transformers' own ``BigBirdForMaskedLM`` (no remote code).

Key behaviors:

- BPE tokens are variable length (1 to ~60 bases), plus ``[CLS]`` and ``[SEP]``.
  Token indices don't line up with sequence offsets, so there is no
  single-position masked prediction here, and SNP scoring uses the
  tokenizer-agnostic whole-sequence pseudo-log-likelihood delta from
  ``dnalm_common.scoring``. The tokenizer collapses any run of >= 10 'N' into a
  single ``-`` token.
- Hard context limit (512 tokens for BERT, 4096 for BigBird, specials
  included). Over-long sequences are rejected with a clear error, never
  silently truncated.
- BigBird switches between block-sparse and full attention depending on the
  (padded) input length, and pads inputs to a multiple of its block size. A
  short sequence batched with a long one would therefore get a different
  attention pattern than on its own. To keep batch == single, BigBird
  sequences are run one per forward pass, and hidden states are trimmed back
  to the real token count (BigBird returns them block-padded).
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
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from .registry import ResolvedModel

DEVICE = select_device("GENALM_DEVICE")
DTYPE = select_dtype("GENALM_DTYPE")

_MAX_RESIDENT_MODELS = int(os.environ.get("GENALM_MAX_RESIDENT_MODELS", "2"))
# score_snp: masked copies per forward pass = this token budget // sequence length
# (32 copies of a 512-token BERT input, 4 of a 4096-token BigBird input).
_PLL_BATCH_TOKENS = int(os.environ.get("GENALM_PLL_BATCH_TOKENS", "16384"))
_cache = LRUModelCache(max_entries=_MAX_RESIDENT_MODELS)

# Fallback for unregistered repos whose config has no max_position_embeddings.
_FALLBACK_MAX_TOKENS = 512
# Typical bases per BPE token on human DNA (authors: 512 tokens ~ 4.5 kb). Only used
# to phrase the over-length error; high-entropy sequence packs fewer bases per token.
_TYPICAL_BASES_PER_TOKEN = 9
# Special tokens added to every sequence: [CLS] ... [SEP].
_NUM_SPECIAL_TOKENS = 2


def _autocast():
    """Autocast to DTYPE for half-precision runs; a no-op for float32 or non-CUDA devices."""
    if DTYPE in (torch.bfloat16, torch.float16) and DEVICE.startswith("cuda"):
        return torch.autocast("cuda", dtype=DTYPE)
    return contextlib.nullcontext()


def _is_bigbird(model) -> bool:
    return getattr(model.config, "model_type", None) == "big_bird"


def load(m: ResolvedModel):
    """Load (and LRU-cache, `GENALM_MAX_RESIDENT_MODELS`, default 2) the tokenizer + model."""

    def _load() -> tuple:
        try:
            tokenizer = AutoTokenizer.from_pretrained(m.repo_id, revision=m.revision)
            architecture = m.architecture
            if architecture is None:
                config = AutoConfig.from_pretrained(
                    m.repo_id, revision=m.revision, trust_remote_code=True
                )
                architecture = "bigbird" if config.model_type == "big_bird" else "bert"
            if architecture == "bigbird":
                model = AutoModelForMaskedLM.from_pretrained(
                    m.repo_id, revision=m.revision, dtype=DTYPE
                )
            else:
                # auto_map registers only AutoModel, and it points at BertForMaskedLM.
                model = AutoModel.from_pretrained(
                    m.repo_id, revision=m.revision, trust_remote_code=True, dtype=DTYPE
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load GENA-LM model '{m.repo_id}' from HuggingFace: {exc}"
            ) from exc
        if not type(model).__name__.endswith("ForMaskedLM"):
            raise RuntimeError(
                f"'{m.repo_id}' loaded as {type(model).__name__}, not a masked-LM model; "
                "this server needs the MLM head (GENA-LM pre-trained checkpoints)."
            )
        tokenizer.padding_side = "right"
        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    return _cache.get_or_load(f"{m.repo_id}@{m.revision or 'main'}", _load)


def max_tokens(m: ResolvedModel, model) -> int:
    """Context limit in tokens (incl. [CLS]/[SEP]): the registry's value, else the config's."""
    if m.max_tokens is not None:
        return m.max_tokens
    # The tokenizer declares no model_max_length (huge sentinel), so read the config.
    declared = getattr(model.config, "max_position_embeddings", None)
    if isinstance(declared, int) and declared > 0:
        return declared
    return _FALLBACK_MAX_TOKENS


def check_token_lengths(lengths: list[int], limit: int) -> None:
    """Raise a clear `ValueError` if any tokenized sequence exceeds `limit` tokens."""
    for i, n in enumerate(lengths):
        if n > limit:
            raise ValueError(
                f"Sequence {i} tokenizes to {n} tokens, above this checkpoint's {limit}-token context "
                f"limit ([CLS] and [SEP] included). GENA-LM's BPE tokens cover a variable number of "
                f"bases: roughly {(limit - _NUM_SPECIAL_TOKENS) * _TYPICAL_BASES_PER_TOKEN // 1000} kb "
                "of typical human DNA fits, less for high-entropy sequence. Sequences are never "
                "truncated -- split it into shorter windows."
            )


def _encode(m: ResolvedModel, tokenizer, model, sequences: list[str]) -> list[list[int]]:
    """Token ids per sequence ([CLS] ... [SEP]), after the context-length check."""
    encoded = tokenizer(sequences, add_special_tokens=True, truncation=False)
    ids = encoded["input_ids"]
    check_token_lengths([len(x) for x in ids], max_tokens(m, model))
    return ids


def _pad(tokenizer, ids: list[list[int]]) -> dict:
    batch = tokenizer.pad(
        {"input_ids": ids}, padding=True, return_tensors="pt", return_attention_mask=True
    )
    return {
        "input_ids": batch["input_ids"].to(DEVICE),
        "attention_mask": batch["attention_mask"].to(DEVICE),
    }


def _forward_hidden_states(
    tokenizer, model, ids: list[list[int]], layer: int
) -> list[torch.Tensor]:
    """Run the model and return, per sequence, hidden-state `layer` trimmed to its real length.

    Each item is a `(num_tokens, hidden)` float32 tensor. BERT sequences share one padded
    forward pass; BigBird ones run one at a time (see the module docstring).
    """
    groups = [[x] for x in ids] if _is_bigbird(model) else [ids]
    per_sequence = []
    for group in groups:
        batch = _pad(tokenizer, group)
        with torch.no_grad(), _autocast():
            out = model(**batch, output_hidden_states=True)
        h = out.hidden_states[layer]
        per_sequence.extend(h[i, : len(row)].float() for i, row in enumerate(group))
    return per_sequence


def _num_layers(model) -> int:
    """Transformer layers; hidden states are indexed 0 (embeddings) .. this value (last)."""
    return int(model.config.num_hidden_layers)


def _parse_layer(layer: str | int | None, n_layers: int) -> int:
    if layer in (None, "last", "LAST"):
        return n_layers
    try:
        idx = int(layer)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"layer must be an integer (as int or string) or 'last'; got {layer!r}"
        ) from exc
    if idx < 0:
        idx = n_layers + 1 + idx
    if not (0 <= idx <= n_layers):
        raise ValueError(f"layer must be between 0 and {n_layers} (or 'last'); got {layer!r}")
    return idx


def model_info(m: ResolvedModel) -> dict:
    tokenizer, model = load(m)
    # Tiny dummy forward pass: confirms the model runs and reports the real hidden size.
    (last,) = _forward_hidden_states(
        tokenizer, model, _encode(m, tokenizer, model, ["ACGT" * 8]), -1
    )
    n_params = sum(p.numel() for p in model.parameters())
    raw_cfg = model.config.to_dict()
    safe_cfg = {
        k: v for k, v in raw_cfg.items() if isinstance(v, (int, float, str, bool)) or v is None
    }
    return {
        "repo_id": m.repo_id,
        "revision": m.revision,
        "architecture": "bigbird" if _is_bigbird(model) else "bert",
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "num_parameters": n_params,
        "vocab_size": len(tokenizer),
        "hidden_size": int(last.shape[-1]),
        "num_hidden_state_layers": _num_layers(model) + 1,
        "max_tokens": max_tokens(m, model),
        "single_nucleotide_tokenizer": is_single_nucleotide_tokenizer(tokenizer),
        "mask_token": tokenizer.mask_token,
        "config": safe_cfg,
    }


def list_embedding_layers(m: ResolvedModel, which: str) -> dict:
    """Enumerate hidden-state layer indices usable as `layer_name` in embed_sequence.

    `which="recommended"` is a heuristic (final layer and the one before it), not an
    AIRI recommendation; `which="all"` returns every index, 0 (input embeddings)
    through the last transformer layer.
    """
    if which not in ("recommended", "all"):
        raise ValueError("which must be 'recommended' or 'all'.")
    _, model = load(m)
    n_layers = _num_layers(model)
    if which == "all":
        layers = list(range(n_layers + 1))
        info = f"All {n_layers + 1} hidden-state layers (0=input embeddings, {n_layers}=final layer/'last')."
    else:
        layers = sorted({n_layers, max(n_layers - 1, 0)})
        info = (
            "Heuristic only: the final layer and the one before it. Note that the bert-base-t2t "
            "and bert-base-t2t-multi checkpoints have no final layernorm, so their last layer is "
            "the raw (large-magnitude) residual stream. Use which='all' to explore others."
        )
    return {"layers": layers, "num_hidden_state_layers": n_layers + 1, "info": info}


def compute_embeddings(
    m: ResolvedModel,
    sequences: list[str],
    layer: str | int | None,
    pooling: str,
) -> tuple[list[np.ndarray], int, int]:
    """Per-sequence embeddings: a 1D vector each for "mean", a (num_tokens, hidden) matrix for "per_token".

    Mean pooling averages over every token of the sequence, [CLS] and [SEP] included
    (padding excluded). Per-token rows are BPE tokens -- [CLS], one row per token, [SEP]
    -- not individual bases.
    """
    if not sequences:
        raise ValueError("sequences must be a non-empty list of DNA sequences.")
    if pooling not in ("mean", "per_token"):
        raise ValueError("pooling must be 'mean' or 'per_token'.")
    tokenizer, model = load(m)
    n_layers = _num_layers(model)
    idx = _parse_layer(layer, n_layers)
    seqs = [validate_sequence(s) for s in sequences]
    rows = _forward_hidden_states(tokenizer, model, _encode(m, tokenizer, model, seqs), idx)
    if pooling == "mean":
        return [r.mean(dim=0).cpu().numpy() for r in rows], idx, n_layers
    return [r.cpu().numpy() for r in rows], idx, n_layers


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
    (ids,) = _encode(m, tokenizer, model, [seq])
    input_ids = torch.tensor(ids, device=DEVICE)
    special = set(tokenizer.all_special_ids)
    positions = [i for i, t in enumerate(ids) if t not in special]
    mask_id = tokenizer.mask_token_id
    batch_size = max(1, _PLL_BATCH_TOKENS // len(ids))

    def logits_fn(batch_ids: torch.Tensor) -> torch.Tensor:
        # Full (batch, seq, 32k-vocab) logits would cost batch x seq x 128 KB (e.g. ~2 GB for
        # 32 x 512 tokens) although the PLL helper reads only the one masked position per row.
        # So run the encoder, apply the MLM head at each row's [MASK] only, and hand back a
        # zero-copy expand() to (batch, seq, vocab): logits[row, any col] is that row's
        # masked-position logits, which is exactly what the helper indexes. Each row has
        # exactly one [MASK] (validated input contains no mask tokens).
        # All copies have the same length, so BigBird uses one attention pattern for them.
        with _autocast():
            hidden = model.bert(input_ids=batch_ids, attention_mask=torch.ones_like(batch_ids))[0]
            rows = torch.arange(batch_ids.shape[0], device=batch_ids.device)
            cols = (batch_ids == mask_id).int().argmax(dim=1)
            masked_logits = model.cls(hidden[rows, cols])
        return masked_logits.unsqueeze(1).expand(-1, batch_ids.shape[1], -1)

    return masked_lm_pseudo_log_likelihood(
        input_ids, positions, mask_id, logits_fn, batch_size=batch_size
    )


def score_variant(
    m: ResolvedModel, sequence: str, alt_allele: str, position: int | None = None
) -> dict:
    """Whole-sequence PLL(mutated) - PLL(reference); see `dnalm_common.scoring`."""
    return score_snp_whole_sequence(
        sequence, alt_allele, lambda s: pseudo_log_likelihood(m, s), position=position
    )
