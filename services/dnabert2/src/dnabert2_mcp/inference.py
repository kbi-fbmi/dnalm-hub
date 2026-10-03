"""Model loading and inference helpers for DNABERT-2.

DNABERT-2 (https://huggingface.co/zhihan1996/DNABERT-2-117M) is a MosaicBERT-style
masked DNA language model shipped as ``trust_remote_code`` (``bert_layers.py``).
Key differences from NTv2 (``ntv2_mcp.inference``):

- BPE tokenizer: tokens are variable-length (1 to ~16 bases, ~4.7 bases/token on
  random sequence), wrapped in ``[CLS] ... [SEP]``; every 'N' becomes one
  ``[UNK]``. Token indices don't line up with sequence offsets, so there is no
  single-position masked prediction, and SNP scoring uses the tokenizer-agnostic
  whole-sequence pseudo-log-likelihood delta from ``dnalm_common.scoring``.
- ALiBi instead of position embeddings: no hard limit in the weights, but we cap
  inputs at the registry's ``max_tokens`` (512, see ``registry.py``) and reject
  longer ones with a clear error, never truncating.
- Remote-code quirks handled in ``load``:
  * ``transformers<5`` (5.x builds the model on the meta device, where the
    remote ALiBi setup crashes), and the config is loaded as the stock
    ``BertConfig`` and passed in explicitly -- the documented workaround for
    transformers > 4.28.
  * No Triton. The remote code prefers a bundled Triton flash-attention
    kernel written for the pre-2.0 Triton API (``tl.dot(..., trans_b=True)``),
    which fails on current Triton. The service excludes the package (pyproject.toml), skips
    transformers' import check for it (``_without_triton_import_check``), and
    forces the remote module's kernel handle to ``None`` so attention always
    runs in plain PyTorch.
- No ``output_hidden_states``: the remote ``BertModel`` ignores that flag, and
  its encoder runs on *unpadded* tokens (``(total_tokens, hidden)``, padding
  removed across the batch). Per-layer embeddings are therefore captured with a
  forward hook on the requested layer and scattered back into a padded
  ``(batch, seq, hidden)`` tensor (``_hidden_states``).
- The remote attention adds a float32 ALiBi bias to the scores, so
  half-precision runs need ``torch.autocast`` (see ``_autocast``).
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from unittest import mock

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
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, dynamic_module_utils

from .registry import ResolvedModel

DEVICE = select_device("DNABERT2_DEVICE")
DTYPE = select_dtype("DNABERT2_DTYPE")

_MAX_RESIDENT_MODELS = int(os.environ.get("DNABERT2_MAX_RESIDENT_MODELS", "2"))
_PLL_BATCH_SIZE = int(os.environ.get("DNABERT2_PLL_BATCH_SIZE", "16"))
_cache = LRUModelCache(max_entries=_MAX_RESIDENT_MODELS)

# Fallback for unregistered repos: DNABERT-2's tokenizer declares no usable model_max_length.
_FALLBACK_MAX_TOKENS = 512
# Only used to phrase the over-length error; BPE token length varies with the sequence.
_APPROX_BASES_PER_TOKEN = 4.7

# Forward hooks are registered on shared modules, so hooked forward passes must
# not interleave across request threads.
_hook_lock = threading.Lock()


def _autocast():
    """Autocast to DTYPE for half-precision runs; a no-op for float32 or non-CUDA devices."""
    if DTYPE in (torch.bfloat16, torch.float16) and DEVICE.startswith("cuda"):
        return torch.autocast("cuda", dtype=DTYPE)
    return contextlib.nullcontext()


@contextlib.contextmanager
def _without_triton_import_check():
    """Let transformers load the remote code although `triton` is not installed.

    `bert_layers.py` imports its Triton kernel inside try/except (and falls back to
    PyTorch attention on ImportError), but transformers also pre-checks the imports
    of every relative module it copies -- including `flash_attn_triton.py`, which
    imports triton unconditionally -- and would refuse to load. Drop `triton` from
    that check only; the real import still fails and triggers the fallback.
    """
    original = dynamic_module_utils.get_imports

    def get_imports(filename):
        return [name for name in original(filename) if name != "triton"]

    with mock.patch.object(dynamic_module_utils, "get_imports", get_imports):
        yield


def _force_pytorch_attention(model) -> None:
    """Make the remote attention use its PyTorch path even if triton is importable."""
    remote_module = sys.modules.get(type(model).__module__)
    if remote_module is not None and hasattr(remote_module, "flash_attn_qkvpacked_func"):
        remote_module.flash_attn_qkvpacked_func = None


def load(m: ResolvedModel):
    """Load (and LRU-cache, `DNABERT2_MAX_RESIDENT_MODELS`, default 2) the tokenizer + MLM model.

    One `BertForMaskedLM` serves every tool: embeddings run its `.bert` encoder,
    `score_snp` needs the MLM head.
    """

    def _load() -> tuple:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                m.repo_id, revision=m.revision, trust_remote_code=True
            )
            config = BertConfig.from_pretrained(m.repo_id, revision=m.revision)
            with _without_triton_import_check():
                model = AutoModelForMaskedLM.from_pretrained(
                    m.repo_id,
                    revision=m.revision,
                    config=config,
                    trust_remote_code=True,
                    dtype=DTYPE,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load DNABERT-2 model '{m.repo_id}' from HuggingFace: {exc}"
            ) from exc
        if not (hasattr(model, "bert") and hasattr(model.bert, "encoder")):
            raise RuntimeError(
                f"'{m.repo_id}' did not load as a DNABERT-2 (MosaicBERT) BertForMaskedLM "
                f"(got {type(model).__name__})."
            )
        _force_pytorch_attention(model)
        tokenizer.padding_side = "right"
        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    return _cache.get_or_load(f"{m.repo_id}@{m.revision or 'main'}", _load)


def max_tokens(m: ResolvedModel, tokenizer) -> int:
    """Context limit in tokens (including [CLS]/[SEP]): the registry's value, else the tokenizer's."""
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
                f"limit (BPE tokens average ~{_APPROX_BASES_PER_TOKEN:g} bp on non-repetitive DNA, so "
                f"roughly {int((limit - 2) * _APPROX_BASES_PER_TOKEN)} bp; each 'N' costs a whole token). "
                "Sequences are never truncated -- split it into shorter windows."
            )


def _tokenize(m: ResolvedModel, tokenizer, sequences: list[str]) -> dict:
    batch = tokenizer(
        sequences,
        add_special_tokens=True,
        truncation=False,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    check_token_lengths(batch["attention_mask"].sum(dim=1).tolist(), max_tokens(m, tokenizer))
    return {
        "input_ids": batch["input_ids"].to(DEVICE),
        "attention_mask": batch["attention_mask"].to(DEVICE),
    }


def num_layers(model) -> int:
    """Number of transformer layers; hidden-state indices run 0 (embeddings) .. num_layers."""
    return len(model.bert.encoder.layer)


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


def repad(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Scatter unpadded `(total_tokens, hidden)` activations back to `(batch, seq, hidden)`.

    The remote encoder unpads with `nonzero(attention_mask.flatten())`, i.e. real
    tokens in row-major order -- the same order boolean-mask indexing uses. Padding
    rows are zeros. Already padded (3D) input is returned unchanged.
    """
    if hidden.dim() == 3:
        return hidden
    mask = attention_mask.bool()
    if hidden.shape[0] != int(mask.sum()):
        raise RuntimeError(
            f"Captured {hidden.shape[0]} token rows but the attention mask has {int(mask.sum())} "
            "real tokens; the remote encoder's unpadding changed."
        )
    out = hidden.new_zeros((*mask.shape, hidden.shape[-1]))
    out[mask] = hidden
    return out


def _hidden_states(model, batch: dict, layer_idx: int) -> torch.Tensor:
    """Padded `(batch, seq, hidden)` output of hidden-state layer `layer_idx`.

    0 is the embedding block (token + token-type embeddings, LayerNorm); i >= 1 is
    the output of encoder layer i. Captured with a forward hook because the remote
    model doesn't return per-layer states.
    """
    module = model.bert.embeddings if layer_idx == 0 else model.bert.encoder.layer[layer_idx - 1]
    captured: list[torch.Tensor] = []
    with _hook_lock:
        handle = module.register_forward_hook(lambda _mod, _inp, out: captured.append(out))
        try:
            with torch.no_grad(), _autocast():
                model.bert(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        finally:
            handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one activation from layer {layer_idx}, got {len(captured)}.")
    return repad(captured[0], batch["attention_mask"])


def model_info(m: ResolvedModel) -> dict:
    tokenizer, model = load(m)
    n_params = sum(p.numel() for p in model.parameters())
    raw_cfg = model.config.to_dict()
    safe_cfg = {
        k: v for k, v in raw_cfg.items() if isinstance(v, (int, float, str, bool)) or v is None
    }
    return {
        "repo_id": m.repo_id,
        "revision": m.revision,
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "num_parameters": n_params,
        "vocab_size": len(tokenizer),
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_state_layers": num_layers(model) + 1,
        "max_tokens": max_tokens(m, tokenizer),
        "single_nucleotide_tokenizer": is_single_nucleotide_tokenizer(tokenizer),
        "mask_token": tokenizer.mask_token,
        "config": safe_cfg,
    }


def list_embedding_layers(m: ResolvedModel, which: str) -> dict:
    """Enumerate hidden-state layer indices usable as `layer_name` in embed_sequence.

    `which="recommended"` is a heuristic (final layer and the one before it), not a
    recommendation from the DNABERT-2 authors (their examples use the final layer);
    `which="all"` returns every index, 0 (embedding block) through the last layer.
    """
    _, model = load(m)
    n_layers = num_layers(model)
    all_layers = list(range(n_layers + 1))
    if which == "all":
        layers = all_layers
        info = f"All {len(all_layers)} hidden-state layers (0=embedding block, {n_layers}=final layer/'last')."
    elif which == "recommended":
        layers = sorted({n_layers, max(n_layers - 1, 0)})
        info = (
            "Heuristic only: the final layer and the one before it. The DNABERT-2 model card "
            "mean-pools the final layer; use which='all' to explore others."
        )
    else:
        raise ValueError("which must be 'recommended' or 'all'.")
    return {"layers": layers, "num_hidden_state_layers": n_layers + 1, "info": info}


def compute_embeddings(
    m: ResolvedModel,
    sequences: list[str],
    layer: str | int | None,
    pooling: str,
) -> tuple[list[np.ndarray], int, int]:
    """Per-sequence embeddings: a 1D vector each for "mean", a (num_tokens, hidden) matrix for "per_token".

    Mean pooling averages over every non-padding token, [CLS] and [SEP] included --
    the recipe on the DNABERT-2 model card (`torch.mean(hidden_states[0], dim=0)`).
    Per-token rows are BPE tokens ([CLS] first, [SEP] last), not individual bases,
    trimmed to each sequence's real length.
    """
    if not sequences:
        raise ValueError("sequences must be a non-empty list of DNA sequences.")
    if pooling not in ("mean", "per_token"):
        raise ValueError("pooling must be 'mean' or 'per_token'.")
    tokenizer, model = load(m)
    n_layers = num_layers(model)
    idx = _parse_layer(layer, n_layers)
    seqs = [validate_sequence(s) for s in sequences]
    batch = _tokenize(m, tokenizer, seqs)
    h = _hidden_states(model, batch, idx).float()
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


def log_likelihood(m: ResolvedModel, sequence: str) -> float:
    """Masked-LM pseudo-log-likelihood of `sequence`, summed over all non-special tokens.

    [CLS], [SEP] and the [UNK] tokens standing in for 'N' are not scored.
    """
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


def score_variant(
    m: ResolvedModel, sequence: str, alt_allele: str, position: int | None = None
) -> dict:
    """Whole-sequence PLL(mutated) - PLL(reference); see `dnalm_common.scoring`."""
    return score_snp_whole_sequence(
        sequence, alt_allele, lambda s: log_likelihood(m, s), position=position
    )
