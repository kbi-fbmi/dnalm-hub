"""Model loading and inference helpers for HyenaDNA.

HyenaDNA checkpoints are causal (autoregressive) DNA language models built from
Hyena operators (implicit long convolutions evaluated by FFT) instead of
attention, shipped as ``trust_remote_code`` architectures on the HuggingFace Hub
(see e.g. https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf).

Key differences from the NT services:

- Causal, not masked: position t only sees bases 0..t. Hence ``generate_sequence``
  exists and ``score_snp`` uses the *exact* autoregressive log-likelihood (one
  forward pass per sequence) instead of a masked-LM pseudo-log-likelihood.
- Character tokenizer, one token per base. The stock tokenizer appends a ``[SEP]``
  token (and no BOS); this service tokenizes WITHOUT special tokens, so token i is
  base i everywhere (per-token embeddings, log-likelihood, generation). In
  pretraining the first token of a window was never a prediction target either.
- Long context (up to ~1M bases), memory linear in length. There is no attention
  mask and no KV cache. Padding is not used at all: a Hyena convolution is causal,
  so right padding would leave real positions mathematically unchanged, but the FFT
  size depends on the padded length and the floats would differ slightly. Batches
  are therefore grouped by exact length, so a batch result is identical to
  embedding each sequence alone (see ``_length_groups``).
- Hidden states: index 0 is the token embedding, 1..n_layer the raw residual
  stream after each Hyena block, and n_layer+1 ("last") the final LayerNorm output
  (what the LM head reads).
- Memory: the remote forward pass needs ~21 GiB for 1M bases (fp32 FFTs over all
  256 channels at once, plus a 4x-wide MLP over the whole sequence). This module
  runs the same blocks with the long FFT convolution split into channel chunks
  (reusing the remote ``fftconv``) and the pointwise MLP split into position
  chunks, and keeps only the requested hidden state (``_hidden_state``). Outputs
  match the remote forward to float32 rounding; peak memory is several times lower.
- The remote code runs its FFT convolution and residual stream in float32
  whatever the weight dtype, so float32 (the default) is the sensible dtype; the
  models are tiny (<7M parameters), so half precision saves next to nothing.
- Works unchanged on transformers 4.57 and 5.x: outputs were compared bit for bit.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from dnalm_common import (
    LRUModelCache,
    is_single_nucleotide_tokenizer,
    score_snp_whole_sequence,
    select_device,
    select_dtype,
    validate_sequence,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from .registry import ResolvedModel

DEVICE = select_device("HYENADNA_DEVICE")
DTYPE = select_dtype("HYENADNA_DTYPE")

_MAX_RESIDENT_MODELS = int(os.environ.get("HYENADNA_MAX_RESIDENT_MODELS", "2"))
_cache = LRUModelCache(max_entries=_MAX_RESIDENT_MODELS)

# Optional server-wide cap on bases per sequence, below the checkpoint's own limit:
# activation memory grows linearly with length (~11 GiB at 1M bases), and the
# GPU may be shared. Unset/0 = only the checkpoint limit applies.
MAX_BASES = int(os.environ.get("HYENADNA_MAX_BASES", "0") or 0)
# Bases per forward pass when a batch of equal-length sequences is stacked.
_MAX_BATCH_BASES = int(os.environ.get("HYENADNA_MAX_BATCH_BASES", "262144"))
# After a forward pass over at least this many bases, return PyTorch's cached
# activation memory to the driver: a 1M-base pass reserves ~11-14 GiB, which would
# otherwise stay claimed by this process on a GPU shared with other services.
_RELEASE_CACHE_ABOVE_BASES = int(os.environ.get("HYENADNA_RELEASE_CACHE_ABOVE_BASES", "32768"))
# Channels per FFT call and positions per pointwise (LayerNorm/MLP) call in the
# memory-lean forward pass (see _block_forward).
_FFT_CHANNEL_CHUNK = int(os.environ.get("HYENADNA_FFT_CHANNEL_CHUNK", "32"))
_POINTWISE_CHUNK = int(os.environ.get("HYENADNA_POINTWISE_CHUNK", "65536"))
# generate_sequence runs a full forward pass per new base (no recurrent/KV cache in
# the HF port), so the number of new bases per call is capped.
MAX_NEW_TOKENS = int(os.environ.get("HYENADNA_MAX_NEW_TOKENS", "2048"))

# Fallback for unregistered repos whose config lacks max_seq_len.
_FALLBACK_MAX_TOKENS = 1026
# Bases the model may emit in generate_sequence (N excluded on purpose).
GENERATION_ALPHABET = ("A", "C", "G", "T")


def load(m: ResolvedModel):
    """Load (and LRU-cache, `HYENADNA_MAX_RESIDENT_MODELS`, default 2) the tokenizer + model."""

    def _load() -> tuple:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                m.repo_id, revision=m.revision, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                m.repo_id, revision=m.revision, trust_remote_code=True, dtype=DTYPE
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HyenaDNA model '{m.repo_id}' from HuggingFace: {exc}"
            ) from exc
        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    return _cache.get_or_load(f"{m.repo_id}@{m.revision or 'main'}", _load)


def max_tokens(m: ResolvedModel, model=None) -> int:
    """Checkpoint context limit in tokens (= bases): the registry's value, else config.max_seq_len."""
    if m.max_tokens is not None:
        return m.max_tokens
    declared = getattr(getattr(model, "config", None), "max_seq_len", None)
    if isinstance(declared, int) and declared > 0:
        return declared
    return _FALLBACK_MAX_TOKENS


def effective_max_bases(checkpoint_limit: int) -> int:
    """The checkpoint limit, lowered to HYENADNA_MAX_BASES if that is set."""
    return min(checkpoint_limit, MAX_BASES) if MAX_BASES > 0 else checkpoint_limit


def check_lengths(lengths: list[int], checkpoint_limit: int) -> None:
    """Raise a clear `ValueError` if any sequence is longer than the allowed number of bases."""
    limit = effective_max_bases(checkpoint_limit)
    for i, n in enumerate(lengths):
        if n > limit:
            why = (
                f"this checkpoint's {checkpoint_limit}-base context limit"
                if limit == checkpoint_limit
                else f"this server's HYENADNA_MAX_BASES={limit} cap (checkpoint limit {checkpoint_limit})"
            )
            raise ValueError(
                f"Sequence {i} is {n} bases long, above {why}. Sequences are never truncated -- "
                "split it into shorter windows or use a longer-context checkpoint."
            )


def _encode(tokenizer, seq: str) -> list[int]:
    """Token ids for an already validated sequence: one id per base, no special tokens."""
    ids = tokenizer(seq, add_special_tokens=False)["input_ids"]
    if len(ids) != len(seq):  # guards custom repos with a different tokenizer
        raise RuntimeError(
            f"Tokenizer produced {len(ids)} tokens for {len(seq)} bases; expected one token per base."
        )
    return ids


def _length_groups(lengths: list[int], max_batch_bases: int) -> list[list[int]]:
    """Indices grouped by equal length, each group split so rows*length <= max_batch_bases.

    Equal-length rows need no padding, so batched results match single-sequence ones.
    """
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(lengths):
        by_len[n].append(i)
    groups = []
    for n, idxs in by_len.items():
        rows = max(1, max_batch_bases // max(n, 1))
        groups.extend(idxs[j : j + rows] for j in range(0, len(idxs), rows))
    return groups


def _release_cached_memory(n_bases: int) -> None:
    """Empty the CUDA caching allocator after large inputs (see _RELEASE_CACHE_ABOVE_BASES)."""
    if n_bases >= _RELEASE_CACHE_ABOVE_BASES and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _num_layers(model) -> int:
    return int(model.config.n_layer)


def _remote_fftconv(model):
    """The `fftconv` function of the model's own (pinned, trust_remote_code) module."""
    return sys.modules[type(model).__module__].fftconv


def _operator_forward(op, u: torch.Tensor, fftconv) -> torch.Tensor:
    """HyenaOperator.forward (remote code) with the long convolution done in channel chunks.

    Same ops in the same order; only `fftconv`, which is independent per channel, is
    called on `_FFT_CHANNEL_CHUNK` channels at a time, and big inputs are freed early.
    """
    seq_len = u.size(-2)
    l_filter = min(seq_len, op.l_max)
    projected = op.in_proj(u).transpose(1, 2)
    uc = op.short_filter(projected)[..., :l_filter]
    del projected
    *x, v = uc.split(op.d_model, dim=1)
    k = op.filter_fn.filter(l_filter)[0]
    k = k.transpose(0, 1).reshape(op.order - 1, op.d_model, l_filter)
    bias = op.filter_fn.bias.reshape(op.order - 1, op.d_model)
    for o, x_i in enumerate(reversed(x[1:])):
        v = op.dropout(v * x_i)
        out = torch.empty_like(v)
        for c in range(0, op.d_model, _FFT_CHANNEL_CHUNK):
            sl = slice(c, c + _FFT_CHANNEL_CHUNK)
            out[:, sl] = fftconv(v[:, sl], k[o][sl], bias[o][sl])
        v = out
    y = (v * x[0]).transpose(1, 2)
    del uc, x, v
    return op.out_proj(y)


def _block_forward(block, h: torch.Tensor, fftconv) -> torch.Tensor:
    """HyenaBlock.forward (remote code), with LayerNorm2 + MLP applied in position chunks."""
    residual = h.to(torch.float32)
    normed = block.norm1(residual.to(dtype=block.norm1.weight.dtype))
    mixed = _operator_forward(block.mixer, normed, fftconv)
    del normed
    residual = mixed + residual
    del mixed
    out = torch.empty_like(residual)
    for s in range(0, residual.shape[1], _POINTWISE_CHUNK):
        r = residual[:, s : s + _POINTWISE_CHUNK]
        out[:, s : s + _POINTWISE_CHUNK] = (
            block.mlp(block.norm2(r.to(block.norm2.weight.dtype))) + r
        )
    return out


def _hidden_state(model, input_ids: torch.Tensor, idx: int) -> torch.Tensor:
    """Hidden state `idx` (0=embeddings, 1..n=blocks, n+1=final norm) without keeping the others.

    Mirrors HyenaLMBackbone.forward of the pinned remote code (eval mode: dropout is a
    no-op) using the memory-lean `_block_forward`, stopping at the requested layer.
    """
    backbone = model.hyena.backbone
    fftconv = _remote_fftconv(model)
    h = backbone.embeddings(input_ids)
    if idx == 0:
        return h
    for i, layer in enumerate(backbone.layers, start=1):
        h = _block_forward(layer, h, fftconv)
        if i == idx:
            return h
    return backbone.ln_f(h.to(dtype=backbone.ln_f.weight.dtype))


def _logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """LM logits (batch, L, vocab) via the memory-lean forward (same as model(...).logits)."""
    h = _hidden_state(model, input_ids, _num_layers(model) + 1)
    return model.lm_head(h).float()


@contextlib.contextmanager
def _oom_as_user_error(n_bases: int):
    """Turn a CUDA OOM into a readable ValueError (and give the memory back)."""
    try:
        yield
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise ValueError(
            f"Out of GPU memory processing {n_bases} bases (the GPU may be shared with other "
            "models). Retry later, use a shorter sequence, or set HYENADNA_MAX_BASES."
        ) from exc


def _parse_layer(layer: str | int | None, n_hidden: int) -> int:
    """Map "last"/int/negative-int to an index in [0, n_hidden - 1]."""
    last = n_hidden - 1
    if layer in (None, "last", "LAST"):
        return last
    try:
        idx = int(layer)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"layer must be an integer (as int or string) or 'last'; got {layer!r}"
        ) from exc
    if idx < 0:
        idx = n_hidden + idx
    if not (0 <= idx <= last):
        raise ValueError(f"layer must be between 0 and {last} (or 'last'); got {layer!r}")
    return idx


def model_info(m: ResolvedModel) -> dict:
    tokenizer, model = load(m)
    n_params = sum(p.numel() for p in model.parameters())
    raw_cfg = model.config.to_dict()
    safe_cfg = {
        k: v for k, v in raw_cfg.items() if isinstance(v, (int, float, str, bool)) or v is None
    }
    limit = max_tokens(m, model)
    return {
        "repo_id": m.repo_id,
        "revision": m.revision,
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "num_parameters": n_params,
        "vocab_size": len(tokenizer),
        "hidden_size": int(model.config.d_model),
        "num_hidden_state_layers": _num_layers(model) + 2,
        "max_tokens": limit,
        "max_bases": effective_max_bases(limit),
        "single_nucleotide_tokenizer": is_single_nucleotide_tokenizer(tokenizer),
        "causal": True,
        "config": safe_cfg,
    }


def list_embedding_layers(m: ResolvedModel, which: str) -> dict:
    """Enumerate hidden-state layer indices usable as `layer_name` in embed_sequence.

    `which="recommended"` is a heuristic (final-norm output and the last block's raw
    output), not a HyenaDNA-author recommendation; `which="all"` returns every index.
    """
    _, model = load(m)
    n = _num_layers(model)
    n_hidden = n + 2
    if which == "all":
        layers = list(range(n_hidden))
        info = (
            f"All {n_hidden} hidden-state layers: 0=token embeddings, 1..{n}=output of each Hyena "
            f"block (un-normalized residual stream), {n + 1}=final LayerNorm output ('last')."
        )
    elif which == "recommended":
        layers = [n, n + 1]
        info = (
            f"Heuristic only: {n + 1} ('last', final LayerNorm output, what the HyenaDNA examples "
            f"and LM head use) and {n} (last block, before the norm). Use which='all' to explore others."
        )
    else:
        raise ValueError("which must be 'recommended' or 'all'.")
    return {"layers": layers, "num_hidden_state_layers": n_hidden, "info": info}


def compute_embeddings(
    m: ResolvedModel,
    sequences: list[str],
    layer: str | int | None,
    pooling: str,
) -> tuple[list[np.ndarray], int, int]:
    """Per-sequence embeddings: a 1D vector each for "mean", a (num_bases, hidden) matrix for "per_token".

    Mean pooling averages over every base position. HyenaDNA is causal, so row t
    summarizes only bases 0..t: the mean is an average of prefix summaries, weighted
    toward the start of the sequence, and only the last row has seen the whole input.
    Per-token rows are exactly one per base (no special tokens).
    """
    if not sequences:
        raise ValueError("sequences must be a non-empty list of DNA sequences.")
    if pooling not in ("mean", "per_token"):
        raise ValueError("pooling must be 'mean' or 'per_token'.")
    tokenizer, model = load(m)
    seqs = [validate_sequence(s) for s in sequences]
    check_lengths([len(s) for s in seqs], max_tokens(m, model))
    n_layers = _num_layers(model)
    idx = _parse_layer(layer, n_layers + 2)

    results: list[np.ndarray | None] = [None] * len(seqs)
    for group in _length_groups([len(s) for s in seqs], _MAX_BATCH_BASES):
        ids = torch.tensor([_encode(tokenizer, seqs[i]) for i in group], device=DEVICE)
        with torch.no_grad(), _oom_as_user_error(ids.numel()):
            h = _hidden_state(model, ids, idx).float()
        out = h.mean(dim=1) if pooling == "mean" else h
        for row, i in enumerate(group):
            results[i] = out[row].cpu().numpy()
        del h, out
    _release_cached_memory(max(len(s) for s in seqs))
    return results, idx, n_layers + 1


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


def causal_log_likelihood(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """Exact autoregressive log-likelihood: sum_t log P(x_t | x_<t) for t = 1..L-1.

    `logits` is (L, vocab) from one forward pass over `input_ids` (L,). Position 0
    has no context and is not scored (same as HyenaDNA pretraining, which has no
    BOS token). Summed in float64: over ~1M bases the total is ~-1e6, where float32
    resolution (~0.1) would swamp a single-SNP delta. Model-agnostic; a candidate
    for dnalm_common.
    """
    if input_ids.dim() != 1 or logits.dim() != 2 or logits.shape[0] != input_ids.shape[0]:
        raise ValueError("expected logits (L, vocab) and input_ids (L,) for one sequence.")
    if input_ids.shape[0] < 2:
        return 0.0
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    token_log_probs = log_probs.gather(-1, input_ids[1:].unsqueeze(-1))
    return float(token_log_probs.double().sum().item())


def log_likelihood(m: ResolvedModel, sequence: str) -> float:
    """Exact causal log-likelihood of `sequence` (natural log, first base unscored)."""
    tokenizer, model = load(m)
    seq = validate_sequence(sequence)
    check_lengths([len(seq)], max_tokens(m, model))
    ids = torch.tensor([_encode(tokenizer, seq)], device=DEVICE)
    with torch.no_grad(), _oom_as_user_error(len(seq)):
        logits = _logits(model, ids)[0]
    result = causal_log_likelihood(logits, ids[0])
    del logits
    _release_cached_memory(len(seq))
    return result


def score_variant(
    m: ResolvedModel, sequence: str, alt_allele: str, position: int | None = None
) -> dict:
    """Whole-sequence log P(mutated) - log P(reference); see `dnalm_common.scoring`."""
    return score_snp_whole_sequence(
        sequence, alt_allele, lambda s: log_likelihood(m, s), position=position
    )


def sample_next(
    logits: torch.Tensor,
    allowed_ids: list[int],
    temperature: float,
    top_k: int,
    generator: torch.Generator | None = None,
) -> int:
    """Pick the next token id from `logits` (vocab,), restricted to `allowed_ids`.

    temperature <= 0 means greedy; top_k <= 0 (or >= len(allowed_ids)) means no top-k cut.
    """
    restricted = logits.float()[allowed_ids]
    if temperature <= 0:
        return allowed_ids[int(torch.argmax(restricted))]
    scores = restricted / temperature
    if 0 < top_k < len(allowed_ids):
        kth = torch.topk(scores, top_k).values[-1]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    choice = torch.multinomial(probs.cpu(), 1, generator=generator)
    return allowed_ids[int(choice)]


def generate(
    m: ResolvedModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int | None = None,
) -> dict:
    """Autoregressively extend `prompt` by `max_new_tokens` bases (A/C/G/T only).

    The HF port has no recurrent inference mode, so every step re-runs the model on
    the whole prefix: cost grows with prompt length x max_new_tokens.
    """
    if not isinstance(max_new_tokens, int) or not (1 <= max_new_tokens <= MAX_NEW_TOKENS):
        raise ValueError(
            f"max_new_tokens must be an integer between 1 and {MAX_NEW_TOKENS} "
            "(HYENADNA_MAX_NEW_TOKENS)."
        )
    if temperature < 0:
        raise ValueError("temperature must be >= 0 (0 = greedy).")
    if top_k < 0:
        raise ValueError("top_k must be >= 0 (0 = no top-k filtering).")
    tokenizer, model = load(m)
    seq = validate_sequence(prompt)
    limit = effective_max_bases(max_tokens(m, model))
    if len(seq) + max_new_tokens > limit:
        raise ValueError(
            f"prompt ({len(seq)} bases) + max_new_tokens ({max_new_tokens}) exceeds the "
            f"{limit}-base limit of this checkpoint/server. Shorten the prompt or ask for fewer bases."
        )
    allowed = [tokenizer.convert_tokens_to_ids(b) for b in GENERATION_ALPHABET]
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    ids = torch.tensor([_encode(tokenizer, seq)], device=DEVICE)
    new_ids: list[int] = []
    with torch.no_grad(), _oom_as_user_error(len(seq) + max_new_tokens):
        for _ in range(max_new_tokens):
            logits = _logits(model, ids)[0, -1]
            nxt = sample_next(logits, allowed, temperature, top_k, generator)
            new_ids.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    _release_cached_memory(int(ids.shape[1]))
    generated = "".join(tokenizer.convert_ids_to_tokens(new_ids))
    return {
        "prompt": seq,
        "generated_sequence": generated,
        "full_sequence": seq + generated,
        "num_new_tokens": len(new_ids),
        "num_new_bases": len(generated),
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
    }
