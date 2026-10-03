"""Model loading and inference for GENERator (GenerTeam).

GENERator checkpoints are Llama-architecture causal DNA language models with a
6-mer tokenizer and a 16384-token (~98 kb) context
(https://huggingface.co/GenerTeam/GENERator-v2-eukaryote-1.2b-base).

Key points (see also README.md):

- No ``trust_remote_code``: the weights load into the stock ``LlamaForCausalLM``
  and the tokenizer is re-implemented from ``vocab.txt`` in ``kmer.py``.
- 6-mer policy. Embedding and generation need a whole number of 6-mers. By
  default a sequence whose length is not a multiple of 6 is rejected with a clear
  error; ``remainder="trim_left"`` (the authors' recipe) or ``"pad_left"`` opt in
  to an explicit, reported adjustment. ``score_snp`` needs no adjustment: its
  log-likelihood marginalizes over a partial last 6-mer (see ``log_likelihood``).
- Embedding recipe from the model cards: input ``<s> + DNA`` (v1) or
  ``<s> + DNA + <s>`` (v2), right-padded with an attention mask; mean pooling
  over every real token (both ``<s>`` included) or the last token.
- ``score_snp`` uses the exact causal log-likelihood of the whole sequence.
  Log-softmax runs on fp32 logits (the LM head is applied in fp32) because bf16
  logits are too coarse for small likelihood differences.
- Generation is a plain KV-cached sampling loop restricted to DNA 6-mers, with
  the authors' per-base (marginal) selection as the default mode.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from dnalm_common import (
    LRUModelCache,
    score_snp_whole_sequence,
    select_device,
    select_dtype,
    validate_sequence,
)
from huggingface_hub import hf_hub_download
from transformers import LlamaForCausalLM

from .kmer import KmerVocab, apply_remainder_policy
from .registry import ResolvedModel

DEVICE = select_device("GENERATOR_DEVICE")
DTYPE = select_dtype("GENERATOR_DTYPE")

_MAX_RESIDENT_MODELS = int(os.environ.get("GENERATOR_MAX_RESIDENT_MODELS", "1"))
MAX_NEW_TOKENS = int(os.environ.get("GENERATOR_MAX_NEW_TOKENS", "2048"))
_cache = LRUModelCache(max_entries=_MAX_RESIDENT_MODELS)

POOLINGS = ("mean", "last_token", "per_token")
SAMPLING_MODES = ("base", "token")


class Loaded:
    """Everything one checkpoint needs at inference time."""

    def __init__(self, vocab: KmerVocab, model: LlamaForCausalLM, max_tokens: int):
        self.vocab = vocab
        self.model = model
        self.max_tokens = max_tokens
        device = model.device
        # fp32 copy of the LM head: log-probabilities from bf16 logits are too coarse.
        self.lm_head_fp32 = model.lm_head.weight.detach().float()
        self.kmer_ids = torch.tensor(vocab.kmer_ids, device=device)
        kmer_base, flat_to_id = vocab.base_tables()
        self.kmer_base = torch.tensor(kmer_base, device=device)  # (k, 4**k)
        self.flat_to_id = torch.tensor(flat_to_id, device=device)
        self.powers = torch.tensor([4 ** (vocab.k - 1 - p) for p in range(vocab.k)], device=device)

    def logits_fp32(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden.float(), self.lm_head_fp32)


def load(m: ResolvedModel) -> Loaded:
    """Load (and LRU-cache, `GENERATOR_MAX_RESIDENT_MODELS`, default 1) vocabulary + model."""

    def _load() -> Loaded:
        try:
            vocab_path = hf_hub_download(m.repo_id, "vocab.txt", revision=m.revision)
            cfg_path = hf_hub_download(m.repo_id, "tokenizer_config.json", revision=m.revision)
            with open(cfg_path, encoding="utf-8") as f:
                k = int(json.load(f).get("k", 6))
            with open(vocab_path, encoding="utf-8") as f:
                vocab = KmerVocab.from_vocab_text(f.read(), k)
            # Stock Llama class on purpose: the repos' GENERatorForCausalLM subclass only
            # adds helpers, and skipping it means no remote code is executed.
            model = LlamaForCausalLM.from_pretrained(
                m.repo_id, revision=m.revision, dtype=DTYPE, trust_remote_code=False
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load GENERator model '{m.repo_id}' from HuggingFace: {exc}"
            ) from exc
        model.to(DEVICE)
        model.eval()
        limit = m.max_tokens or int(model.config.max_position_embeddings)
        return Loaded(vocab, model, limit)

    return _cache.get_or_load(f"{m.repo_id}@{m.revision or 'main'}", _load)


def check_token_lengths(lengths: list[int], limit: int, k: int = 6) -> None:
    """Raise a clear `ValueError` if any input (special tokens included) exceeds `limit` tokens."""
    for i, n in enumerate(lengths):
        if n > limit:
            raise ValueError(
                f"Sequence {i} needs {n} tokens, above this checkpoint's {limit}-token context "
                f"limit (about {(limit - 2) * k} bp at {k} bases per token). Sequences are never "
                "truncated -- split it into shorter windows."
            )


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


def prepare_sequences(
    sequences: list[str], remainder: str, k: int = 6
) -> tuple[list[str], list[int]]:
    """Validate and apply the 6-mer remainder policy; returns `(sequences, adjusted_bases)`."""
    if not sequences:
        raise ValueError("sequences must be a non-empty list of DNA sequences.")
    out, adjusted = [], []
    for i, s in enumerate(sequences):
        seq = validate_sequence(s)
        try:
            seq, n = apply_remainder_policy(seq, remainder, k)
        except ValueError as exc:
            raise ValueError(f"Sequence {i}: {exc}" if len(sequences) > 1 else str(exc)) from exc
        out.append(seq)
        adjusted.append(n)
    return out, adjusted


def _embedding_ids(m: ResolvedModel, lm: Loaded, seq: str) -> list[int]:
    bos = lm.vocab.bos_id
    return [bos, *lm.vocab.encode(seq), *([bos] if m.separator else [])]


def _hidden_states(lm: Loaded, ids: list[list[int]]):
    """Right-padded batch forward through the decoder; returns (hidden_states tuple, mask)."""
    width = max(len(x) for x in ids)
    pad = lm.vocab.pad_id
    input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in ids], device=DEVICE)
    mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids], device=DEVICE)
    with torch.inference_mode():
        out = lm.model.model(
            input_ids=input_ids, attention_mask=mask, output_hidden_states=True, use_cache=False
        )
    return out.hidden_states, mask


def model_info(m: ResolvedModel) -> dict:
    lm = load(m)
    hidden, _ = _hidden_states(lm, [_embedding_ids(m, lm, "ACGTAC" * 4)])
    cfg = lm.model.config.to_dict()
    safe_cfg = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool)) or v is None}
    return {
        "repo_id": m.repo_id,
        "revision": m.revision,
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "num_parameters": sum(p.numel() for p in lm.model.parameters()),
        "vocab_size": len(lm.vocab.token_to_id),
        "hidden_size": int(hidden[-1].shape[-1]),
        "num_hidden_state_layers": len(hidden),
        "max_tokens": lm.max_tokens,
        "bases_per_token": lm.vocab.k,
        "max_bases": (lm.max_tokens - 2) * lm.vocab.k,
        "embedding_separator_token": m.separator,
        "single_nucleotide_tokenizer": False,
        "causal": True,
        "config": safe_cfg,
    }


def list_embedding_layers(m: ResolvedModel, which: str) -> dict:
    """Enumerate hidden-state layer indices usable as `layer_name` in embed_sequence."""
    lm = load(m)
    n_layers = int(lm.model.config.num_hidden_layers)
    all_layers = list(range(n_layers + 1))
    if which == "all":
        layers = all_layers
        info = (
            f"All {len(all_layers)} hidden-state layers (0=input embeddings, "
            f"{n_layers}=final layer after the last RMSNorm/'last')."
        )
    elif which == "recommended":
        layers = sorted({n_layers, max(n_layers - 1, 0)})
        info = (
            "Heuristic only: the final layer and the one before it. The GENERator model cards "
            "embed with the final layer (hidden_states[-1]); use which='all' to explore others."
        )
    else:
        raise ValueError("which must be 'recommended' or 'all'.")
    return {"layers": layers, "num_hidden_state_layers": n_layers + 1, "info": info}


def compute_embeddings(
    m: ResolvedModel,
    sequences: list[str],
    layer: str | int | None,
    pooling: str,
    remainder: str = "reject",
) -> tuple[list[np.ndarray], int, int, list[int]]:
    """Embeddings per sequence; returns `(embeddings, layer_index, num_layers, adjusted_bases)`.

    Input is ``<s> + 6-mers`` (+ a trailing ``<s>`` for v2 checkpoints), as in the
    model cards. "mean" averages every real token, ``<s>`` markers included (the
    cards' mean-pooling option); "last_token" is the final real token (the v2
    separator, or the last 6-mer for v1); "per_token" returns one row per token.
    """
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {list(POOLINGS)}; got {pooling!r}.")
    seqs, adjusted = prepare_sequences(sequences, remainder)
    lm = load(m)
    ids = [_embedding_ids(m, lm, s) for s in seqs]
    check_token_lengths([len(x) for x in ids], lm.max_tokens, lm.vocab.k)
    hidden, mask = _hidden_states(lm, ids)
    n_layers = len(hidden) - 1
    idx = _parse_layer(layer, n_layers)
    h = hidden[idx].float()
    lengths = [len(x) for x in ids]

    if pooling == "mean":
        weights = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * weights).sum(dim=1) / weights.sum(dim=1)
        return list(pooled.cpu().numpy()), idx, n_layers, adjusted
    if pooling == "last_token":
        last = h[torch.arange(len(ids), device=h.device), torch.tensor(lengths) - 1]
        return list(last.cpu().numpy()), idx, n_layers, adjusted
    return [h[i, :n].cpu().numpy() for i, n in enumerate(lengths)], idx, n_layers, adjusted


def compare_sequences(
    m: ResolvedModel,
    sequence_a: str,
    sequence_b: str,
    layer: str | int | None,
    remainder: str = "reject",
) -> tuple[float, int]:
    embeddings, layer_idx, _, _ = compute_embeddings(
        m, [sequence_a, sequence_b], layer, "mean", remainder
    )
    a, b = embeddings
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return (float(np.dot(a, b) / denom) if denom > 0 else 0.0), layer_idx


def log_likelihood(m: ResolvedModel, sequence: str) -> float:
    """Exact causal log P(sequence | <s>) in nats, summed over the whole sequence.

    The sequence is cut into 6-mers from the left. Each complete 6-mer contributes
    log P(token | preceding tokens) under the full-vocabulary softmax. Two cases
    are marginalized exactly instead of dropping or padding bases:

    - a 6-mer containing ``N``: log of the summed probability of every 6-mer that
      matches the known bases (it is fed back to the model as ``<oov>``, exactly
      as the stock tokenizer encodes it);
    - a trailing partial 6-mer of r < 6 bases: log of the summed probability of
      every 6-mer that starts with those r bases.

    So any length works, and for an A/C/G/T sequence whose length is a multiple of
    6 this is the model's exact joint log-likelihood. (The model cards' own
    ``score_sequence`` helper reports per-base marginals instead.)
    """
    seq = validate_sequence(sequence)
    lm = load(m)
    vocab, k = lm.vocab, lm.vocab.k
    n_full = len(seq) // k
    full, tail = seq[: n_full * k], seq[n_full * k :]
    input_ids = [vocab.bos_id, *vocab.encode(full)]
    check_token_lengths([len(input_ids) + (1 if tail else 0)], lm.max_tokens, k)
    with torch.inference_mode():
        out = lm.model.model(input_ids=torch.tensor([input_ids], device=DEVICE), use_cache=False)
        # Position t predicts token t+1; the last position predicts the partial tail.
        log_probs = torch.log_softmax(lm.logits_fp32(out.last_hidden_state[0]), dim=-1)
        targets = [full[i : i + k] for i in range(0, len(full), k)] + ([tail] if tail else [])
        total = 0.0
        regular = [
            (t, vocab.token_to_id[p]) for t, p in enumerate(targets) if len(p) == k and "N" not in p
        ]
        if regular:
            rows = torch.tensor([t for t, _ in regular], device=DEVICE)
            cols = torch.tensor([i for _, i in regular], device=DEVICE)
            total += log_probs[rows, cols].sum().item()
        for t, pattern in enumerate(targets):
            if len(pattern) < k or "N" in pattern:
                ids = torch.tensor(vocab.matching_kmer_ids(pattern), device=DEVICE)
                total += torch.logsumexp(log_probs[t, ids], dim=0).item()
    return total


def score_variant(
    m: ResolvedModel, sequence: str, alt_allele: str, position: int | None = None
) -> dict:
    """Whole-sequence log P(mutated) - log P(reference); see `dnalm_common.scoring`."""
    return score_snp_whole_sequence(
        sequence, alt_allele, lambda s: log_likelihood(m, s), position=position
    )


def validate_generation_args(
    max_new_tokens: int, temperature: float, top_k: int, sampling: str
) -> None:
    if not isinstance(max_new_tokens, int) or not (1 <= max_new_tokens <= MAX_NEW_TOKENS):
        raise ValueError(
            f"max_new_tokens must be an integer between 1 and {MAX_NEW_TOKENS} "
            f"(6-mer tokens, i.e. up to {MAX_NEW_TOKENS * 6} bases); got {max_new_tokens!r}."
        )
    if not isinstance(temperature, (int, float)) or temperature < 0:
        raise ValueError(f"temperature must be >= 0 (0 = greedy); got {temperature!r}.")
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError(f"top_k must be an integer >= 0 (0 = no top-k filter); got {top_k!r}.")
    if sampling not in SAMPLING_MODES:
        raise ValueError(f"sampling must be one of {list(SAMPLING_MODES)}; got {sampling!r}.")


def _next_token(
    lm: Loaded,
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    sampling: str,
    generator: torch.Generator | None,
) -> int:
    """Pick the next DNA 6-mer token from fp32 full-vocabulary logits."""
    kmer_logits = logits[lm.kmer_ids]  # special tokens are never generated
    greedy = temperature == 0
    if not greedy:
        kmer_logits = kmer_logits / temperature
    if top_k and top_k < kmer_logits.numel():
        threshold = torch.topk(kmer_logits, top_k).values[-1]
        kmer_logits = kmer_logits.masked_fill(kmer_logits < threshold, float("-inf"))
    probs = torch.softmax(kmer_logits, dim=-1)
    if sampling == "token":
        j = probs.argmax() if greedy else torch.multinomial(probs, 1, generator=generator)[0]
        return int(lm.kmer_ids[j])
    # "base": marginalize the 6-mer distribution to one distribution over A/C/G/T per
    # position and choose each base from its marginal (GenerTeam's generate()).
    k = lm.vocab.k
    base_probs = torch.zeros(k, 4, device=probs.device)
    base_probs.scatter_add_(1, lm.kmer_base, probs.expand(k, -1))
    if greedy:
        bases = base_probs.argmax(dim=-1)
    else:
        bases = torch.multinomial(base_probs, 1, generator=generator).squeeze(-1)
    return int(lm.flat_to_id[(bases * lm.powers).sum()])


def generate(
    m: ResolvedModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    sampling: str = "base",
    remainder: str = "reject",
    seed: int | None = None,
) -> dict:
    """Autoregressively extend `prompt` by `max_new_tokens` 6-mers (6 bases each)."""
    validate_generation_args(max_new_tokens, temperature, top_k, sampling)
    (seq,), (adjusted,) = prepare_sequences([prompt], remainder)
    lm = load(m)
    ids = [lm.vocab.bos_id, *lm.vocab.encode(seq)]
    if len(ids) + max_new_tokens > lm.max_tokens:
        raise ValueError(
            f"Prompt ({len(ids)} tokens incl. <s>) + max_new_tokens ({max_new_tokens}) exceeds "
            f"this checkpoint's {lm.max_tokens}-token context. Use a shorter prompt or fewer tokens."
        )
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(int(seed))
    new_ids: list[int] = []
    past = None
    step_input = torch.tensor([ids], device=DEVICE)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = lm.model.model(input_ids=step_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = lm.logits_fp32(out.last_hidden_state[0, -1])
            nxt = _next_token(lm, logits, float(temperature), top_k, sampling, generator)
            new_ids.append(nxt)
            step_input = torch.tensor([[nxt]], device=DEVICE)
    generated = lm.vocab.decode(new_ids)
    return {
        "prompt": seq,
        "generated_sequence": generated,
        "full_sequence": seq + generated,
        "num_new_tokens": len(new_ids),
        "num_new_bases": len(generated),
        "remainder": remainder,
        "remainder_bases": adjusted,
    }
