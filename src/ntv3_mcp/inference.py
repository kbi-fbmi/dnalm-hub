"""Model loading and inference helpers for NTv3.

NTv3 checkpoints are single-base (character-level) DNA language models shipped as
custom ``trust_remote_code`` architectures on the HuggingFace Hub, loadable through
``AutoTokenizer`` / ``AutoModelForMaskedLM``. See the model cards under
https://huggingface.co/InstaDeepAI for details, e.g.
https://huggingface.co/InstaDeepAI/NTv3_100M_pre

All heavy lifting (tokenization, batching, padding, masking) happens here so the
MCP tool functions in ``server.py`` stay thin and easy to reason about.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

VALID_NUCLEOTIDES = set("ACGTN")

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}

_lock = threading.Lock()
_cache: dict[str, tuple] = {}


def _select_device() -> str:
    override = os.environ.get("NTV3_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_dtype() -> torch.dtype:
    override = os.environ.get("NTV3_DTYPE", "").strip().lower()
    if override:
        if override not in _DTYPE_MAP:
            raise ValueError(f"Invalid NTV3_DTYPE={override!r}; choose one of {sorted(_DTYPE_MAP)}.")
        return _DTYPE_MAP[override]
    return torch.float32


DEVICE = _select_device()
DTYPE = _select_dtype()


def validate_sequence(sequence: str) -> str:
    """Normalize and validate a DNA sequence, raising ``ValueError`` on bad input."""
    if not isinstance(sequence, str):
        raise ValueError("Sequence must be a string.")
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("Sequence must not be empty.")
    bad_chars = sorted(set(seq) - VALID_NUCLEOTIDES)
    if bad_chars:
        raise ValueError(
            f"Sequence contains invalid character(s) {bad_chars}; only A, C, G, T, N are allowed."
        )
    return seq


def load(repo_id: str):
    """Load (and cache) the tokenizer + model for a given HuggingFace repo id."""
    with _lock:
        cached = _cache.get(repo_id)
        if cached is not None:
            return cached
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
            model = AutoModelForMaskedLM.from_pretrained(
                repo_id, trust_remote_code=True, torch_dtype=DTYPE
            )
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
            raise RuntimeError(
                f"Failed to load NTv3 model '{repo_id}' from HuggingFace: {exc}"
            ) from exc
        tokenizer.padding_side = "right"
        model.to(DEVICE)
        model.eval()
        _cache[repo_id] = (tokenizer, model)
        return tokenizer, model


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


def _tokenize(tokenizer, sequences: list[str], pad_multiple: int):
    batch = tokenizer(
        sequences,
        add_special_tokens=False,
        padding=True,
        pad_to_multiple_of=pad_multiple,
        return_tensors="pt",
    )
    return {k: v.to(DEVICE) for k, v in batch.items()}


def _probe_hidden_states(repo_id: str, pad_multiple: int):
    """Run a tiny dummy forward pass to inspect how many hidden-state layers a model exposes."""
    tokenizer, model = load(repo_id)
    dummy = "A" * pad_multiple
    batch = _tokenize(tokenizer, [dummy], pad_multiple)
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
    return tokenizer, model, out


def model_info(repo_id: str, pad_multiple: int) -> dict:
    tokenizer, model, out = _probe_hidden_states(repo_id, pad_multiple)
    n_params = sum(p.numel() for p in model.parameters())
    raw_cfg = model.config.to_dict()
    safe_cfg = {k: v for k, v in raw_cfg.items() if isinstance(v, (int, float, str, bool)) or v is None}
    return {
        "repo_id": repo_id,
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "num_parameters": n_params,
        "vocab_size": len(tokenizer),
        "hidden_size": int(out.hidden_states[-1].shape[-1]),
        "num_hidden_state_layers": len(out.hidden_states),
        "pad_multiple": pad_multiple,
        "mask_token": tokenizer.mask_token,
        "config": safe_cfg,
    }


def list_embedding_layers(repo_id: str, pad_multiple: int, which: str) -> dict:
    """Enumerate hidden-state layer indices usable as `layer_name` in embed_sequence.

    `which="recommended"` is a lightweight heuristic (the final layer, plus the one
    before it) rather than an InstaDeep-published recommendation -- NTv3's model
    cards do not (yet) specify a best layer per checkpoint the way some embedding
    guides do. `which="all"` returns every valid index, 0 (input embeddings) through
    the last transformer layer.
    """
    _, _, out = _probe_hidden_states(repo_id, pad_multiple)
    n_layers = len(out.hidden_states) - 1
    all_layers = list(range(0, n_layers + 1))
    if which == "all":
        layers = all_layers
        info = f"All {len(all_layers)} hidden-state layers (0=input embeddings, {n_layers}=final layer/'last')."
    elif which == "recommended":
        layers = sorted({n_layers, max(n_layers - 1, 0)})
        info = (
            "Heuristic only: the final layer and the one before it. NTv3 does not publish a "
            "recommended embedding layer per checkpoint; use which='all' to explore others."
        )
    else:
        raise ValueError("which must be 'recommended' or 'all'.")
    return {"layers": layers, "num_hidden_state_layers": len(out.hidden_states), "info": info}


def compute_embeddings(
    repo_id: str,
    sequences: list[str],
    layer: str | int | None,
    pooling: str,
    pad_multiple: int,
) -> tuple[np.ndarray, int, int]:
    if not sequences:
        raise ValueError("sequences must be a non-empty list of DNA sequences.")
    tokenizer, model = load(repo_id)
    seqs = [validate_sequence(s) for s in sequences]
    batch = _tokenize(tokenizer, seqs, pad_multiple)
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
    hidden_states = out.hidden_states
    n_layers = len(hidden_states) - 1
    idx = _parse_layer(layer, n_layers)
    h = hidden_states[idx]
    mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)

    if pooling == "mean":
        summed = (h * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = (summed / counts).float().cpu().numpy()
    elif pooling == "per_token":
        pooled = h.float().cpu().numpy()
    else:
        raise ValueError("pooling must be 'mean' or 'per_token'.")
    return pooled, idx, n_layers


def score_variant(
    repo_id: str,
    sequence: str,
    alt_allele: str,
    pad_multiple: int,
    position: int | None = None,
    ref_allele: str | None = None,
) -> dict:
    """Zero-shot single-nucleotide substitution scoring via one masked-LM forward pass.

    If `position` is omitted it defaults to the sequence center (matching evo2-mcp's
    `score_snp`, which always scores the center position). If `ref_allele` is
    omitted it is read directly from `sequence[position]`; if given, it must match.
    """
    tokenizer, model = load(repo_id)
    seq = validate_sequence(sequence)
    if position is None:
        position = len(seq) // 2
    if not (0 <= position < len(seq)):
        raise ValueError(f"position must be within [0, {len(seq) - 1}] for a sequence of length {len(seq)}.")

    ref = (ref_allele.strip().upper() if ref_allele else seq[position])
    alt = alt_allele.strip().upper()
    if len(ref) != 1 or ref not in VALID_NUCLEOTIDES:
        raise ValueError(f"ref_allele must be a single character in {sorted(VALID_NUCLEOTIDES)}; got {ref_allele!r}.")
    if len(alt) != 1 or alt not in VALID_NUCLEOTIDES:
        raise ValueError(f"alt_allele must be a single character in {sorted(VALID_NUCLEOTIDES)}; got {alt_allele!r}.")
    if seq[position] != ref:
        raise ValueError(
            f"Reference allele mismatch: sequence has '{seq[position]}' at position {position}, expected '{ref}'."
        )
    if tokenizer.mask_token_id is None:
        raise RuntimeError(f"Tokenizer for '{repo_id}' has no mask token; variant scoring is unavailable.")

    batch = _tokenize(tokenizer, [seq], pad_multiple)
    input_ids = batch["input_ids"].clone()
    input_ids[0, position] = tokenizer.mask_token_id
    batch["input_ids"] = input_ids

    with torch.no_grad():
        out = model(**batch)
    logits = out.logits[0, position].float()
    log_probs = torch.log_softmax(logits, dim=-1)

    ref_id = tokenizer.convert_tokens_to_ids(ref)
    alt_id = tokenizer.convert_tokens_to_ids(alt)
    if ref_id is None or ref_id == tokenizer.unk_token_id:
        raise ValueError(f"ref_allele '{ref}' is not a recognized token for this tokenizer.")
    if alt_id is None or alt_id == tokenizer.unk_token_id:
        raise ValueError(f"alt_allele '{alt}' is not a recognized token for this tokenizer.")

    ref_log_prob = log_probs[ref_id].item()
    alt_log_prob = log_probs[alt_id].item()
    mutated_seq = seq[:position] + alt + seq[position + 1 :]
    return {
        "sequence": seq,
        "mutated_sequence": mutated_seq,
        "position": position,
        "ref_allele": ref,
        "alt_allele": alt,
        "ref_log_prob": ref_log_prob,
        "alt_log_prob": alt_log_prob,
        "log_likelihood_ratio": alt_log_prob - ref_log_prob,
    }


def predict_masked(
    repo_id: str,
    sequence: str,
    positions: list[int],
    top_k: int,
    pad_multiple: int,
) -> list[dict]:
    tokenizer, model = load(repo_id)
    seq = validate_sequence(sequence)

    if not positions:
        positions = [i for i, c in enumerate(seq) if c == "N"]
        if not positions:
            raise ValueError(
                "No 'positions' given and the sequence contains no 'N' characters to auto-mask."
            )
    for p in positions:
        if not (0 <= p < len(seq)):
            raise ValueError(f"position {p} is out of range [0, {len(seq) - 1}] for this sequence.")
    if tokenizer.mask_token_id is None:
        raise RuntimeError(f"Tokenizer for '{repo_id}' has no mask token; masked prediction is unavailable.")
    top_k = max(1, min(top_k, len(VALID_NUCLEOTIDES)))

    single = _tokenize(tokenizer, [seq], pad_multiple)
    base_ids = single["input_ids"][0]
    base_attn = single["attention_mask"][0]

    batch_ids = base_ids.unsqueeze(0).repeat(len(positions), 1).clone()
    for row, p in enumerate(positions):
        batch_ids[row, p] = tokenizer.mask_token_id
    batch_attn = base_attn.unsqueeze(0).repeat(len(positions), 1)

    with torch.no_grad():
        out = model(input_ids=batch_ids, attention_mask=batch_attn)

    nt_ids = {nt: tokenizer.convert_tokens_to_ids(nt) for nt in sorted(VALID_NUCLEOTIDES)}
    nt_ids = {nt: i for nt, i in nt_ids.items() if i is not None and i != tokenizer.unk_token_id}

    results = []
    for row, p in enumerate(positions):
        probs = torch.softmax(out.logits[row, p].float(), dim=-1)
        ranked = sorted(((nt, probs[i].item()) for nt, i in nt_ids.items()), key=lambda kv: kv[1], reverse=True)
        results.append(
            {
                "position": p,
                "original": seq[p],
                "predictions": [{"nucleotide": nt, "probability": prob} for nt, prob in ranked[:top_k]],
            }
        )
    return results


def compare_sequences(
    repo_id: str,
    sequence_a: str,
    sequence_b: str,
    layer: str | int | None,
    pad_multiple: int,
) -> tuple[float, int]:
    embeddings, layer_idx, _ = compute_embeddings(repo_id, [sequence_a, sequence_b], layer, "mean", pad_multiple)
    a, b = embeddings[0], embeddings[1]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    similarity = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    return similarity, layer_idx
