"""Tokenizer-agnostic zero-shot SNP scoring: whole-sequence log-likelihood delta.

NTv3 scores a SNP by masking the single affected position, which only works
because its tokenizer is 1 base = 1 token. NTv2 (6-mer), DNABERT-2 (BPE), and
the causal backends instead tokenize the reference and mutated sequences
*independently*, score each whole sequence, and report the difference -- no
shared base -> token position mapping is needed, so the tokenizer shape does
not matter.

"Whole-sequence log-likelihood" depends on the model type:

- masked LMs (NTv2, DNABERT-2): pseudo-log-likelihood (PLL) -- mask each
  scored token in turn and sum the log-probability the model assigns to the
  true token there (`masked_lm_pseudo_log_likelihood` below). Costs one forward
  pass per token, batched.
- causal LMs (HyenaDNA, Evo2t): exact autoregressive log-likelihood (see
  `causal.py`).

Either way the service passes a `str -> float` scorer into `score_snp_whole_sequence`,
which owns allele validation and the response shape.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .validation import DEFAULT_VALID_NUCLEOTIDES, validate_sequence

WHOLE_SEQUENCE_METHOD = "whole_sequence_log_prob_delta"


def apply_snp(
    sequence: str,
    alt_allele: str,
    position: int | None = None,
    valid_nucleotides: set[str] = DEFAULT_VALID_NUCLEOTIDES,
) -> tuple[str, str, int, str, str]:
    """Validate a SNP request and return `(ref_seq, mutated_seq, position, ref, alt)`.

    `position` defaults to the sequence center (same default as NTv3/evo2-mcp).
    The reference allele is read from the sequence itself.
    """
    seq = validate_sequence(sequence, valid_nucleotides)
    if position is None:
        position = len(seq) // 2
    if not (0 <= position < len(seq)):
        raise ValueError(
            f"position must be within [0, {len(seq) - 1}] for a sequence of length {len(seq)}."
        )
    alt = alt_allele.strip().upper() if isinstance(alt_allele, str) else ""
    if len(alt) != 1 or alt not in valid_nucleotides:
        raise ValueError(
            f"alt_allele must be a single character in {sorted(valid_nucleotides)}; got {alt_allele!r}."
        )
    ref = seq[position]
    mutated = seq[:position] + alt + seq[position + 1 :]
    return seq, mutated, position, ref, alt


def score_snp_whole_sequence(
    sequence: str,
    alt_allele: str,
    log_likelihood: Callable[[str], float],
    position: int | None = None,
) -> dict:
    """Score a SNP as `log_likelihood(mutated) - log_likelihood(reference)`.

    Returns the same keys NTv3's `score_snp` tool returns (minus `checkpoint`,
    which the service adds), with `method` set to `WHOLE_SEQUENCE_METHOD`.
    """
    seq, mutated, position, ref, alt = apply_snp(sequence, alt_allele, position)
    original_score = float(log_likelihood(seq))
    mutated_score = original_score if mutated == seq else float(log_likelihood(mutated))
    return {
        "original_sequence": seq,
        "mutated_sequence": mutated,
        "center_position": position,
        "reference_allele": ref,
        "alternative_allele": alt,
        "original_score": original_score,
        "mutated_score": mutated_score,
        "score_delta": mutated_score - original_score,
        "method": WHOLE_SEQUENCE_METHOD,
    }


def masked_lm_pseudo_log_likelihood(
    input_ids: torch.Tensor,
    scored_positions: list[int],
    mask_token_id: int,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int = 16,
) -> float:
    """Sum of log P(true token | rest) over `scored_positions`, masking one position at a time.

    Args:
        input_ids: 1D token ids for a single (unpadded) sequence, special tokens included.
        scored_positions: token indices to score -- normally every non-special token.
        mask_token_id: the tokenizer's mask token id.
        logits_fn: runs the model on a `(batch, seq_len)` id tensor and returns
            `(batch, seq_len, vocab)` logits. Keeps this module model-agnostic
            (e.g. NTv2 needs an extra `encoder_attention_mask` kwarg).
        batch_size: masked copies per forward pass.
    """
    if input_ids.dim() != 1:
        raise ValueError("input_ids must be a 1D tensor for a single sequence.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    total = 0.0
    for start in range(0, len(scored_positions), batch_size):
        chunk = scored_positions[start : start + batch_size]
        batch = input_ids.unsqueeze(0).repeat(len(chunk), 1)
        rows = torch.arange(len(chunk), device=input_ids.device)
        cols = torch.tensor(chunk, device=input_ids.device)
        batch[rows, cols] = mask_token_id
        with torch.no_grad():
            logits = logits_fn(batch)
        log_probs = torch.log_softmax(logits[rows, cols].float(), dim=-1)
        total += log_probs[rows, input_ids[cols]].sum().item()
    return total
