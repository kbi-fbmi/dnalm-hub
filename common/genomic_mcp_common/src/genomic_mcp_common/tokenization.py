"""Guard against single-position masked scoring on a non-single-nucleotide tokenizer.

NTv3 and HyenaDNA tokenize one base per token, so a raw sequence index can be
used directly as a token index for masking/scoring a single position. NTv2
(6-mer) and DNABERT-2 (BPE) do not have that property -- masking "one base"
there either shifts/overlaps neighboring bases or isn't a well-defined
operation. Backends for those tokenizers use a whole-sequence scoring method
instead (see each service's `scoring.py`); this guard exists so a checkpoint
with the wrong tokenizer shape fails loudly instead of returning a
plausible-looking but meaningless number.
"""

from __future__ import annotations

DEFAULT_SINGLE_NUCLEOTIDES = ("A", "C", "G", "T", "N")


def is_single_nucleotide_tokenizer(tokenizer, nucleotides: tuple[str, ...] = DEFAULT_SINGLE_NUCLEOTIDES) -> bool:
    """True if every base in `nucleotides` maps to its own distinct, known token id."""
    try:
        ids = [tokenizer.convert_tokens_to_ids(nt) for nt in nucleotides]
    except Exception:  # noqa: BLE001 - any tokenizer failure means "not this shape"
        return False
    if any(i is None for i in ids):
        return False
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and any(i == unk for i in ids):
        return False
    return len(set(ids)) == len(ids)


def assert_single_nucleotide_tokenizer(tokenizer, repo_id: str) -> None:
    """Raise a clear `RuntimeError` if `tokenizer` is not single-nucleotide-per-token."""
    if not is_single_nucleotide_tokenizer(tokenizer):
        raise RuntimeError(
            f"'{repo_id}' does not use a single-nucleotide tokenizer (1 token = 1 base), so "
            "single-position masked scoring/prediction is not valid for it -- the raw sequence "
            "offset would not correspond to a single token. Use this backend's whole-sequence "
            "scoring method instead."
        )
