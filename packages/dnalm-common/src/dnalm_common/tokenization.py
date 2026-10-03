"""Guard against single-position masked scoring on a non-single-nucleotide tokenizer.

NTv3 and HyenaDNA tokenize one base per token, so a raw sequence index can be
used directly as a token index for masking/scoring a single position. NTv2
(6-mer) and DNABERT-2 (BPE) do not have that property -- masking "one base"
there either shifts/overlaps neighboring bases or isn't a well-defined
operation. Backends for those tokenizers use a whole-sequence scoring method
instead (see `dnalm_common.scoring`); this guard exists so a checkpoint
with the wrong tokenizer shape fails loudly instead of returning a
plausible-looking but meaningless number.
"""

from __future__ import annotations

DEFAULT_SINGLE_NUCLEOTIDES = ("A", "C", "G", "T", "N")

# Long enough to contain a full 6-mer (NTv2) and typical BPE merges (DNABERT-2).
_PROBE_SEQUENCE = "ACGTACGTACGTAC"


def is_single_nucleotide_tokenizer(tokenizer, nucleotides: tuple[str, ...] = DEFAULT_SINGLE_NUCLEOTIDES) -> bool:
    """True if every base maps to its own distinct token id AND sequences tokenize 1 base -> 1 token.

    Checking the vocab alone is not enough: NTv2's 6-mer vocab also contains the
    bare bases A/C/G/T/N (used for leftover bases that don't fill a 6-mer), yet
    "ACGTAC" still tokenizes to a single token. So when the tokenizer exposes
    `tokenize`, a probe sequence must also come back as exactly one token per base.
    """
    try:
        ids = [tokenizer.convert_tokens_to_ids(nt) for nt in nucleotides]
    except Exception:  # noqa: BLE001 - any tokenizer failure means "not this shape"
        return False
    if any(i is None for i in ids):
        return False
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and any(i == unk for i in ids):
        return False
    if len(set(ids)) != len(ids):
        return False
    if hasattr(tokenizer, "tokenize"):
        try:
            tokens = tokenizer.tokenize(_PROBE_SEQUENCE)
        except Exception:  # noqa: BLE001
            return False
        if list(tokens) != list(_PROBE_SEQUENCE):
            return False
    return True


def assert_single_nucleotide_tokenizer(tokenizer, repo_id: str) -> None:
    """Raise a clear `RuntimeError` if `tokenizer` is not single-nucleotide-per-token."""
    if not is_single_nucleotide_tokenizer(tokenizer):
        raise RuntimeError(
            f"'{repo_id}' does not use a single-nucleotide tokenizer (1 token = 1 base), so "
            "single-position masked scoring/prediction is not valid for it -- the raw sequence "
            "offset would not correspond to a single token. Use this backend's whole-sequence "
            "scoring method instead."
        )
