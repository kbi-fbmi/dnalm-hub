"""DNA sequence validation shared by every backend."""

from __future__ import annotations

DEFAULT_VALID_NUCLEOTIDES = set("ACGTN")


def validate_sequence(
    sequence: str, valid_nucleotides: set[str] = DEFAULT_VALID_NUCLEOTIDES
) -> str:
    """Normalize and validate a DNA sequence, raising `ValueError` on bad input."""
    if not isinstance(sequence, str):
        # ValueError, not TypeError: only ValueError reaches MCP clients (user_errors_as_tool_errors).
        raise ValueError("Sequence must be a string.")  # noqa: TRY004
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("Sequence must not be empty.")
    bad_chars = sorted(set(seq) - valid_nucleotides)
    if bad_chars:
        raise ValueError(
            f"Sequence contains invalid character(s) {bad_chars}; only {sorted(valid_nucleotides)} are allowed."
        )
    return seq
