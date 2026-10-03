"""Pure-logic tests that don't require downloading any model weights."""

import pytest
from ntv3_mcp.inference import validate_sequence


def test_validate_sequence_uppercases_and_strips():
    assert validate_sequence("  acgtn  ") == "ACGTN"


def test_validate_sequence_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_sequence("   ")


def test_validate_sequence_rejects_invalid_characters():
    with pytest.raises(ValueError, match="invalid character"):
        validate_sequence("ACGTX")


def test_validate_sequence_rejects_non_string():
    with pytest.raises(ValueError, match="must be a string"):
        validate_sequence(123)  # type: ignore[arg-type]
