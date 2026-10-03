import pytest
from dnalm_common.validation import validate_sequence


def test_normalizes_case_and_whitespace():
    assert validate_sequence("  acgtn  ") == "ACGTN"


def test_rejects_empty_sequence():
    with pytest.raises(ValueError, match="empty"):
        validate_sequence("   ")


def test_rejects_non_string():
    with pytest.raises(ValueError, match="string"):
        validate_sequence(123)  # type: ignore[arg-type]


def test_rejects_invalid_characters():
    with pytest.raises(ValueError, match="invalid character"):
        validate_sequence("ACGTX")


def test_custom_alphabet_allows_amino_acids():
    assert validate_sequence("mkv", valid_nucleotides=set("MKV")) == "MKV"
