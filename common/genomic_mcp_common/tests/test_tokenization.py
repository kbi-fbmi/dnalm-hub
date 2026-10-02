"""No-download tests for the single-nucleotide-tokenizer guard."""

import pytest

from genomic_mcp_common.tokenization import assert_single_nucleotide_tokenizer, is_single_nucleotide_tokenizer


class _FakeTokenizer:
    """Minimal stand-in exposing just what the guard needs, no real model/download."""

    def __init__(self, vocab: dict[str, int], unk_token_id: int | None = None) -> None:
        self.vocab = vocab
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token: str):
        return self.vocab.get(token, self.unk_token_id)


def test_single_character_vocab_is_single_nucleotide():
    tok = _FakeTokenizer({"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "[MASK]": 5}, unk_token_id=6)
    assert is_single_nucleotide_tokenizer(tok) is True


def test_six_mer_vocab_is_not_single_nucleotide():
    # A 6-mer tokenizer has no entries for the bare single-character bases at all.
    tok = _FakeTokenizer({"ACGTAC": 10, "GGTTAA": 11}, unk_token_id=0)
    assert is_single_nucleotide_tokenizer(tok) is False


def test_missing_token_is_not_single_nucleotide():
    tok = _FakeTokenizer({"A": 0, "C": 1, "G": 2, "T": 3}, unk_token_id=None)  # no "N"
    assert is_single_nucleotide_tokenizer(tok) is False


def test_bases_mapping_to_unk_is_not_single_nucleotide():
    tok = _FakeTokenizer({"A": 0, "C": 1, "G": 2, "T": 3, "N": 99}, unk_token_id=99)
    assert is_single_nucleotide_tokenizer(tok) is False


def test_assert_raises_clear_error_for_non_single_nucleotide_tokenizer():
    tok = _FakeTokenizer({"ACGTAC": 10}, unk_token_id=0)
    with pytest.raises(RuntimeError, match="single-nucleotide tokenizer"):
        assert_single_nucleotide_tokenizer(tok, "some/repo-id")


def test_assert_passes_silently_for_single_nucleotide_tokenizer():
    tok = _FakeTokenizer({"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}, unk_token_id=5)
    assert_single_nucleotide_tokenizer(tok, "some/repo-id")  # must not raise
