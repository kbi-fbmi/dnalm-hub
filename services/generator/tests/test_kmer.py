"""The local 6-mer tokenizer must reproduce GenerTeam's DNAKmerTokenizer ids."""

import pytest
from generator_mcp.kmer import KmerVocab, apply_remainder_policy

from .conftest import vocab_text


def test_vocab_layout(vocab):
    assert (vocab.bos_id, vocab.pad_id, vocab.oov_id) == (1, 3, 0)
    assert len(vocab.kmers) == 4096
    # Remote tokenizer order: specials, then itertools.product("ATCG", repeat=6).
    assert vocab.token_to_id["AAAAAA"] == 32
    assert vocab.token_to_id["AAAAAT"] == 33
    assert vocab.token_to_id["GGGGGG"] == 4127


def test_encode_matches_remote_tokenizer_rules(vocab):
    assert vocab.encode("AAAAAAGGGGGG") == [32, 4127]
    # A 6-mer containing N is not in the vocabulary -> <oov>, as in DNAKmerTokenizer.
    assert vocab.encode("AAAAAN") == [vocab.oov_id]
    with pytest.raises(ValueError, match="multiple of 6"):
        vocab.encode("AAAAA")


def test_decode_round_trip(vocab):
    seq = "ACGTACGGTTCA"
    assert vocab.decode(vocab.encode(seq)) == seq
    with pytest.raises(ValueError):
        vocab.decode([vocab.bos_id])


def test_matching_kmer_ids(vocab):
    assert vocab.matching_kmer_ids("ACGTAC") == [vocab.token_to_id["ACGTAC"]]
    assert len(vocab.matching_kmer_ids("ACG")) == 4**3  # partial 6-mer: any completion
    assert len(vocab.matching_kmer_ids("NNNNNN")) == 4096
    ids = vocab.matching_kmer_ids("ANGTAC")
    assert sorted(ids) == sorted(vocab.token_to_id[f"A{b}GTAC"] for b in "ACGT")


def test_base_tables_are_consistent(vocab):
    kmer_base, flat_to_id = vocab.base_tables()
    j = vocab.kmers.index("CATGGA")
    assert [kmer_base[p][j] for p in range(6)] == ["ACGT".index(b) for b in "CATGGA"]
    flat = 0
    for b in "CATGGA":
        flat = flat * 4 + "ACGT".index(b)
    assert flat_to_id[flat] == vocab.token_to_id["CATGGA"]
    assert sorted(flat_to_id) == sorted(vocab.kmer_ids)


def test_bad_vocab_rejected():
    with pytest.raises(ValueError, match="DNA 6-mers"):
        KmerVocab.from_vocab_text(vocab_text(5), 6)


@pytest.mark.parametrize(
    ("seq", "policy", "expected"),
    [
        ("ACGTAC", "reject", ("ACGTAC", 0)),
        ("ACGTACGT", "trim_left", ("GTACGT", 2)),
        ("ACGTACGT", "pad_left", ("AAAAACGTACGT", 4)),
        ("ACG", "pad_left", ("AAAACG", 3)),
    ],
)
def test_remainder_policies(seq, policy, expected):
    assert apply_remainder_policy(seq, policy, 6) == expected


def test_reject_names_the_remainder_and_the_options():
    with pytest.raises(ValueError, match=r"8 bp \(8 % 6 = 2\).*trim_left.*pad_left"):
        apply_remainder_policy("ACGTACGT", "reject", 6)


def test_trim_left_of_too_short_sequence_and_unknown_policy():
    with pytest.raises(ValueError, match="shorter than one 6-mer"):
        apply_remainder_policy("ACG", "trim_left", 6)
    with pytest.raises(ValueError, match="remainder must be one of"):
        apply_remainder_policy("ACGTAC", "truncate", 6)
