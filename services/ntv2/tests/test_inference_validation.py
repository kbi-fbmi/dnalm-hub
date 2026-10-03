"""Pure-logic tests for the context-length guard (no model weights)."""

import pytest

from ntv2_mcp.inference import check_token_lengths, max_tokens
from ntv2_mcp.registry import ResolvedModel, resolve_model


class _Tok:
    def __init__(self, model_max_length):
        self.model_max_length = model_max_length


def test_lengths_at_limit_pass():
    check_token_lengths([1, 2048], 2048)


def test_over_limit_raises_clear_error_naming_the_sequence():
    with pytest.raises(ValueError, match=r"Sequence 1 tokenizes to 2049 tokens.*never truncated"):
        check_token_lengths([10, 2049], 2048)


def test_registry_limit_wins_over_tokenizer():
    assert max_tokens(resolve_model("2.5b-v1"), _Tok(2048)) == 1000


def test_unregistered_repo_uses_tokenizer_limit():
    assert max_tokens(ResolvedModel("u/x", None, None), _Tok(512)) == 512


def test_unregistered_repo_with_sentinel_limit_falls_back():
    assert max_tokens(ResolvedModel("u/x", None, None), _Tok(int(1e30))) == 2048
