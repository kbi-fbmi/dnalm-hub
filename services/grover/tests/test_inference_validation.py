"""Pure-logic tests for the context-length guard and layer parsing (no model weights)."""

import pytest
from grover_mcp.inference import _parse_layer, check_token_lengths, max_tokens
from grover_mcp.registry import ResolvedModel, resolve_model


class _Tok:
    def __init__(self, model_max_length):
        self.model_max_length = model_max_length


def test_lengths_at_limit_pass():
    check_token_lengths([1, 512], 512)


def test_over_limit_raises_clear_error_naming_the_sequence():
    with pytest.raises(ValueError, match=r"Sequence 1 tokenizes to 513 tokens.*never truncated"):
        check_token_lengths([10, 513], 512)


def test_registry_limit_wins_over_tokenizer():
    assert max_tokens(resolve_model("grover"), _Tok(4096)) == 512


def test_unregistered_repo_uses_tokenizer_limit():
    assert max_tokens(ResolvedModel("u/x", None, None), _Tok(1024)) == 1024


def test_unregistered_repo_with_sentinel_limit_falls_back():
    assert max_tokens(ResolvedModel("u/x", None, None), _Tok(int(1e30))) == 512


@pytest.mark.parametrize(
    ("layer", "expected"), [("last", 12), (None, 12), ("6", 6), (0, 0), (-1, 12), ("-2", 11)]
)
def test_parse_layer_accepts_ints_strings_and_last(layer, expected):
    assert _parse_layer(layer, 12) == expected


@pytest.mark.parametrize("layer", ["13", "-14", "first", 1.5j])
def test_parse_layer_rejects_out_of_range_or_garbage(layer):
    with pytest.raises(ValueError, match="layer must be"):
        _parse_layer(layer, 12)
