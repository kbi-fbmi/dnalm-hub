"""Pure-logic tests for the context-length guard and layer parsing (no model weights)."""

from types import SimpleNamespace

import pytest
from genalm_mcp.inference import _parse_layer, check_token_lengths, max_tokens
from genalm_mcp.registry import ResolvedModel, resolve_model


def _model(**config):
    return SimpleNamespace(config=SimpleNamespace(**config))


def test_lengths_at_limit_pass():
    check_token_lengths([1, 512], 512)


def test_over_limit_raises_clear_error_naming_the_sequence():
    with pytest.raises(ValueError, match=r"Sequence 1 tokenizes to 513 tokens.*never truncated"):
        check_token_lengths([10, 513], 512)


def test_registry_limit_wins_over_config():
    m = resolve_model("bigbird-base-t2t")
    assert max_tokens(m, _model(max_position_embeddings=512)) == 4096


def test_unregistered_repo_uses_config_limit():
    m = ResolvedModel("u/x", None, None)
    assert max_tokens(m, _model(max_position_embeddings=1024)) == 1024


def test_unregistered_repo_without_config_limit_falls_back():
    assert max_tokens(ResolvedModel("u/x", None, None), _model()) == 512


@pytest.mark.parametrize(
    ("layer", "expected"), [("last", 12), (None, 12), ("0", 0), (6, 6), ("-1", 12), (-13, 0)]
)
def test_parse_layer(layer, expected):
    assert _parse_layer(layer, 12) == expected


@pytest.mark.parametrize("layer", ["13", -14, "first"])
def test_parse_layer_rejects_bad_values(layer):
    with pytest.raises(ValueError, match="layer must be"):
        _parse_layer(layer, 12)
