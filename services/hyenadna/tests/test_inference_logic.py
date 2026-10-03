"""Pure-logic tests for inference helpers (no model weights)."""

import math

import pytest
import torch
from hyenadna_mcp import inference
from hyenadna_mcp.inference import (
    _length_groups,
    _parse_layer,
    causal_log_likelihood,
    check_lengths,
    max_tokens,
    sample_next,
)
from hyenadna_mcp.registry import ResolvedModel, resolve_model


class _Cfg:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len


class _Model:
    def __init__(self, max_seq_len):
        self.config = _Cfg(max_seq_len)


def test_lengths_at_limit_pass():
    check_lengths([1, 1026], 1026)


def test_over_limit_raises_clear_error_naming_the_sequence():
    with pytest.raises(ValueError, match=r"Sequence 1 is 1027 bases.*never truncated"):
        check_lengths([10, 1027], 1026)


def test_server_cap_lowers_limit(monkeypatch):
    monkeypatch.setattr(inference, "MAX_BASES", 100)
    with pytest.raises(ValueError, match="HYENADNA_MAX_BASES=100"):
        check_lengths([101], 1026)
    check_lengths([100], 1026)


def test_registry_limit_wins_over_config():
    assert max_tokens(resolve_model("tiny-1k"), _Model(5)) == 1026


def test_unregistered_repo_uses_config_limit():
    assert max_tokens(ResolvedModel("u/x", None, None), _Model(32770)) == 32770


def test_length_groups_only_batch_equal_lengths_and_respect_budget():
    groups = _length_groups([5, 7, 5, 5, 7], max_batch_bases=10)
    assert sorted(map(sorted, groups)) == [[0, 2], [1], [3], [4]]
    # A single sequence longer than the budget still gets its own group.
    assert _length_groups([50], max_batch_bases=10) == [[0]]


def test_parse_layer():
    assert _parse_layer("last", 10) == 9
    assert _parse_layer(None, 10) == 9
    assert _parse_layer("0", 10) == 0
    assert _parse_layer(-2, 10) == 8
    with pytest.raises(ValueError, match="between 0 and 9"):
        _parse_layer(10, 10)
    with pytest.raises(ValueError, match="integer"):
        _parse_layer("first", 10)


def test_causal_log_likelihood_skips_first_token_and_shifts():
    # Uniform logits over 4 tokens: each scored position contributes log(1/4).
    logits = torch.zeros(5, 4)
    ids = torch.tensor([0, 1, 2, 3, 0])
    assert causal_log_likelihood(logits, ids) == pytest.approx(4 * math.log(0.25))
    # Position t's logits predict token t+1.
    logits = torch.full((3, 4), -1e9)
    logits[0, 2] = 0.0
    logits[1, 3] = 0.0
    assert causal_log_likelihood(logits, torch.tensor([1, 2, 3])) == pytest.approx(0.0)
    assert causal_log_likelihood(torch.zeros(1, 4), torch.tensor([1])) == 0.0


def test_causal_log_likelihood_sums_in_float64():
    # 1M positions of log(1/4): the total is ~-1.39e6, where a float32 result is only
    # resolved to 0.125 -- coarser than a typical SNP delta. Per-token values are
    # float32 (log_softmax), so compare against their exact float64 sum.
    n = 1_000_001
    per_token = float(torch.log_softmax(torch.zeros(4), dim=-1)[0])
    total = causal_log_likelihood(torch.zeros(n, 4), torch.zeros(n, dtype=torch.long))
    assert total == pytest.approx((n - 1) * per_token, abs=1e-6)


def test_sample_next_greedy_and_top_k():
    logits = torch.tensor([9.0, 0.0, 1.0, 3.0, 2.0])
    allowed = [1, 2, 3, 4]  # token 0 is excluded even though it is the global max
    assert sample_next(logits, allowed, temperature=0, top_k=0) == 3
    gen = torch.Generator().manual_seed(0)
    picks = {sample_next(logits, allowed, 1.0, top_k=1, generator=gen) for _ in range(20)}
    assert picks == {3}
    picks = {sample_next(logits, allowed, 1.0, top_k=2, generator=gen) for _ in range(200)}
    assert picks <= {3, 4}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"temperature": -1.0}, "temperature"),
        ({"top_k": -1}, "top_k"),
    ],
)
def test_generate_rejects_bad_arguments_before_loading(kwargs, match):
    args = {"max_new_tokens": 5, "temperature": 1.0, "top_k": 4, **kwargs}
    with pytest.raises(ValueError, match=match):
        inference.generate(resolve_model("tiny-1k"), "ACGT", **args)
