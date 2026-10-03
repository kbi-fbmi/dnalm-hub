"""Pure-logic tests for evo2_mcp.inference, using a tiny stand-in model (no weights)."""

import math
from types import SimpleNamespace

import pytest
import torch
from evo2_mcp import inference
from evo2_mcp.registry import ResolvedModel, resolve_model
from torch import nn


class _Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Linear(hidden, hidden)

    def forward(self, x):
        return x + self.mlp(x)


class _Inner(nn.Module):
    """Same module names as the port's Evo2Model: embed_tokens, layers[i].mlp, final_norm."""

    def __init__(self, n_blocks=4, hidden=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(512, hidden)
        self.layers = nn.ModuleList(_Block(hidden) for _ in range(n_blocks))
        self.final_norm = nn.LayerNorm(hidden)
        self.calls = []

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            self.calls.append(i)
            x = layer(x)
        return self.final_norm(x)


class _Fake(nn.Module):
    def __init__(self, n_blocks=4, hidden=8):
        super().__init__()
        self.model = _Inner(n_blocks, hidden)
        self.config = SimpleNamespace(max_position_embeddings=8192, hidden_size=hidden)


UNREGISTERED = ResolvedModel("u/x", None, None)


def test_parse_layer_variants():
    fake = _Fake(n_blocks=4)
    m = resolve_model("7b")
    assert inference.parse_layer(m, fake, "last") == 4
    assert inference.parse_layer(m, fake, None) == 4
    assert inference.parse_layer(m, fake, "2") == 2
    assert inference.parse_layer(m, fake, -1) == 4
    assert inference.parse_layer(m, fake, "blocks.3.mlp.l3") == "blocks.3.mlp.l3"


def test_parse_layer_recommended_uses_registry_then_depth_heuristic():
    big = _Fake(n_blocks=32)
    assert inference.parse_layer(resolve_model("7b"), big, "recommended") == "blocks.28.mlp.l3"
    assert inference.parse_layer(UNREGISTERED, big, "recommended") == "blocks.27.mlp.l3"


@pytest.mark.parametrize("bad", ["5", "-6", "blocks.4.mlp.l3", "blocks.x", "first"])
def test_parse_layer_rejects_invalid(bad):
    with pytest.raises(ValueError):
        inference.parse_layer(UNREGISTERED, _Fake(n_blocks=4), bad)


def test_capture_matches_full_forward_and_stops_early():
    torch.manual_seed(0)
    fake = _Fake(n_blocks=4).eval()
    ids = torch.tensor([[65, 67, 71, 84, 65]])
    with torch.no_grad():
        full = fake.model(ids)[0]
        fake.model.calls.clear()
        h2 = inference._capture(fake, ids, 2)  # output of block 1
        assert fake.model.calls == [0, 1]  # blocks 2 and 3 were skipped
        mlp3 = inference._capture(fake, ids, "blocks.3.mlp.l3")
        last = inference._capture(fake, ids, 4)
        emb = inference._capture(fake, ids, 0)
        x = fake.model.layers[1](fake.model.layers[0](fake.model.embed_tokens(ids)))
        x3 = fake.model.layers[2](x)
        assert torch.allclose(h2, x[0])
        assert torch.allclose(mlp3, fake.model.layers[3].mlp(x3)[0])
        assert torch.allclose(last, full)
        assert torch.allclose(emb, fake.model.embed_tokens(ids)[0])


def test_token_log_probs_skips_first_base():
    logits = torch.zeros(5, 512)
    ids = torch.tensor([65, 67, 71, 84, 65])
    lp = inference.token_log_probs(logits, ids)
    assert lp.shape == (4,)
    assert torch.allclose(lp, torch.full((4,), -math.log(512)))


def test_sample_next_only_returns_acgt_and_greedy_is_argmax():
    logits = torch.full((512,), -5.0)
    logits[0] = 100.0  # eos would win without the A/C/G/T restriction
    logits[ord("G")] = 3.0
    assert inference.sample_next(logits, 0.0, 4, None) == ord("G")
    gen = torch.Generator().manual_seed(1)
    picks = {inference.sample_next(logits, 1.0, 4, gen) for _ in range(50)}
    assert picks <= {ord(c) for c in "ACGT"}
    assert {inference.sample_next(logits, 1.0, 1, gen) for _ in range(20)} == {ord("G")}


@pytest.mark.parametrize(
    "args",
    [(0, 1.0, 4), (inference.MAX_NEW_TOKENS + 1, 1.0, 4), (5, -1.0, 4), (5, 1.0, 5), (5, 1.0, -1)],
)
def test_generation_args_validated(args):
    with pytest.raises(ValueError):
        inference.validate_generation_args(*args)


def test_length_limit_and_cap():
    assert inference.max_length(resolve_model("7b")) == inference.MAX_SEQ_LEN
    assert inference.max_length(UNREGISTERED, _Fake()) == min(8192, inference.MAX_SEQ_LEN)
    inference.check_length(100, 100)
    with pytest.raises(ValueError, match=r"101 bp.*never truncated"):
        inference.check_length(101, 100)


def test_restore_fp32_upcasts_only_named_tensors():
    lin = nn.Linear(2, 2).to(torch.bfloat16)
    lin.register_buffer("inv_freq", torch.ones(2, dtype=torch.bfloat16))
    exact = torch.tensor([1.0 + 1e-6, 2.0])
    n = inference._restore_fp32(lin, {"weight": torch.ones(2, 2) / 3, "inv_freq": exact})
    assert n == 2
    assert lin.weight.dtype == torch.float32 and lin.bias.dtype == torch.bfloat16
    assert torch.equal(lin.inv_freq, exact)


def _reference_fft_conv(u, k, d_bias):
    """Same math as the port's `_fft_conv` (FFT causal convolution + D skip)."""
    n = 2 * u.shape[-1]
    k_f = torch.fft.rfft(k.float(), n=n) / n
    k_f = k_f.reshape(-1, k_f.shape[-1]).unsqueeze(0)
    y = torch.fft.irfft(torch.fft.rfft(u.float(), n=n) * k_f, n=n, norm="forward")
    y = y[..., : u.shape[-1]]
    return (y + u.float() * d_bias.float().unsqueeze(-1)).to(u.dtype)


@pytest.mark.parametrize("k_shape", ["1HL", "H1K"])
def test_chunked_fft_conv_matches_unchunked(k_shape):
    torch.manual_seed(0)
    u = torch.randn(1, 10, 33)
    k = torch.randn(1, 10, 33) if k_shape == "1HL" else torch.randn(10, 1, 7)
    d = torch.randn(10)
    chunked = inference.chunked_fft_conv(_reference_fft_conv, chunk=4)
    assert torch.allclose(chunked(u, k, d), _reference_fft_conv(u, k, d), atol=1e-5)
    assert chunked.evo2_mcp_chunked


def test_iir_filter_low_memory_matches_port_formula():
    torch.manual_seed(0)
    mixer = SimpleNamespace(log_poles=-torch.rand(6, 16, 1), residues=torch.randn(6, 16))
    t = torch.arange(20, dtype=torch.float32)[None, None]
    expected = (mixer.residues[..., None] * (mixer.log_poles * t).exp()).sum(1)[None]
    got = inference._iir_filter_low_memory(mixer, 20, "cpu", torch.float32)
    assert torch.allclose(got, expected, atol=1e-5)


def test_top_k_zero_means_no_filtering():
    logits = torch.zeros(512)
    gen = torch.Generator().manual_seed(0)
    picks = {inference.sample_next(logits, 1.0, 0, gen) for _ in range(200)}
    assert picks == {ord(c) for c in "ACGT"}


def test_lm_logits_uses_fp32_head():
    model = SimpleNamespace(lm_head=nn.Linear(4, 3, bias=False).to(torch.bfloat16))
    out = inference.lm_logits(model, torch.ones(2, 4, dtype=torch.bfloat16))
    assert out.dtype == torch.float32 and out.shape == (2, 3)
