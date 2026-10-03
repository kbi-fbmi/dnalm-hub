"""Pure-logic tests for the DNABERT-2 inference helpers (no model weights).

The hidden-state capture is exercised against a tiny fake model that mimics the
remote code's structure (`model.bert.embeddings`, `model.bert.encoder.layer[i]`)
and its quirk of running encoder layers on *unpadded* `(total_tokens, hidden)`
activations.
"""

import types

import pytest
import torch
from dnabert2_mcp import inference
from dnabert2_mcp.inference import (
    _force_pytorch_attention,
    _hidden_states,
    _parse_layer,
    _without_triton_import_check,
    check_token_lengths,
    max_tokens,
    repad,
)
from dnabert2_mcp.registry import ResolvedModel, resolve_model
from torch import nn
from transformers import dynamic_module_utils


class _Tok:
    def __init__(self, model_max_length):
        self.model_max_length = model_max_length


# --- context limit -----------------------------------------------------------


def test_lengths_at_limit_pass():
    check_token_lengths([3, 512], 512)


def test_over_limit_raises_clear_error_naming_the_sequence():
    with pytest.raises(ValueError, match=r"Sequence 1 tokenizes to 513 tokens.*never truncated"):
        check_token_lengths([10, 513], 512)


def test_registry_limit_wins_over_tokenizer():
    assert max_tokens(resolve_model("117m"), _Tok(4096)) == 512


def test_unregistered_repo_uses_tokenizer_limit():
    assert max_tokens(ResolvedModel("u/x", None, None), _Tok(128)) == 128


def test_unregistered_repo_with_sentinel_limit_falls_back():
    # DNABERT-2's own tokenizer reports this sentinel.
    assert max_tokens(ResolvedModel("u/x", None, None), _Tok(int(1e30))) == 512


# --- layer parsing -------------------------------------------------------------


@pytest.mark.parametrize(
    ("layer", "expected"), [("last", 12), (None, 12), ("0", 0), (6, 6), ("-1", 12), (-13, 0)]
)
def test_parse_layer(layer, expected):
    assert _parse_layer(layer, 12) == expected


@pytest.mark.parametrize("layer", ["13", "-14", "middle"])
def test_parse_layer_rejects_bad_values(layer):
    with pytest.raises(ValueError, match="layer must be"):
        _parse_layer(layer, 12)


# --- remote-code workarounds ---------------------------------------------------


def test_triton_import_check_is_skipped_only_inside_the_context(tmp_path):
    f = tmp_path / "flash_attn_triton.py"
    f.write_text("import torch\nimport triton\nimport triton.language as tl\n")
    assert "triton" in dynamic_module_utils.get_imports(f)
    with _without_triton_import_check():
        assert dynamic_module_utils.get_imports(f) == ["torch"]
    assert "triton" in dynamic_module_utils.get_imports(f)


def test_force_pytorch_attention_clears_the_remote_kernel(monkeypatch):
    remote = types.ModuleType("fake_remote_bert_layers")
    remote.flash_attn_qkvpacked_func = object()
    monkeypatch.setitem(__import__("sys").modules, remote.__name__, remote)
    model_cls = type("BertForMaskedLM", (), {"__module__": remote.__name__})
    _force_pytorch_attention(model_cls())
    assert remote.flash_attn_qkvpacked_func is None


# --- unpadded hidden-state capture ---------------------------------------------


def test_repad_scatters_rows_in_row_major_mask_order():
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    rows = torch.arange(5, dtype=torch.float32).unsqueeze(-1)  # (5 real tokens, hidden=1)
    out = repad(rows, mask)
    assert out.shape == (2, 3, 1)
    assert out[..., 0].tolist() == [[0, 1, 2], [3, 4, 0]]


def test_repad_passes_padded_tensors_through():
    h = torch.randn(2, 3, 4)
    assert repad(h, torch.ones(2, 3)) is h


def test_repad_detects_row_count_mismatch():
    with pytest.raises(RuntimeError, match="unpadding changed"):
        repad(torch.zeros(4, 2), torch.tensor([[1, 1, 1], [1, 1, 0]]))


class _FakeLayer(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, h):  # (total_tokens, hidden), like the remote encoder layers
        return h + self.k


class _FakeBert(nn.Module):
    def __init__(self, n_layers=3, hidden=4):
        super().__init__()
        self.hidden = hidden
        self.embeddings = nn.Identity()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(_FakeLayer(k + 1) for k in range(n_layers))

    def forward(self, input_ids, attention_mask):
        h = self.embeddings(input_ids.float().unsqueeze(-1).expand(-1, -1, self.hidden))
        flat = h[attention_mask.bool()]  # unpad
        for layer in self.encoder.layer:
            flat = layer(flat)
        return repad(flat, attention_mask), None


def test_hidden_states_captures_requested_layer_and_repads():
    model = types.SimpleNamespace(bert=_FakeBert())
    batch = {
        "input_ids": torch.tensor([[10, 20, 30], [40, 50, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
    }
    emb = _hidden_states(model, batch, 0)
    assert emb.shape == (2, 3, 4)
    assert emb[1, 2, 0].item() == 0  # padding id passes through the embedding block as-is
    layer2 = _hidden_states(model, batch, 2)  # +1 then +2
    assert layer2[..., 0].tolist() == [[13, 23, 33], [43, 53, 0]]  # padding row is zeroed
    # hooks are removed afterwards
    assert all(not layer._forward_hooks for layer in model.bert.encoder.layer)


def test_compute_embeddings_mean_pooling_ignores_padding(monkeypatch):
    class _Tokenizer:
        def __call__(self, seqs, **_):
            ids = [[1] + [5] * len(s) + [2] for s in seqs]
            width = max(map(len, ids))
            return {
                "input_ids": torch.tensor([i + [3] * (width - len(i)) for i in ids]),
                "attention_mask": torch.tensor(
                    [[1] * len(i) + [0] * (width - len(i)) for i in ids]
                ),
            }

    model = types.SimpleNamespace(bert=_FakeBert(n_layers=2))
    monkeypatch.setattr(inference, "load", lambda m: (_Tokenizer(), model))
    monkeypatch.setattr(inference, "DEVICE", "cpu")
    m = resolve_model("117m")
    (single,), _, _ = inference.compute_embeddings(m, ["AC"], "last", "mean")
    (batched, _), idx, n = inference.compute_embeddings(m, ["AC", "ACGTACGT"], "last", "mean")
    assert (idx, n) == (2, 2)
    assert single.tolist() == pytest.approx(batched.tolist())
    rows, _, _ = inference.compute_embeddings(m, ["AC", "ACGTACGT"], 1, "per_token")
    assert [r.shape for r in rows] == [(4, 4), (10, 4)]
    with pytest.raises(ValueError, match="invalid character"):
        inference.compute_embeddings(m, ["ACGX"], "last", "mean")
    with pytest.raises(ValueError, match="pooling"):
        inference.compute_embeddings(m, ["AC"], "last", "max")
