"""Shared fixtures: GENERator's vocabulary rebuilt locally, and a tiny random Llama (no download)."""

import itertools

import pytest
from generator_mcp.kmer import KmerVocab

# Same layout as GenerTeam's vocab.txt: 32 special tokens, then product("ATCG", repeat=6).
SPECIALS = [
    "<oov>",
    "<s>",
    "</s>",
    "<pad>",
    "<mask>",
    "<bog>",
    "<eog>",
    "<bok>",
    "<eok>",
    "<+>",
    "<->",
    "<cds>",
    "<pseudo>",
    "<tRNA>",
    "<rRNA>",
    "<ncRNA>",
    "<miscRNA>",
    "<mam>",
    "<vrt>",
    "<inv>",
    "<pln>",
    "<fng>",
    "<prt>",
    "<arc>",
    "<bct>",
    "<mit>",
    "<plt>",
    "<plm>",
    "<vir>",
    "<sp0>",
    "<sp1>",
    "<sp2>",
]


def vocab_text(k: int = 6) -> str:
    tokens = SPECIALS + ["".join(p) for p in itertools.product("ATCG", repeat=k)]
    return "\n".join(f"{t} {i}" for i, t in enumerate(tokens)) + "\n"


@pytest.fixture(scope="session")
def vocab() -> KmerVocab:
    return KmerVocab.from_vocab_text(vocab_text(), 6)


@pytest.fixture
def tiny_model(monkeypatch, vocab):
    """Patch `inference.load` to return a tiny random-weight Llama with GENERator's vocab."""
    import torch
    from generator_mcp import inference
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=4128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).eval()
    monkeypatch.setattr(inference, "DEVICE", "cpu")
    loaded = inference.Loaded(vocab, model, max_tokens=64)
    monkeypatch.setattr(inference, "load", lambda m: loaded)
    return loaded
