"""Inference logic on a tiny random-weight Llama with GENERator's vocabulary (no download)."""

import itertools

import numpy as np
import pytest
import torch
from generator_mcp import inference
from generator_mcp.registry import ResolvedModel

V2 = ResolvedModel("x/GENERator-v2-test", None, 64, separator=True)
V1 = ResolvedModel("x/GENERator-test", None, 64, separator=False)


def _token_log_probs(lm, ids):
    with torch.no_grad():
        logits = lm.model(input_ids=torch.tensor([ids])).logits[0].float()
    return torch.log_softmax(logits, dim=-1)


def test_check_token_lengths():
    inference.check_token_lengths([10, 64], 64)
    with pytest.raises(ValueError, match=r"Sequence 1 needs 65 tokens.*never\s+truncated"):
        inference.check_token_lengths([10, 65], 64)


def test_log_likelihood_is_the_exact_token_log_likelihood(tiny_model):
    seq = "ACGTACGGTTCAAAAAAA"
    ids = [1, *tiny_model.vocab.encode(seq)]
    lp = _token_log_probs(tiny_model, ids)
    expected = sum(lp[t, ids[t + 1]].item() for t in range(len(ids) - 1))
    assert inference.log_likelihood(V2, seq) == pytest.approx(expected, abs=1e-4)


def test_partial_last_kmer_is_marginalized_not_dropped(tiny_model):
    vocab = tiny_model.vocab
    full, tail = "ACGTAC", "GG"
    lp = _token_log_probs(tiny_model, [1, *vocab.encode(full)])
    # Brute force: sum P over every 6-mer starting with the tail.
    completions = ["GG" + "".join(p) for p in itertools.product("ACGT", repeat=4)]
    p_tail = sum(lp[1, vocab.token_to_id[c]].exp().item() for c in completions)
    expected = lp[0, vocab.token_to_id[full]].item() + np.log(p_tail)
    assert inference.log_likelihood(V2, full + tail) == pytest.approx(expected, abs=1e-4)


def test_n_kmer_is_marginalized_and_fed_as_oov(tiny_model):
    vocab = tiny_model.vocab
    seq = "ACNTACGGTTCA"
    ids = [1, vocab.oov_id, vocab.token_to_id["GGTTCA"]]
    lp = _token_log_probs(tiny_model, ids)
    p_first = sum(lp[0, vocab.token_to_id[f"AC{b}TAC"]].exp().item() for b in "ACGT")
    expected = np.log(p_first) + lp[1, ids[2]].item()
    assert inference.log_likelihood(V2, seq) == pytest.approx(expected, abs=1e-4)


def test_score_snp_shape_and_method(tiny_model):
    res = inference.score_variant(V2, "ACGTACGGTTCAAAA", "T", position=7)
    assert res["method"] == "whole_sequence_log_prob_delta"
    assert res["reference_allele"] == "G"
    assert res["score_delta"] == pytest.approx(res["mutated_score"] - res["original_score"])


def test_batch_equals_single_and_padding_is_ignored(tiny_model):
    seqs = ["ACGTACGGTTCA", "ACGTAC" * 5]
    for pooling in ("mean", "last_token"):
        batch, *_ = inference.compute_embeddings(V2, seqs, "last", pooling)
        for s, b in zip(seqs, batch, strict=True):
            (single,), *_ = inference.compute_embeddings(V2, [s], "last", pooling)
            np.testing.assert_allclose(b, single, atol=1e-5)


def test_per_token_rows_are_tokens(tiny_model):
    (v2,), *_ = inference.compute_embeddings(V2, ["ACGTAC" * 3], 1, "per_token")
    (v1,), *_ = inference.compute_embeddings(V1, ["ACGTAC" * 3], 1, "per_token")
    assert v2.shape == (5, 32)  # <s> + 3 six-mers + <s> separator
    assert v1.shape == (4, 32)  # <s> + 3 six-mers


def test_embedding_remainder_policy(tiny_model):
    with pytest.raises(ValueError, match="multiple of 6"):
        inference.compute_embeddings(V2, ["ACGTACGT"], "last", "mean")
    _, _, _, adjusted = inference.compute_embeddings(
        V2, ["ACGTACGT", "ACGTAC"], "last", "mean", remainder="trim_left"
    )
    assert adjusted == [2, 0]


def test_embedding_errors(tiny_model):
    with pytest.raises(ValueError, match="pooling"):
        inference.compute_embeddings(V2, ["ACGTAC"], "last", "max")
    with pytest.raises(ValueError, match="layer must be between"):
        inference.compute_embeddings(V2, ["ACGTAC"], 9, "mean")
    with pytest.raises(ValueError, match="invalid character"):
        inference.compute_embeddings(V2, ["ACGTAX"], "last", "mean")
    with pytest.raises(ValueError, match="context"):
        inference.compute_embeddings(V2, ["ACGTAC" * 63], "last", "mean")


@pytest.mark.parametrize("sampling", ["base", "token"])
def test_generate_returns_bases_and_is_seeded(tiny_model, sampling):
    a = inference.generate(V2, "ACGTAC", 3, 1.0, 0, sampling=sampling, seed=7)
    b = inference.generate(V2, "ACGTAC", 3, 1.0, 0, sampling=sampling, seed=7)
    assert a == b
    assert a["num_new_tokens"] == 3 and a["num_new_bases"] == 18
    assert set(a["generated_sequence"]) <= set("ACGT")
    assert a["full_sequence"] == "ACGTAC" + a["generated_sequence"]


def test_greedy_token_generation_is_argmax(tiny_model):
    res = inference.generate(V2, "ACGTAC", 1, 0.0, 0, sampling="token")
    lp = _token_log_probs(tiny_model, [1, tiny_model.vocab.token_to_id["ACGTAC"]])
    kmer_ids = list(tiny_model.vocab.kmer_ids)
    best = kmer_ids[int(lp[-1, kmer_ids].argmax())]
    assert res["generated_sequence"] == tiny_model.vocab.decode([best])


def test_generate_validation(tiny_model):
    with pytest.raises(ValueError, match="multiple of 6"):
        inference.generate(V2, "ACGTA", 2, 1.0, 0)
    res = inference.generate(V2, "ACGTA", 1, 0.0, 0, remainder="pad_left")
    assert res["prompt"] == "AACGTA" and res["remainder_bases"] == 1
    with pytest.raises(ValueError, match="max_new_tokens"):
        inference.generate(V2, "ACGTAC", 0, 1.0, 0)
    with pytest.raises(ValueError, match="temperature"):
        inference.generate(V2, "ACGTAC", 1, -1.0, 0)
    with pytest.raises(ValueError, match="sampling"):
        inference.generate(V2, "ACGTAC", 1, 1.0, 0, sampling="beam")
    with pytest.raises(ValueError, match="exceeds"):
        inference.generate(V2, "ACGTAC", 63, 1.0, 0)
