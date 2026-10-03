"""No-download tests for the tokenizer-agnostic whole-sequence SNP scoring."""

import math

import pytest
import torch

from dnalm_common.scoring import (
    WHOLE_SEQUENCE_METHOD,
    apply_snp,
    masked_lm_pseudo_log_likelihood,
    score_snp_whole_sequence,
)

MASK_ID = 9
VOCAB = 10


def test_apply_snp_defaults_to_center_and_reads_ref_from_sequence():
    seq, mutated, pos, ref, alt = apply_snp("acgta", "t")
    assert (seq, mutated, pos, ref, alt) == ("ACGTA", "ACTTA", 2, "G", "T")


def test_apply_snp_rejects_out_of_range_position():
    with pytest.raises(ValueError, match="position must be within"):
        apply_snp("ACGT", "A", position=4)


@pytest.mark.parametrize("alt", ["", "AC", "X"])
def test_apply_snp_rejects_bad_alt_allele(alt):
    with pytest.raises(ValueError, match="alt_allele"):
        apply_snp("ACGT", alt)


def test_score_snp_whole_sequence_reports_delta_and_method():
    scores = {"ACGTA": -10.0, "ACTTA": -12.5}
    result = score_snp_whole_sequence("ACGTA", "T", scores.__getitem__)
    assert result["original_score"] == -10.0
    assert result["mutated_score"] == -12.5
    assert result["score_delta"] == pytest.approx(-2.5)
    assert result["method"] == WHOLE_SEQUENCE_METHOD
    assert result["center_position"] == 2
    assert result["reference_allele"] == "G"


def test_score_snp_whole_sequence_skips_second_pass_when_alt_equals_ref():
    calls = []

    def ll(s):
        calls.append(s)
        return -1.0

    result = score_snp_whole_sequence("ACGTA", "G", ll)
    assert calls == ["ACGTA"]
    assert result["score_delta"] == 0.0


def _uniform_except_mask_logits(batch: torch.Tensor) -> torch.Tensor:
    """Fake model: at a masked column `c`, token id `c % VOCAB` gets 3x the weight of every
    other token; unmasked columns are NaN so reading one would poison the sum."""
    b, n = batch.shape
    logits = torch.zeros(b, n, VOCAB)
    for row in range(b):
        for col in range(n):
            # Only masked positions should ever be read; make unmasked ones poison.
            if batch[row, col] != MASK_ID:
                logits[row, col] = float("nan")
            else:
                logits[row, col, col % VOCAB] = math.log(3.0)  # this token gets 3x the weight
    return logits


def test_pll_sums_true_token_log_probs_at_masked_positions_only():
    # Token ids chosen so position i holds id i: each true token is the favored one.
    input_ids = torch.tensor([0, 1, 2, 3, 4])
    total = masked_lm_pseudo_log_likelihood(input_ids, [1, 2, 3], MASK_ID, _uniform_except_mask_logits, batch_size=2)
    favored = math.log(3.0 / (3.0 + (VOCAB - 1)))
    assert total == pytest.approx(3 * favored)


def test_pll_is_independent_of_batch_size():
    input_ids = torch.tensor([0, 5, 2, 7, 4, 1])
    positions = [0, 1, 2, 3, 4, 5]
    results = {
        bs: masked_lm_pseudo_log_likelihood(input_ids, positions, MASK_ID, _uniform_except_mask_logits, bs)
        for bs in (1, 4, 100)
    }
    assert results[1] == pytest.approx(results[4]) == pytest.approx(results[100])


def test_pll_rejects_batched_input():
    with pytest.raises(ValueError, match="1D"):
        masked_lm_pseudo_log_likelihood(torch.zeros(2, 3, dtype=torch.long), [0], MASK_ID, _uniform_except_mask_logits)
