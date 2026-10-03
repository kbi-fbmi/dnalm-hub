"""Pure-logic tests for the post-trained (species / annotation / tracks) helpers. No downloads."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ntv3_mcp.inference import (
    POST_MODEL_TYPE,
    bin_mean,
    center_region,
    check_output_size,
    layer_lengths,
    resolve_species,
)


def _model(model_type):
    species = {"<pad>": 1, "<unk>": 0, "human": 6, "mouse": 7}
    return SimpleNamespace(
        config=SimpleNamespace(model_type=model_type, species_to_token_id=species)
    )


def test_post_model_defaults_to_human():
    assert resolve_species(_model(POST_MODEL_TYPE), None) == "human"


def test_post_model_species_is_normalized():
    assert resolve_species(_model(POST_MODEL_TYPE), "  Mouse ") == "mouse"


def test_post_model_rejects_unknown_species_and_lists_valid_ones():
    with pytest.raises(ValueError, match=r"Unknown species 'martian'.*'human', 'mouse'"):
        resolve_species(_model(POST_MODEL_TYPE), "martian")


def test_pre_model_rejects_species():
    with pytest.raises(ValueError, match="only applies to post-trained"):
        resolve_species(_model("ntv3"), "human")
    assert resolve_species(_model("ntv3"), None) is None


def test_center_region_matches_model_crop():
    # 256 padded positions, keep 37.5% -> 96 positions starting at 80 (verified on the real model).
    assert center_region(256, 256, 0.375) == (80, 80, 176)


def test_center_region_is_clipped_to_real_sequence():
    assert center_region(256, 150, 0.375) == (80, 80, 150)


def test_center_region_too_short_sequence_raises():
    with pytest.raises(ValueError, match="Sequence too short"):
        center_region(256, 60, 0.375)


def test_bin_mean_averages_and_keeps_partial_last_bin():
    values = np.arange(10, dtype=float).reshape(5, 2)  # rows: [0,1],[2,3],[4,5],[6,7],[8,9]
    np.testing.assert_allclose(bin_mean(values, 2), [[1, 2], [5, 6], [8, 9]])
    assert bin_mean(values, 1) is values


def test_output_size_guard_suggests_bin_size():
    check_output_size(1000, 21, 1)  # 21k values: fine
    with pytest.raises(ValueError, match=r"bin_size \(>= 5\)"):
        check_output_size(100_000, 21, 1)


def test_layer_lengths_ceil_divides_by_downsampling():
    # 200 real bases padded to 256: full layer 200, /2 -> 100, /128 -> 2, /4 of 75 -> 19
    real = torch.tensor([200, 75])
    assert layer_lengths(real, 256, 256).tolist() == [200, 75]
    assert layer_lengths(real, 256, 128).tolist() == [100, 38]
    assert layer_lengths(real, 256, 2).tolist() == [2, 1]
