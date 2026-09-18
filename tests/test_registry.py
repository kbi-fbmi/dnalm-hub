import pytest

from ntv3_mcp.registry import DEFAULT_MODEL_ALIAS, MODEL_REGISTRY, resolve_model


def test_default_alias_is_registered():
    assert DEFAULT_MODEL_ALIAS in MODEL_REGISTRY


def test_resolve_known_alias():
    repo_id, pad_multiple = resolve_model("100m-pre")
    assert repo_id == "InstaDeepAI/NTv3_100M_pre"
    assert pad_multiple == 128


def test_resolve_alias_is_case_insensitive():
    repo_id, _ = resolve_model("100M-PRE")
    assert repo_id == "InstaDeepAI/NTv3_100M_pre"


def test_resolve_none_falls_back_to_default():
    repo_id, _ = resolve_model(None)
    default_repo_id = MODEL_REGISTRY[DEFAULT_MODEL_ALIAS].repo_id
    assert repo_id == default_repo_id


def test_resolve_custom_repo_id_passthrough():
    repo_id, pad_multiple = resolve_model("someuser/my-finetuned-ntv3")
    assert repo_id == "someuser/my-finetuned-ntv3"
    assert pad_multiple == 128


def test_resolve_custom_5downsample_repo_id_guesses_pad_multiple():
    repo_id, pad_multiple = resolve_model("someuser/My5DownsampleModel")
    assert repo_id == "someuser/My5DownsampleModel"
    assert pad_multiple == 32


def test_resolve_unknown_alias_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model("not-a-real-alias")


def test_all_registry_entries_have_slash_in_repo_id():
    for alias, spec in MODEL_REGISTRY.items():
        assert "/" in spec.repo_id, f"{alias} repo_id looks malformed: {spec.repo_id}"
