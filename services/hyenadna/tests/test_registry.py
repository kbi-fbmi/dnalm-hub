import pytest
from hyenadna_mcp.registry import DEFAULT_MODEL_ALIAS, MODEL_REGISTRY, resolve_model


def test_default_alias_is_registered():
    assert DEFAULT_MODEL_ALIAS in MODEL_REGISTRY


def test_resolve_none_falls_back_to_default():
    assert resolve_model(None).repo_id == MODEL_REGISTRY[DEFAULT_MODEL_ALIAS].repo_id


def test_resolve_custom_repo_id_passthrough_is_unpinned():
    m = resolve_model("someuser/my-finetune")
    assert (m.repo_id, m.revision, m.max_tokens) == ("someuser/my-finetune", None, None)


def test_resolve_unknown_alias_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model("not-a-real-alias")
