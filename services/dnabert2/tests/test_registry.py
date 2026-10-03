import pytest
from dnabert2_mcp.registry import DEFAULT_MODEL_ALIAS, MODEL_REGISTRY, resolve_model


def test_default_alias_is_registered():
    assert DEFAULT_MODEL_ALIAS in MODEL_REGISTRY


def test_resolve_known_alias_is_pinned():
    m = resolve_model("117m")
    assert m.repo_id == "zhihan1996/DNABERT-2-117M"
    assert m.revision and len(m.revision) == 40
    assert m.max_tokens == 512


def test_resolve_alias_is_case_insensitive():
    assert resolve_model("117M").repo_id == "zhihan1996/DNABERT-2-117M"


def test_resolve_none_falls_back_to_default():
    assert resolve_model(None).repo_id == MODEL_REGISTRY[DEFAULT_MODEL_ALIAS].repo_id


def test_only_dnabert2_is_registered():
    # DNABERT-S is deliberately not offered.
    assert [spec.repo_id for spec in MODEL_REGISTRY.values()] == ["zhihan1996/DNABERT-2-117M"]


def test_resolve_custom_repo_id_passthrough_is_unpinned():
    m = resolve_model("someuser/my-finetune")
    assert (m.repo_id, m.revision, m.max_tokens) == ("someuser/my-finetune", None, None)


def test_resolve_unknown_alias_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model("not-a-real-alias")
