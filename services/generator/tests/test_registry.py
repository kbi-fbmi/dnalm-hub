import pytest
from generator_mcp.registry import DEFAULT_MODEL_ALIAS, MODEL_REGISTRY, resolve_model


def test_default_alias_is_registered():
    assert DEFAULT_MODEL_ALIAS in MODEL_REGISTRY


def test_resolve_none_falls_back_to_default():
    assert resolve_model(None).repo_id == MODEL_REGISTRY[DEFAULT_MODEL_ALIAS].repo_id


def test_resolve_known_alias_is_pinned():
    m = resolve_model("V2-Eukaryote-1.2B")
    assert m.repo_id == "GenerTeam/GENERator-v2-eukaryote-1.2b-base"
    assert m.revision and len(m.revision) == 40
    assert m.max_tokens == 16384
    assert m.separator is True


def test_v1_checkpoints_embed_without_separator():
    assert resolve_model("eukaryote-1.2b").separator is False


def test_resolve_custom_repo_id_passthrough_is_unpinned():
    m = resolve_model("someuser/my-finetune")
    assert (m.repo_id, m.revision, m.max_tokens, m.separator) == (
        "someuser/my-finetune",
        None,
        None,
        False,
    )
    assert resolve_model("someuser/GENERator-v2-ft").separator is True


def test_resolve_unknown_alias_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model("not-a-real-alias")


def test_all_registry_entries_are_well_formed():
    for alias, spec in MODEL_REGISTRY.items():
        assert spec.repo_id.startswith("GenerTeam/GENERator-"), alias
        assert spec.revision and len(spec.revision) == 40, alias
        assert spec.generation in ("v1", "v2"), alias
        assert spec.domain in ("eukaryote", "prokaryote"), alias
        assert (spec.generation == "v2") == ("-v2-" in spec.repo_id), alias
        assert spec.domain in spec.repo_id, alias
