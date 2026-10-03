import pytest
from ntv2_mcp.registry import DEFAULT_MODEL_ALIAS, MODEL_REGISTRY, resolve_model


def test_default_alias_is_registered():
    assert DEFAULT_MODEL_ALIAS in MODEL_REGISTRY


def test_resolve_known_alias_is_pinned():
    m = resolve_model("500m")
    assert m.repo_id == "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    assert m.revision and len(m.revision) == 40
    assert m.max_tokens == 2048


def test_resolve_alias_is_case_insensitive():
    assert (
        resolve_model("100M").repo_id == "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
    )


def test_resolve_none_falls_back_to_default():
    assert resolve_model(None).repo_id == MODEL_REGISTRY[DEFAULT_MODEL_ALIAS].repo_id


def test_v1_checkpoint_has_shorter_context():
    m = resolve_model("2.5b-v1")
    assert m.repo_id == "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
    assert m.max_tokens == 1000


def test_resolve_custom_repo_id_passthrough_is_unpinned():
    m = resolve_model("someuser/my-finetuned-ntv2")
    assert (m.repo_id, m.revision, m.max_tokens) == ("someuser/my-finetuned-ntv2", None, None)


def test_resolve_unknown_alias_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model("not-a-real-alias")


def test_all_registry_entries_are_well_formed():
    for alias, spec in MODEL_REGISTRY.items():
        assert spec.repo_id.startswith("InstaDeepAI/"), alias
        assert spec.revision and len(spec.revision) == 40, alias
        assert spec.generation in ("v1", "v2"), alias
