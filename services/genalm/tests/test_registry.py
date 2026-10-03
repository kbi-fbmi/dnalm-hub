import pytest
from genalm_mcp.registry import DEFAULT_MODEL_ALIAS, MODEL_REGISTRY, resolve_model


def test_default_alias_is_registered():
    assert DEFAULT_MODEL_ALIAS in MODEL_REGISTRY


def test_default_is_bert_large():
    assert resolve_model(None).repo_id == "AIRI-Institute/gena-lm-bert-large-t2t"


def test_resolve_known_alias_is_pinned():
    m = resolve_model("bert-base-t2t")
    assert m.repo_id == "AIRI-Institute/gena-lm-bert-base-t2t"
    assert m.revision and len(m.revision) == 40
    assert (m.max_tokens, m.architecture) == (512, "bert")


def test_resolve_alias_is_case_insensitive():
    assert resolve_model("BERT-Large-T2T").repo_id == "AIRI-Institute/gena-lm-bert-large-t2t"


def test_bigbird_has_longer_context():
    m = resolve_model("bigbird-base-t2t")
    assert m.repo_id == "AIRI-Institute/gena-lm-bigbird-base-t2t"
    assert (m.max_tokens, m.architecture) == (4096, "bigbird")


def test_resolve_custom_repo_id_passthrough_is_unpinned():
    m = resolve_model("someuser/my-finetune")
    assert (m.repo_id, m.revision, m.max_tokens, m.architecture) == (
        "someuser/my-finetune",
        None,
        None,
        None,
    )


def test_resolve_unknown_alias_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model("not-a-real-alias")


def test_all_registry_entries_are_well_formed():
    for alias, spec in MODEL_REGISTRY.items():
        assert spec.repo_id.startswith("AIRI-Institute/gena-lm-"), alias
        assert spec.revision and len(spec.revision) == 40, alias
        assert spec.architecture in ("bert", "bigbird"), alias
        assert spec.max_tokens == (4096 if spec.architecture == "bigbird" else 512), alias
