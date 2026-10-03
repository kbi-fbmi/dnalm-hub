"""Locks in the MCP tool surface: names and required params, aligned with evo2-mcp
conventions where applicable. Does not download any model weights."""

import asyncio

from ntv3_mcp.server import mcp

EXPECTED_TOOLS = {
    "list_available_checkpoints",
    "get_model_info",
    "get_embedding_layers",
    "embed_sequence",
    "score_snp",
    "predict_masked_positions",
    "compare_sequences",
    # post-trained checkpoints only
    "list_species",
    "list_tracks",
    "annotate_sequence",
    "predict_tracks",
}


def _list_tools():
    return asyncio.run(mcp.list_tools())


def test_expected_tools_are_registered():
    names = {t.name for t in _list_tools()}
    assert names == EXPECTED_TOOLS


def test_checkpoint_param_used_instead_of_model():
    tools = {t.name: t for t in _list_tools()}
    for name in [
        "get_model_info",
        "embed_sequence",
        "score_snp",
        "predict_masked_positions",
        "compare_sequences",
    ]:
        props = tools[name].input_schema["properties"]
        assert "checkpoint" in props, f"{name} should expose a 'checkpoint' param"
        assert "model" not in props, f"{name} should not expose a legacy 'model' param"


def test_score_snp_matches_evo2_call_shape():
    props = {"sequence", "alternative_allele", "checkpoint", "position"}
    tools = {t.name: t for t in _list_tools()}
    schema_props = set(tools["score_snp"].input_schema["properties"])
    assert props <= schema_props
    # sequence and alternative_allele are required; checkpoint/position have defaults
    assert set(tools["score_snp"].input_schema.get("required", [])) == {
        "sequence",
        "alternative_allele",
    }


def test_species_is_optional_on_shared_tools():
    tools = {t.name: t for t in _list_tools()}
    for name in ["embed_sequence", "score_snp", "predict_masked_positions", "compare_sequences"]:
        schema = tools[name].input_schema
        assert "species" in schema["properties"], name
        assert "species" not in schema.get("required", []), name


def test_post_tools_require_only_their_inputs():
    tools = {t.name: t for t in _list_tools()}
    assert set(tools["annotate_sequence"].input_schema.get("required", [])) == {"sequence"}
    assert set(tools["predict_tracks"].input_schema.get("required", [])) == {
        "sequence",
        "track_ids",
    }
    assert not tools["list_species"].input_schema.get("required")
