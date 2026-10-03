"""Locks in the MCP tool surface. Does not download any model weights."""

import asyncio

from generator_mcp.server import mcp

EXPECTED_TOOLS = {
    "list_available_checkpoints",
    "get_model_info",
    "get_embedding_layers",
    "embed_sequence",
    "score_snp",
    "compare_sequences",
    "generate_sequence",
}


def _list_tools():
    return {t.name: t for t in asyncio.run(mcp.list_tools())}


def test_expected_tools_are_registered():
    assert set(_list_tools()) == EXPECTED_TOOLS


def test_no_masked_prediction_tool():
    # Causal model with 6-mer tokens: masking a single base is not defined.
    assert "predict_masked_positions" not in _list_tools()


def test_checkpoint_param_used_instead_of_model():
    tools = _list_tools()
    for name in EXPECTED_TOOLS - {"list_available_checkpoints"}:
        props = tools[name].input_schema["properties"]
        assert "checkpoint" in props, f"{name} should expose a 'checkpoint' param"
        assert "model" not in props


def test_score_snp_matches_shared_call_shape():
    schema = _list_tools()["score_snp"].input_schema
    assert {"sequence", "alternative_allele", "checkpoint", "position"} <= set(schema["properties"])
    assert set(schema.get("required", [])) == {"sequence", "alternative_allele"}


def test_generate_sequence_matches_client_signature():
    # dnalm_client.GenericMcpClient.generate_sequence sends exactly these arguments.
    schema = _list_tools()["generate_sequence"].input_schema
    assert {"prompt", "checkpoint", "max_new_tokens", "temperature", "top_k"} <= set(
        schema["properties"]
    )
    assert set(schema.get("required", [])) == {"prompt"}


def test_list_available_checkpoints_marks_exactly_one_default():
    from generator_mcp.server import list_available_checkpoints

    entries = list_available_checkpoints()
    assert sum(e["default"] for e in entries) == 1
    assert all(e["revision"] for e in entries)
