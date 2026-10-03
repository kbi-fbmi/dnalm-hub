"""Locks in the MCP tool surface. Does not download any model weights."""

import asyncio

from evo2_mcp.server import mcp

# Causal LM: generate_sequence, no predict_masked_positions (no mask token).
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


def test_score_snp_matches_shared_call_shape():
    schema = _list_tools()["score_snp"].input_schema
    assert {"sequence", "alternative_allele", "checkpoint", "position"} <= set(schema["properties"])
    assert set(schema.get("required", [])) == {"sequence", "alternative_allele"}


def test_generate_sequence_matches_client_call_shape():
    schema = _list_tools()["generate_sequence"].input_schema
    expected = {"prompt", "checkpoint", "max_new_tokens", "temperature", "top_k"}
    assert expected <= set(schema["properties"])
    assert set(schema.get("required", [])) == {"prompt"}


def test_list_available_checkpoints_needs_no_model():
    from evo2_mcp.server import list_available_checkpoints

    listed = list_available_checkpoints()
    assert [c["name"] for c in listed if c["default"]] == ["7b"]
    assert all(c["max_sequence_length"] <= c["max_tokens"] for c in listed)
