"""Locks in the MCP tool surface. Does not download any model weights."""

import asyncio

from __PKG__.server import mcp

# TODO: add "predict_masked_positions" (single-nucleotide masked LM) or
# "generate_sequence" (causal LM) here if this model registers them.
EXPECTED_TOOLS = {
    "list_available_checkpoints",
    "get_model_info",
    "embed_sequence",
    "score_snp",
    "compare_sequences",
}


def _list_tools():
    return {t.name: t for t in asyncio.run(mcp.list_tools())}


def test_expected_tools_are_registered():
    assert set(_list_tools()) == EXPECTED_TOOLS


def test_score_snp_matches_shared_call_shape():
    schema = _list_tools()["score_snp"].input_schema
    assert {"sequence", "alternative_allele", "checkpoint", "position"} <= set(schema["properties"])
    assert set(schema.get("required", [])) == {"sequence", "alternative_allele"}
