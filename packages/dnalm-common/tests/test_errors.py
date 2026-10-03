"""Checks that ValueError text reaches MCP clients and crashes stay masked (in-process, no network)."""

import asyncio

import pytest
from dnalm_common.errors import user_errors_as_tool_errors
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer("errors-test")


@mcp.tool()
@user_errors_as_tool_errors
def bad_input(sequence: str, checkpoint: str = "x") -> dict:
    """Docstring must survive wrapping."""
    raise ValueError(f"Sequence {sequence!r} is too long for {checkpoint}.")


@mcp.tool()
@user_errors_as_tool_errors
def crashes() -> dict:
    raise RuntimeError("internal detail /secret/path")


def _call(name, args):
    return asyncio.run(mcp.call_tool(name, args))


def test_value_error_message_reaches_client():
    with pytest.raises(ToolError, match=r"'ACGT' is too long for y"):
        _call("bad_input", {"sequence": "ACGT", "checkpoint": "y"})


def test_other_exceptions_stay_masked():
    with pytest.raises(ToolError) as info:
        _call("crashes", {})
    assert "secret" not in str(info.value)


def test_wrapping_preserves_tool_schema_and_docs():
    tool = {t.name: t for t in asyncio.run(mcp.list_tools())}["bad_input"]
    assert set(tool.input_schema["properties"]) == {"sequence", "checkpoint"}
    assert tool.input_schema.get("required") == ["sequence"]
    assert "Docstring must survive" in (tool.description or "")
