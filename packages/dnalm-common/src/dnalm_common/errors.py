"""Surface input-validation errors to MCP clients.

The MCP SDK's `MCPServer` (mcp>=2) only forwards the message of a `ToolError`;
any other exception reaches the client as a bare "Error executing tool <name>"
(the traceback stays in the server log). Every backend here reports bad input
-- invalid bases, out-of-range positions, over-long sequences, unknown
checkpoints -- as `ValueError`, and those messages are exactly what a caller
needs to fix the request. `user_errors_as_tool_errors` re-raises them as
`ToolError`; anything else is still treated as a crash and stays masked.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar

from mcp.server.mcpserver.exceptions import ToolError

F = TypeVar("F", bound=Callable)


def user_errors_as_tool_errors(fn: F) -> F:
    """Wrap a (sync) tool function so `ValueError` messages reach the MCP client."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    return wrapper  # type: ignore[return-value]
