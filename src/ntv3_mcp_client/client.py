"""Minimal HTTP helpers for ntv3-mcp deployments.

This module intentionally stays dependency-free (stdlib only) so it can be
used in small scripts and tests without adding a requests/httpx dependency.

Example:
    >>> from ntv3_mcp_client import Ntv3McpClient
    >>> client = Ntv3McpClient("http://kbi-cs2.fbmi.cvut.cz:8000", auth_token="your-token")
    >>> client.health()
    'ok'
    >>> result = client.embed_sequence("ACGTACGTACGT", checkpoint="100m-pre")
    >>> sorted(result.keys())
    ['checkpoint', 'embedding', 'layer_name', 'num_layers', 'pooling', 'sequence']
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class Ntv3McpClient:
    """Small helper for checking health and sending JSON requests.

    Args:
        base_url: Server base URL, for example http://127.0.0.1:8000.
        auth_token: Optional bearer token used as Authorization header.
        timeout: Timeout in seconds for each HTTP request.
    """

    def __init__(self, base_url: str, auth_token: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout

    def health(self) -> str:
        """Return the plain-text /health response (typically 'ok')."""
        req = self._request("GET", "/health")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8")

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON payload and return a parsed JSON object.

        This is useful for generic HTTP interactions, including custom MCP
        bridge endpoints that proxy calls into ntv3-mcp tools.
        """
        body = json.dumps(payload).encode("utf-8")
        req = self._request("POST", path, body=body, content_type="application/json")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read().decode("utf-8")
        if not data:
            return {}
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Response from {path!r} is not valid JSON: {data[:200]!r}") from exc

    def initialize_mcp(
        self,
        protocol_version: str = "2025-06-18",
        client_name: str = "ntv3_mcp_client",
        client_version: str = "1.0",
    ) -> str:
        """Initialize an MCP streamable-HTTP session and return `mcp-session-id`."""
        payload = {
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": client_version},
            },
        }
        _, headers = self._post_mcp_sse_json("/mcp", payload)
        session_id = headers.get("mcp-session-id", "").strip()
        if not session_id:
            raise RuntimeError("MCP initialize did not return mcp-session-id header.")
        return session_id

    def notify_initialized(self, session_id: str) -> None:
        """Send MCP `notifications/initialized` for an existing session."""
        payload = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        self._post_mcp_sse_json("/mcp", payload, session_id=session_id)

    def call_tool(self, session_id: str, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call one MCP tool and return the JSON-RPC result payload.

        Raises `RuntimeError` when the tool call itself returns `isError=true`.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": "tool-1",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
        message, _ = self._post_mcp_sse_json("/mcp", payload, session_id=session_id)
        result = message.get("result", {})
        if result.get("isError"):
            text_parts = [item.get("text", "") for item in result.get("content", []) if isinstance(item, dict)]
            detail = " | ".join([p for p in text_parts if p]) or "Unknown MCP tool error."
            raise RuntimeError(detail)
        return result

    def embed_sequence(
        self,
        sequence: str,
        checkpoint: str = "100m-pre",
        layer_name: str = "last",
        pooling: str = "mean",
    ) -> dict[str, Any]:
        """Run `embed_sequence` in one short-lived MCP session and return structured content."""
        session_id = self.initialize_mcp()
        self.notify_initialized(session_id)
        result = self.call_tool(
            session_id,
            "embed_sequence",
            {
                "sequence": sequence,
                "checkpoint": checkpoint,
                "layer_name": layer_name,
                "pooling": pooling,
            },
        )
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        # Fallback: return full MCP result if server omits structuredContent.
        return result

    def list_available_checkpoints(self) -> list[dict[str, Any]]:
        """Return checkpoint metadata exposed by `list_available_checkpoints`."""
        session_id = self.initialize_mcp()
        self.notify_initialized(session_id)
        result = self.call_tool(session_id, "list_available_checkpoints")
        structured = result.get("structuredContent", {})
        checkpoints = structured.get("result", []) if isinstance(structured, dict) else []
        return checkpoints if isinstance(checkpoints, list) else []

    def _post_mcp_sse_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """POST JSON to MCP streamable-HTTP and parse the first SSE data message."""
        body = json.dumps(payload).encode("utf-8")
        req = self._request("POST", path, body=body, content_type="application/json", session_id=session_id)
        req.add_header("Accept", "application/json, text/event-stream")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
            headers = {k.lower(): v for k, v in resp.headers.items()}
        data_lines = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
        if not data_lines:
            raise ValueError(f"MCP response did not include SSE data lines: {raw[:300]!r}")
        try:
            message = json.loads(data_lines[0])
        except json.JSONDecodeError as exc:
            raise ValueError(f"MCP SSE data was not valid JSON: {data_lines[0][:300]!r}") from exc
        return message, headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        session_id: str | None = None,
    ) -> urllib.request.Request:
        full_path = path if path.startswith("/") else f"/{path}"
        headers = {"Accept": "application/json, text/plain;q=0.9, */*;q=0.8"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        if content_type:
            headers["Content-Type"] = content_type
        return urllib.request.Request(
            url=f"{self.base_url}{full_path}",
            data=body,
            headers=headers,
            method=method,
        )
