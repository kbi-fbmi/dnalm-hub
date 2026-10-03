"""Minimal HTTP helpers for dnalm-hub deployments.

This module intentionally stays dependency-free (stdlib only) so it can be
used in small scripts and tests without adding a requests/httpx dependency.

A single client reuses one MCP session across calls (created lazily on first
use, and transparently re-created if the server reports it expired), so
calling several tools back to back only pays the initialize/notify round
trip once.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class GenericMcpClient:
    """Small helper for calling a dnalm-hub server's tools over streamable-HTTP MCP.

    Every service in this family (ntv3-mcp and siblings) exposes the same tool
    names and argument shapes, so this one base class works against any of
    them; each service's own thin subclass exists mainly for a family-specific
    docstring/name, not different behavior. A server that omits a given tool
    (e.g. `predict_masked_positions` on a non-single-nucleotide-tokenizer
    backend) will simply raise `RuntimeError` from `call_tool` when asked for
    it -- see each service's README for which tools it supports.

    Args:
        base_url: Server base URL, for example http://127.0.0.1:8000.
        auth_token: Optional bearer token used as Authorization header.
        timeout: Timeout in seconds for each HTTP request.
    """

    def __init__(self, base_url: str, auth_token: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout
        self._session_id: str | None = None

    def __enter__(self) -> "GenericMcpClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Drop the cached MCP session id; the next tool call starts a fresh session."""
        self._session_id = None

    def health(self) -> str:
        """Return the plain-text /health response (typically 'ok')."""
        req = self._request("GET", "/health")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8")

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON payload and return a parsed JSON object.

        This is useful for generic HTTP interactions, including custom MCP
        bridge endpoints that proxy calls into this server's tools.
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
        client_name: str = "dnalm_client",
        client_version: str = "1.0",
    ) -> str:
        """Initialize an MCP streamable-HTTP session and return `mcp-session-id`.

        Most callers don't need this directly -- `call_tool` and the tool
        wrapper methods below create and cache a session automatically.
        """
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
        """Send MCP `notifications/initialized` for an existing session.

        Notifications get no JSON-RPC reply (servers respond with an empty
        202), so this posts the payload without parsing an SSE body.
        """
        payload = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        body = json.dumps(payload).encode("utf-8")
        req = self._request("POST", "/mcp", body=body, content_type="application/json", session_id=session_id)
        req.add_header("Accept", "application/json, text/event-stream")
        with urllib.request.urlopen(req, timeout=self.timeout):
            pass

    def _ensure_session(self) -> str:
        """Return the cached session id, creating one on first use."""
        if self._session_id is None:
            session_id = self.initialize_mcp()
            self.notify_initialized(session_id)
            self._session_id = session_id
        return self._session_id

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call one MCP tool and return the JSON-RPC result payload.

        Reuses this client's cached MCP session, creating one on first call
        and transparently re-creating it once if the server reports the
        session is gone (e.g. after a server restart or idle timeout).
        Raises `RuntimeError` when the tool call itself returns `isError=true`
        (including when `name` isn't a tool this server registers).
        """
        payload = {
            "jsonrpc": "2.0",
            "id": "tool-1",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
        session_id = self._ensure_session()
        try:
            message, _ = self._post_mcp_sse_json("/mcp", payload, session_id=session_id)
        except urllib.error.HTTPError as exc:
            if exc.code not in (400, 404):
                raise
            self.close()
            session_id = self._ensure_session()
            message, _ = self._post_mcp_sse_json("/mcp", payload, session_id=session_id)
        result = message.get("result", {})
        if result.get("isError"):
            text_parts = [item.get("text", "") for item in result.get("content", []) if isinstance(item, dict)]
            detail = " | ".join([p for p in text_parts if p]) or "Unknown MCP tool error."
            raise RuntimeError(detail)
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the server's tool definitions (`name`, `description`, `inputSchema`, ...).

        Tool sets differ per model family (e.g. only NTv3 has
        `predict_masked_positions`), so check this before calling a tool.
        """
        payload = {"jsonrpc": "2.0", "id": "tools-1", "method": "tools/list", "params": {}}
        message, _ = self._post_mcp_sse_json("/mcp", payload, session_id=self._ensure_session())
        tools = message.get("result", {}).get("tools", [])
        return tools if isinstance(tools, list) else []

    def call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        """Call any tool by name and return its parsed result.

        For model-specific tools that have no wrapper method here, e.g.
        `client.call("annotate_sequence", sequence=seq, species="human")` on ntv3-mcp.
        Tools that return a list come back as `{"result": [...]}`.
        """
        return self._call_structured(tool, arguments)

    def _call_structured(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call a tool and return its structured result.

        Prefers `structuredContent` when the server provides it; otherwise
        parses the JSON text out of `content[0].text` (servers that return a
        plain `dict` from a tool function may only populate `content`, not
        `structuredContent`, depending on MCP SDK version/configuration).
        Falls back to the raw result if neither shape is present/parseable.
        """
        result = self.call_tool(name, arguments)
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        content = result.get("content")
        if isinstance(content, list) and content:
            text = content[0].get("text") if isinstance(content[0], dict) else None
            if isinstance(text, str):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    return parsed
        return result

    def list_available_checkpoints(self) -> list[dict[str, Any]]:
        """Return checkpoint metadata exposed by `list_available_checkpoints`."""
        structured = self._call_structured("list_available_checkpoints")
        checkpoints = structured.get("result", [])
        return checkpoints if isinstance(checkpoints, list) else []

    def get_model_info(self, checkpoint: str = "100m-pre") -> dict[str, Any]:
        """Load a checkpoint and report architecture details (hidden size, layers, device, ...)."""
        return self._call_structured("get_model_info", {"checkpoint": checkpoint})

    def get_embedding_layers(self, checkpoint: str, which: str = "recommended") -> dict[str, Any]:
        """List valid hidden-state layer indices for `embed_sequence`."""
        return self._call_structured("get_embedding_layers", {"checkpoint": checkpoint, "which": which})

    def embed_sequence(
        self,
        sequence: str | list[str],
        checkpoint: str = "100m-pre",
        layer_name: str = "last",
        pooling: str = "mean",
        species: str | None = None,
    ) -> dict[str, Any]:
        """Compute feature representations (embeddings) for one or more DNA sequences.

        `sequence` may be a single string, or a list of strings to embed as a
        batch (see `embed_sequence` on the server for the resulting shape).
        `species` is only for species-conditioned models (NTv3 post-trained).
        """
        arguments = {"sequence": sequence, "checkpoint": checkpoint, "layer_name": layer_name, "pooling": pooling}
        return self._call_structured("embed_sequence", _with_species(arguments, species))

    def score_snp(
        self,
        sequence: str,
        alternative_allele: str,
        checkpoint: str = "100m-pre",
        position: int | None = None,
        species: str | None = None,
    ) -> dict[str, Any]:
        """Zero-shot effect score for a single-nucleotide substitution.

        `position` defaults to the sequence center; pass it explicitly to
        score any other site. The scoring *method* varies by backend (see
        that server's docstring/README) but the response shape is the same.
        """
        arguments: dict[str, Any] = {
            "sequence": sequence,
            "alternative_allele": alternative_allele,
            "checkpoint": checkpoint,
        }
        if position is not None:
            arguments["position"] = position
        return self._call_structured("score_snp", _with_species(arguments, species))

    def predict_masked_positions(
        self,
        sequence: str,
        positions: list[int] | None = None,
        checkpoint: str = "100m-pre",
        top_k: int = 4,
        species: str | None = None,
    ) -> dict[str, Any]:
        """Predict top-k nucleotide probabilities at masked ('N') positions in a sequence.

        Only available on backends with a single-nucleotide tokenizer; raises
        `RuntimeError` against a server that doesn't register this tool.
        If `positions` is omitted, every 'N' character in `sequence` is used.
        """
        arguments = {"sequence": sequence, "positions": positions or [], "checkpoint": checkpoint, "top_k": top_k}
        return self._call_structured("predict_masked_positions", _with_species(arguments, species))

    def compare_sequences(
        self,
        sequence_a: str,
        sequence_b: str,
        checkpoint: str = "100m-pre",
        layer_name: str = "last",
        species: str | None = None,
    ) -> dict[str, Any]:
        """Cosine similarity between the mean-pooled embeddings of two sequences."""
        arguments = {"sequence_a": sequence_a, "sequence_b": sequence_b, "checkpoint": checkpoint, "layer_name": layer_name}
        return self._call_structured("compare_sequences", _with_species(arguments, species))

    def generate_sequence(
        self,
        prompt: str,
        checkpoint: str = "100m-pre",
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 4,
    ) -> dict[str, Any]:
        """Autoregressively sample a continuation. Only available on causal (generative) backends."""
        return self._call_structured(
            "generate_sequence",
            {
                "prompt": prompt,
                "checkpoint": checkpoint,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
            },
        )

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


def _with_species(arguments: dict[str, Any], species: str | None) -> dict[str, Any]:
    """Add `species` only when given, so servers without that parameter keep working."""
    if species is not None:
        arguments["species"] = species
    return arguments
