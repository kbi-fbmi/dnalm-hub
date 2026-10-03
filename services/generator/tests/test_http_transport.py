"""Exercises the HTTP transport's auth/health wiring in-process (no real socket,
no model weights needed)."""

from generator_mcp.server import _build_http_app
from starlette.testclient import TestClient


def test_health_is_open_without_token(monkeypatch):
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    app = _build_http_app("127.0.0.1", "/mcp")
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200


def test_mcp_endpoint_open_when_no_token_configured(monkeypatch):
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    app = _build_http_app("127.0.0.1", "/mcp")
    with TestClient(app) as client:
        resp = client.get("/mcp")
    # No auth token configured -> request reaches the MCP app itself (whatever it
    # replies, it must not be the auth middleware's 401).
    assert resp.status_code != 401


def test_mcp_endpoint_requires_token_when_configured(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_TOKEN", "s3cret")
    app = _build_http_app("127.0.0.1", "/mcp")
    with TestClient(app) as client:
        assert client.get("/mcp").status_code == 401
        assert client.get("/mcp", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/mcp", headers={"Authorization": "Bearer s3cret"}).status_code != 401


def test_health_stays_open_even_when_token_configured(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_TOKEN", "s3cret")
    app = _build_http_app("127.0.0.1", "/mcp")
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
