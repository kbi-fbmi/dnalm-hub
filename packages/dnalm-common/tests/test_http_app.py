"""Exercises the shared auth/health wiring in-process (no real socket, no model weights)."""

from dnalm_common.http_app import build_http_app, register_health_route
from mcp.server.mcpserver import MCPServer
from starlette.testclient import TestClient


def _make_app(host: str, path: str, monkeypatch, token: str | None):
    mcp = MCPServer("test-server")
    register_health_route(mcp)
    if token is None:
        monkeypatch.delenv("TEST_AUTH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("TEST_AUTH_TOKEN", token)
    return build_http_app(
        mcp,
        host,
        path,
        auth_token_env="TEST_AUTH_TOKEN",
        logger=__import__("logging").getLogger("test"),
    )


def test_health_is_open_without_token(monkeypatch):
    app = _make_app("127.0.0.1", "/mcp", monkeypatch, token=None)
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200


def test_mcp_endpoint_open_when_no_token_configured(monkeypatch):
    app = _make_app("127.0.0.1", "/mcp", monkeypatch, token=None)
    with TestClient(app) as client:
        resp = client.get("/mcp")
    assert resp.status_code != 401


def test_mcp_endpoint_requires_token_when_configured(monkeypatch):
    app = _make_app("127.0.0.1", "/mcp", monkeypatch, token="s3cret")
    with TestClient(app) as client:
        assert client.get("/mcp").status_code == 401
        assert client.get("/mcp", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/mcp", headers={"Authorization": "Bearer s3cret"}).status_code != 401


def test_health_stays_open_even_when_token_configured(monkeypatch):
    app = _make_app("127.0.0.1", "/mcp", monkeypatch, token="s3cret")
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
