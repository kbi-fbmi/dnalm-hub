"""Shared HTTP wiring: bearer-token auth, /health, and transport selection.

Every service in this family (`ntv3-mcp` and siblings) exposes the exact same
shape: a streamable-HTTP MCP endpoint behind an optional shared-secret bearer
token, plus an always-unauthenticated `GET /health` for container healthchecks,
and picks stdio vs. HTTP transport from the same four env var names. This
module holds that logic once so each service's `server.py` only needs to
register its own tools and call `run_server(mcp, logger=logger)`.
"""

from __future__ import annotations

import logging
import os


class BearerAuthMiddleware:
    """Minimal shared-secret auth for the HTTP transport.

    Intentionally simple (a single static bearer token, no OAuth/JWT) -- enough
    to keep the MCP endpoint from being wide open when exposed on a server,
    while leaving real authn/authz (mTLS, OAuth, per-user tokens, ...) to a
    reverse proxy in front of this container if you need it.
    """

    def __init__(self, app, token: str, exempt_paths: set[str]) -> None:
        self.app = app
        self.token = token
        self.exempt_paths = exempt_paths

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["path"] in self.exempt_paths:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        provided = headers.get(b"authorization", b"").decode("latin-1")
        if provided != f"Bearer {self.token}":
            from starlette.responses import PlainTextResponse

            response = PlainTextResponse("Unauthorized", status_code=401)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


async def _health_check(request):
    from starlette.responses import PlainTextResponse

    return PlainTextResponse("ok")


def register_health_route(mcp) -> None:
    """Register an always-unauthenticated `GET /health` on `mcp`. Call before building the app."""
    mcp.custom_route("/health", methods=["GET"])(_health_check)


def build_http_app(mcp, host: str, path: str, *, auth_token_env: str, logger: logging.Logger):
    """Build the streamable-HTTP ASGI app for `mcp`, wrapped in bearer auth if `auth_token_env` is set."""
    app = mcp.streamable_http_app(streamable_http_path=path, host=host)
    token = os.environ.get(auth_token_env, "").strip()
    if token:
        app = BearerAuthMiddleware(app, token=token, exempt_paths={"/health"})
    else:
        logger.warning(
            "%s is not set: the HTTP endpoint at %s is UNAUTHENTICATED. Set %s, or put this "
            "behind a reverse proxy / VPN / firewall before exposing it beyond localhost.",
            auth_token_env,
            path,
            auth_token_env,
        )
    return app


def run_server(
    mcp,
    *,
    logger: logging.Logger,
    transport_env: str = "MCP_TRANSPORT",
    host_env: str = "MCP_HOST",
    port_env: str = "MCP_PORT",
    path_env: str = "MCP_PATH",
    auth_token_env: str = "MCP_AUTH_TOKEN",
    default_port: int = 8000,
) -> None:
    """Run `mcp` on stdio or streamable-HTTP, selected by `transport_env`.

    - "stdio" (default): standard transport for local desktop MCP clients.
    - "http"/"streamable-http": streamable-HTTP transport for server/container
      deployments, exposing the MCP endpoint at http://HOST:PORT/PATH plus an
      unauthenticated GET /health for liveness/readiness checks.
    """
    transport = os.environ.get(transport_env, "stdio").strip().lower()
    if transport == "stdio":
        mcp.run()
        return
    if transport in ("http", "streamable-http"):
        import uvicorn

        host = os.environ.get(host_env, "127.0.0.1")
        port = int(os.environ.get(port_env, str(default_port)))
        path = os.environ.get(path_env, "/mcp")
        app = build_http_app(mcp, host, path, auth_token_env=auth_token_env, logger=logger)
        logger.info("Starting %s (streamable-HTTP) on http://%s:%s%s", mcp.name, host, port, path)
        uvicorn.run(app, host=host, port=port, log_level=mcp.settings.log_level.lower())
        return
    raise ValueError(f"Unsupported {transport_env}={transport!r}; use 'stdio' or 'http'.")
