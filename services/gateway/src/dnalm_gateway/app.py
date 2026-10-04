"""OpenAI-compatible embeddings API in front of the dnalm-hub model services.

    POST /v1/embeddings     {"model": "ntv3/100m-pre", "input": "ACGT..." | ["ACGT...", ...]}
    GET  /v1/models         every checkpoint of every service, as "<service>/<checkpoint>"
    GET  /health            unauthenticated liveness check
    GET  /version           unauthenticated {"name", "version"}

Works with the official OpenAI SDKs (`base_url="http://host:8080/v1"`). Each input is
one DNA sequence; the vector is the mean-pooled embedding from the service's
`embed_sequence` tool. Any non-OpenAI request field (send it via `extra_body`) is passed
to that tool as an argument, e.g. `layer` (alias of `layer_name`), `species` (NTv3
post-trained) or `remainder` (GENERator); the service rejects arguments it doesn't know.
`usage` counts bases, not model tokens.
"""

from __future__ import annotations

import base64
import logging
import os
import struct
from importlib.metadata import version
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from .backends import Backends, BackendUnavailableError, UnknownModelError, parse_backends

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dnalm-gateway")


def _error(
    status: int, message: str, *, type_: str, param: str | None = None, code: str | None = None
):
    """An OpenAI-style error body, so SDKs raise their usual exception types."""
    body = {"error": {"message": message, "type": type_, "param": param, "code": code}}
    return JSONResponse(body, status_code=status)


def _invalid(message: str, param: str | None = None):
    return _error(400, message, type_="invalid_request_error", param=param)


def _encode(vector: list[float], encoding_format: str) -> list[float] | str:
    if encoding_format == "base64":
        # Little-endian float32, which is what the OpenAI SDKs decode.
        return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")
    return vector


# Standard OpenAI embeddings fields; everything else goes to the service's embed_sequence.
_OPENAI_FIELDS = {"model", "input", "encoding_format", "dimensions", "user"}
# embed_sequence arguments the gateway sets itself.
_RESERVED_TOOL_ARGS = {"sequence", "checkpoint", "pooling"}


def _parse_embedding_request(body: Any) -> dict[str, Any] | JSONResponse:
    """Validate the request body; return the parsed fields or an error response."""
    if not isinstance(body, dict):
        return _invalid("Request body must be a JSON object.")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        return _invalid("'model' is required, e.g. 'ntv3/100m-pre' (see GET /v1/models).", "model")
    inputs = body.get("input")
    if isinstance(inputs, str):
        inputs = [inputs]
    if (
        not isinstance(inputs, list)
        or not inputs
        or not all(isinstance(s, str) and s for s in inputs)
    ):
        return _invalid(
            "'input' must be a non-empty DNA sequence or a non-empty list of them "
            "(token-id arrays are not supported).",
            "input",
        )
    encoding_format = body.get("encoding_format") or "float"
    if encoding_format not in ("float", "base64"):
        return _invalid("'encoding_format' must be 'float' or 'base64'.", "encoding_format")
    if body.get("dimensions") is not None:
        return _invalid(
            "'dimensions' is not supported: vectors have the model's hidden size.", "dimensions"
        )
    options = {k: v for k, v in body.items() if k not in _OPENAI_FIELDS}
    reserved = sorted(options.keys() & _RESERVED_TOOL_ARGS)
    if reserved:
        return _invalid(f"{reserved[0]!r} is set by the gateway and can't be passed.", reserved[0])
    if "layer" in options:
        layer = options.pop("layer")
        if isinstance(layer, bool) or not isinstance(layer, (str, int)):
            return _invalid("'layer' must be 'last' or a layer index.", "layer")
        options["layer_name"] = str(layer)
    return {
        "model": model.strip(),
        "inputs": inputs,
        "encoding_format": encoding_format,
        "options": options,
    }


def create_app(backends: Backends, *, auth_token: str | None = None) -> Starlette:
    """Build the ASGI app. With `auth_token`, every route except /health needs `Bearer <token>`."""

    def authorized(request: Request) -> bool:
        return not auth_token or request.headers.get("authorization") == f"Bearer {auth_token}"

    def unauthorized():
        return _error(
            401,
            "Invalid or missing API key.",
            type_="invalid_request_error",
            code="invalid_api_key",
        )

    async def health(request: Request):
        return PlainTextResponse("ok")

    async def version_info(request: Request):
        return JSONResponse({"name": "dnalm-gateway", "version": version("dnalm-gateway")})

    async def list_models(request: Request):
        if not authorized(request):
            return unauthorized()
        models = await run_in_threadpool(backends.list_models)
        return JSONResponse({"object": "list", "data": models})

    async def retrieve_model(request: Request):
        if not authorized(request):
            return unauthorized()
        model_id = request.path_params["model"]
        models = await run_in_threadpool(backends.list_models)
        for model in models:
            if model["id"] == model_id:
                return JSONResponse(model)
        return _error(
            404,
            f"Model {model_id!r} not found.",
            type_="invalid_request_error",
            code="model_not_found",
        )

    async def embeddings(request: Request):
        if not authorized(request):
            return unauthorized()
        try:
            body = await request.json()
        except ValueError:
            return _invalid("Request body is not valid JSON.")
        parsed = _parse_embedding_request(body)
        if isinstance(parsed, JSONResponse):
            return parsed
        try:
            backend, checkpoint = await run_in_threadpool(backends.resolve, parsed["model"])
            vectors = await run_in_threadpool(
                backends.embed,
                backend,
                checkpoint,
                parsed["inputs"],
                parsed["options"],
            )
        except UnknownModelError as exc:
            return _error(
                404, str(exc), type_="invalid_request_error", param="model", code="model_not_found"
            )
        except BackendUnavailableError as exc:
            logger.warning("%s", exc)
            return _error(502, str(exc), type_="server_error")
        except (RuntimeError, ValueError) as exc:
            # The service's own message: invalid bases, sequence too long, bad layer, ...
            return _invalid(str(exc))
        n_bases = sum(len(s) for s in parsed["inputs"])
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": _encode(v, parsed["encoding_format"]),
                    }
                    for i, v in enumerate(vectors)
                ],
                "model": f"{backend}/{checkpoint}",
                "usage": {"prompt_tokens": n_bases, "total_tokens": n_bases},
            }
        )

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/version", version_info, methods=["GET"]),
            Route("/v1/models", list_models, methods=["GET"]),
            Route("/v1/models/{model:path}", retrieve_model, methods=["GET"]),
            Route("/v1/embeddings", embeddings, methods=["POST"]),
        ]
    )


def app_from_env() -> Starlette:
    """Build the app from GATEWAY_* / MCP_AUTH_TOKEN environment variables (see README)."""
    auth_token = os.environ.get("MCP_AUTH_TOKEN", "").strip() or None
    backend_token = os.environ.get("GATEWAY_BACKEND_TOKEN", "").strip() or auth_token
    backends = Backends(
        urls=parse_backends(os.environ.get("GATEWAY_BACKENDS", "")),
        auth_token=backend_token,
        timeout=float(os.environ.get("GATEWAY_TIMEOUT", "600")),
        max_batch=max(1, int(os.environ.get("GATEWAY_MAX_BATCH", "32"))),
    )
    if not auth_token:
        logger.warning(
            "MCP_AUTH_TOKEN is not set: the OpenAI API is UNAUTHENTICATED. Set it, or keep this "
            "behind a reverse proxy / VPN / firewall before exposing it beyond localhost."
        )
    logger.info("dnalm-gateway %s, model services: %s", version("dnalm-gateway"), backends.urls)
    return create_app(backends, auth_token=auth_token)


def main() -> None:
    """Entry point used by `uv run dnalm-gateway` and the Docker image."""
    import uvicorn

    host = os.environ.get("GATEWAY_HOST", "127.0.0.1")
    port = int(os.environ.get("GATEWAY_PORT", "8080"))
    uvicorn.run(app_from_env(), host=host, port=port)


if __name__ == "__main__":
    main()
