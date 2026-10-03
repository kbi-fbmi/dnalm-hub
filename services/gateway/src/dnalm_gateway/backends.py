"""Model services behind the gateway: configuration, model-name resolution, embedding calls.

OpenAI model ids are `<service>/<checkpoint>`, e.g. `ntv3/100m-pre` or
`ntv2/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`. A bare checkpoint
alias or HuggingFace repo id also works when exactly one service lists it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dnalm_client import GenericMcpClient

logger = logging.getLogger("dnalm-gateway")

ClientFactory = Callable[..., GenericMcpClient]


class UnknownModelError(ValueError):
    """The requested model id matches no service or checkpoint (OpenAI: 404 model_not_found)."""


class BackendUnavailableError(RuntimeError):
    """A model service could not be reached or rejected the gateway (OpenAI: 502)."""


def parse_backends(spec: str) -> dict[str, str]:
    """Parse `GATEWAY_BACKENDS` ("ntv3=http://ntv3:8000,ntv2=http://ntv2:8000") into name -> URL."""
    backends: dict[str, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, url = item.partition("=")
        name, url = name.strip(), url.strip()
        if not sep or not name or not url or "/" in name:
            raise ValueError(
                f"Invalid GATEWAY_BACKENDS entry {item!r}; expected name=http://host:port."
            )
        backends[name] = url.rstrip("/")
    if not backends:
        raise ValueError("GATEWAY_BACKENDS is empty; expected e.g. ntv3=http://ntv3:8000.")
    return backends


@dataclass
class Backends:
    """The configured model services, with a lazily filled cache of their checkpoint lists.

    A new MCP client is created per call: the client's session cache is not
    thread-safe, and the extra initialize round trip is negligible next to a
    forward pass.
    """

    urls: dict[str, str]
    auth_token: str | None = None
    timeout: float = 600.0
    max_batch: int = 32
    client_factory: ClientFactory = GenericMcpClient
    _checkpoints: dict[str, list[dict[str, Any]]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _client(self, backend: str) -> GenericMcpClient:
        return self.client_factory(
            self.urls[backend], auth_token=self.auth_token, timeout=self.timeout
        )

    def checkpoints(self, backend: str, *, refresh: bool = False) -> list[dict[str, Any]]:
        """Checkpoints listed by one service; cached after the first successful call."""
        with self._lock:
            cached = self._checkpoints.get(backend)
        if cached is not None and not refresh:
            return cached
        try:
            listed = self._client(backend).list_available_checkpoints()
        except (RuntimeError, ValueError):
            raise
        except Exception as exc:
            raise BackendUnavailableError(
                f"Model service {backend!r} is unavailable: {exc}"
            ) from exc
        with self._lock:
            self._checkpoints[backend] = listed
        return listed

    def list_models(self) -> list[dict[str, Any]]:
        """Every checkpoint of every reachable service, as OpenAI model objects."""
        models = []
        for backend in self.urls:
            try:
                checkpoints = self.checkpoints(backend)
            except BackendUnavailableError as exc:
                logger.warning("Skipping %s in /v1/models: %s", backend, exc)
                continue
            for ckpt in checkpoints:
                models.append(
                    {
                        "id": f"{backend}/{ckpt['name']}",
                        "object": "model",
                        "created": 0,
                        "owned_by": backend,
                        "description": ckpt.get("description"),
                        "repo_id": ckpt.get("repo_id"),
                    }
                )
        return models

    def resolve(self, model: str) -> tuple[str, str]:
        """Map an OpenAI model id to (service, checkpoint)."""
        model = model.strip()
        backend, sep, checkpoint = model.partition("/")
        if sep and backend in self.urls and checkpoint:
            return backend, checkpoint
        for refresh in (False, True):
            matches = self._find_checkpoint(model, refresh=refresh)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                options = ", ".join(f"{b}/{c}" for b, c in matches)
                raise UnknownModelError(f"Model {model!r} is ambiguous; use one of: {options}.")
        raise UnknownModelError(
            f"Model {model!r} not found. Use '<service>/<checkpoint>' with a service from "
            f"{sorted(self.urls)}; GET /v1/models lists the checkpoints."
        )

    def _find_checkpoint(self, model: str, *, refresh: bool) -> list[tuple[str, str]]:
        """Services listing `model` as a checkpoint alias or repo id (unreachable ones are skipped)."""
        matches = []
        for backend in self.urls:
            try:
                checkpoints = self.checkpoints(backend, refresh=refresh)
            except BackendUnavailableError:
                continue
            for ckpt in checkpoints:
                if model in (ckpt.get("name"), ckpt.get("repo_id")):
                    matches.append((backend, ckpt["name"]))
                    break
        return matches

    def embed(
        self,
        backend: str,
        checkpoint: str,
        sequences: list[str],
        *,
        layer: str = "last",
        species: str | None = None,
    ) -> list[list[float]]:
        """Mean-pooled embeddings, in `max_batch`-sized calls to the service's `embed_sequence`.

        Raises `RuntimeError` with the service's message for bad input (invalid
        bases, too long, unknown checkpoint, ...) and `BackendUnavailableError`
        when the service can't be reached.
        """
        client = self._client(backend)
        vectors: list[list[float]] = []
        for start in range(0, len(sequences), self.max_batch):
            chunk = sequences[start : start + self.max_batch]
            try:
                result = client.embed_sequence(
                    chunk, checkpoint=checkpoint, layer_name=layer, pooling="mean", species=species
                )
            except (RuntimeError, ValueError):
                raise
            except Exception as exc:
                raise BackendUnavailableError(
                    f"Model service {backend!r} is unavailable: {exc}"
                ) from exc
            embeddings = result.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(chunk):
                raise BackendUnavailableError(
                    f"Model service {backend!r} returned an unexpected embed_sequence response."
                )
            vectors.extend(embeddings)
        return vectors
