"""MCP server exposing InstaDeep's Nucleotide Transformer v3 (NTv3) as tools.

Run directly for local testing:

    uv run ntv3-mcp

Or point any MCP-compatible client (Claude Desktop, Claude Code, Cursor, etc.) at
this module over stdio -- see README.md for client configuration examples.

Tool names, the `checkpoint` parameter, and JSON return shapes are aligned with
evo2-mcp (https://evo2-mcp.readthedocs.io/) wherever the two models' semantics
allow, so that a service driving both a causal (Evo 2) and a masked (NTv3)
genomic LM through MCP can reuse similar call shapes. See README.md's
"Compatibility with evo2-mcp" section for exactly where the two diverge and why.
"""

import logging
import os

from mcp.server.mcpserver import MCPServer

from . import inference
from .registry import DEFAULT_MODEL_ALIAS, MODEL_REGISTRY, resolve_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ntv3-mcp")

mcp = MCPServer(
    "ntv3-mcp",
    instructions=(
        "Tools for running InstaDeep's Nucleotide Transformer v3 (NTv3) genomic "
        "language models over raw DNA sequences: computing sequence embeddings, "
        "zero-shot single-nucleotide variant effect scoring, and masked-nucleotide "
        "prediction. Sequences are plain strings using IUPAC letters A/C/G/T/N only "
        "(case-insensitive). Call list_available_checkpoints first to see available "
        "models. NTv3 is a masked (bidirectional) DNA language model, not a "
        "generative one -- there is no sequence-generation tool here. NOTE: NTv3 "
        "model weights are distributed under a NON-COMMERCIAL license by InstaDeep "
        "and are gated on HuggingFace (see README); check the relevant model card "
        "before production or commercial use. This server's own code is MIT licensed."
    ),
)


@mcp.tool()
def list_available_checkpoints() -> list[dict]:
    """List NTv3 checkpoints available as the `checkpoint` argument of the other tools.

    Same shape as evo2-mcp's tool of the same name: a list of {"name", "description"}
    dicts (plus extra NTv3-specific metadata: repo_id, params, stage, context,
    pad_multiple). Any other HuggingFace repo id implementing the NTv3
    trust_remote_code architecture -- e.g. your own fine-tuned checkpoint -- also
    works as `checkpoint` even though it won't be listed here.
    """
    return [
        {
            "name": alias,
            "description": spec.notes or f"NTv3 {spec.params} ({spec.stage}-trained, {spec.context} context).",
            "repo_id": spec.repo_id,
            "params": spec.params,
            "stage": spec.stage,
            "context": spec.context,
            "pad_multiple": spec.pad_multiple,
            "default": alias == DEFAULT_MODEL_ALIAS,
        }
        for alias, spec in MODEL_REGISTRY.items()
    ]


@mcp.tool()
def get_model_info(checkpoint: str = DEFAULT_MODEL_ALIAS) -> dict:
    """Load an NTv3 checkpoint and report its architecture details.

    Downloads the model on first use (cached afterwards for the life of the server
    process). Returns device/dtype, parameter count, vocab size, hidden size, number
    of hidden-state layers available for `embed_sequence`, and the raw model config.
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    return inference.model_info(repo_id, pad_multiple)


@mcp.tool()
def get_embedding_layers(checkpoint: str, which: str = "recommended") -> dict:
    """List hidden-state layer indices usable as `layer_name` in embed_sequence.

    Mirrors evo2-mcp's tool of the same name. `which="recommended"` is a light
    heuristic (final layer and the one before it) since InstaDeep does not publish
    a best-layer recommendation per NTv3 checkpoint the way some embedding models
    do; `which="all"` lists every valid layer index.
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    result = inference.list_embedding_layers(repo_id, pad_multiple, which)
    return {"checkpoint": repo_id, **result}


@mcp.tool()
def embed_sequence(
    sequence: str | list[str],
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    layer_name: str = "last",
    pooling: str = "mean",
) -> dict:
    """Return NTv3 feature representations (embeddings) for one or more DNA sequences.

    Args:
        sequence: A single DNA sequence, or a list of sequences to embed as a batch
            (letters A/C/G/T/N only). A single string in, a single embedding out --
            matching evo2-mcp's `embed_sequence` call shape; passing a list is an
            NTv3-mcp extension for batching.
        checkpoint: A name from list_available_checkpoints, or a full HuggingFace repo id.
        layer_name: Hidden-state layer to read, as an integer string (e.g. "6") or
            "last". See get_embedding_layers for valid values. (NTv3 layers are
            indices, not named modules like evo2-mcp's "blocks.2.mlp.l3".)
        pooling: "mean" averages token embeddings over the real (non-padding)
            sequence into one vector per sequence (default; convenient for
            similarity/search). "per_token" returns the full per-nucleotide
            embedding matrix, matching evo2-mcp's un-pooled 2D output -- can be
            large for long sequences.
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    is_batch = isinstance(sequence, list)
    sequences = sequence if is_batch else [sequence]
    embeddings, layer_idx, n_layers = inference.compute_embeddings(
        repo_id, sequences, layer_name, pooling, pad_multiple
    )
    base = {
        "checkpoint": repo_id,
        "layer_name": layer_idx,
        "num_layers": n_layers,
        "pooling": pooling,
    }
    if is_batch:
        return {**base, "sequences": sequences, "embeddings": embeddings.tolist()}
    return {**base, "sequence": sequences[0], "embedding": embeddings[0].tolist()}


@mcp.tool()
def score_snp(
    sequence: str,
    alternative_allele: str,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    position: int | None = None,
) -> dict:
    """Zero-shot effect score for a single-nucleotide substitution.

    Mirrors evo2-mcp's `score_snp`: pass a reference `sequence` and the
    `alternative_allele` to test. `position` defaults to the sequence center (same
    default evo2-mcp always uses); pass it explicitly to score any other site.

    Method differs from evo2-mcp by necessity: NTv3 is a masked (bidirectional)
    model, so instead of comparing whole-sequence log-likelihoods of the original
    vs. mutated sequence (evo2-mcp's approach, valid for its causal model), this
    masks the single position and compares NTv3's predicted log-probability of the
    alternative vs. reference allele there (`original_score`/`mutated_score` below
    are that reference/alternative log-probability, and `score_delta` their
    difference). A negative delta means NTv3 finds the alternate allele less likely
    than the reference in context (a common heuristic for "more likely
    deleterious"). This is a rough, unvalidated heuristic, not a clinical or
    functional prediction.
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    result = inference.score_variant(repo_id, sequence, alternative_allele, pad_multiple, position=position)
    return {
        "checkpoint": repo_id,
        "original_sequence": result["sequence"],
        "mutated_sequence": result["mutated_sequence"],
        "center_position": result["position"],
        "reference_allele": result["ref_allele"],
        "alternative_allele": result["alt_allele"],
        "original_score": result["ref_log_prob"],
        "mutated_score": result["alt_log_prob"],
        "score_delta": result["log_likelihood_ratio"],
        "method": "single_position_masked_lm_log_prob",
    }


@mcp.tool()
def predict_masked_positions(
    sequence: str,
    positions: list[int] | None = None,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    top_k: int = 4,
) -> dict:
    """Predict nucleotide probabilities at masked positions in a DNA sequence.

    NTv3-specific: has no evo2-mcp equivalent (evo2 is generative and would instead
    use `generate_sequence`, which has no NTv3 analog since NTv3 cannot generate
    novel sequence). If `positions` is omitted, every 'N' character in `sequence`
    is treated as a position to fill in. Returns the top_k most likely nucleotides
    (softmax probabilities restricted to A/C/G/T/N) for each requested position.
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    results = inference.predict_masked(repo_id, sequence, positions or [], top_k, pad_multiple)
    return {"checkpoint": repo_id, "predictions": results}


@mcp.tool()
def compare_sequences(
    sequence_a: str,
    sequence_b: str,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    layer_name: str = "last",
) -> dict:
    """Compute cosine similarity between the mean-pooled NTv3 embeddings of two DNA sequences.

    NTv3-specific: has no evo2-mcp equivalent.
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    similarity, layer_idx = inference.compare_sequences(repo_id, sequence_a, sequence_b, layer_name, pad_multiple)
    return {"checkpoint": repo_id, "layer_name": layer_idx, "cosine_similarity": similarity}


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Liveness/readiness probe for container orchestration -- never requires auth."""
    from starlette.responses import PlainTextResponse

    return PlainTextResponse("ok")


class _BearerAuthMiddleware:
    """Minimal shared-secret auth for the HTTP transport.

    This is intentionally simple (a single static bearer token, no OAuth/JWT) --
    enough to keep the MCP endpoint from being wide open when exposed on a server,
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


def _build_http_app(host: str, path: str):
    app = mcp.streamable_http_app(streamable_http_path=path, host=host)
    token = os.environ.get("MCP_AUTH_TOKEN", "").strip()
    if token:
        app = _BearerAuthMiddleware(app, token=token, exempt_paths={"/health"})
    else:
        logger.warning(
            "MCP_AUTH_TOKEN is not set: the HTTP endpoint at %s is UNAUTHENTICATED. "
            "Set MCP_AUTH_TOKEN, or put this behind a reverse proxy / VPN / firewall "
            "before exposing it beyond localhost.",
            path,
        )
    return app


def main() -> None:
    """Entry point used by `uv run ntv3-mcp` and the Docker image's CMD.

    MCP_TRANSPORT selects the transport:
      - "stdio" (default): standard transport for local desktop MCP clients.
      - "http": streamable-HTTP transport for server/container deployments,
        exposing the MCP endpoint at http://MCP_HOST:MCP_PORT/MCP_PATH plus an
        unauthenticated GET /health for liveness/readiness checks.
    """
    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    if transport == "stdio":
        mcp.run()
        return
    if transport in ("http", "streamable-http"):
        import uvicorn

        host = os.environ.get("MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_PORT", "8000"))
        path = os.environ.get("MCP_PATH", "/mcp")
        app = _build_http_app(host, path)
        logger.info("Starting ntv3-mcp (streamable-HTTP) on http://%s:%s%s", host, port, path)
        uvicorn.run(app, host=host, port=port, log_level=mcp.settings.log_level.lower())
        return
    raise ValueError(f"Unsupported MCP_TRANSPORT={transport!r}; use 'stdio' or 'http'.")


if __name__ == "__main__":
    main()
