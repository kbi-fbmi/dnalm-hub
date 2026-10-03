"""MCP server exposing __NAME__ as tools.

Uses the tool names, `checkpoint` parameter and response shapes shared by every
service in this repo (see docs/adding-a-model.md), so clients and the gateway
can treat all model families alike. Register only the tools the model can
honestly support:

- `predict_masked_positions`: masked LMs with a 1-base-per-token tokenizer only
  (see `dnalm_common.assert_single_nucleotide_tokenizer`),
- `generate_sequence`: causal LMs only.
"""

import logging

from dnalm_common import user_errors_as_tool_errors
from dnalm_common.http_app import build_http_app, register_health_route, run_server
from mcp.server.mcpserver import MCPServer

from . import inference
from .registry import DEFAULT_MODEL_ALIAS, MODEL_REGISTRY, resolve_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__NAME__-mcp")

mcp = MCPServer(
    "__NAME__-mcp",
    instructions=(
        # TODO: what the model is, what the tools do, context limits, license.
        "Tools for running the __NAME__ genomic language model(s) over raw DNA sequences "
        "(A/C/G/T/N). Call list_available_checkpoints first."
    ),
)


@mcp.tool()
@user_errors_as_tool_errors
def list_available_checkpoints() -> list[dict]:
    """List checkpoints available as the `checkpoint` argument of the other tools."""
    return [
        {
            "name": alias,
            "description": spec.notes or f"__NAME__ {spec.params}, {spec.max_tokens}-token context.",
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "params": spec.params,
            "max_tokens": spec.max_tokens,
            "default": alias == DEFAULT_MODEL_ALIAS,
        }
        for alias, spec in MODEL_REGISTRY.items()
    ]


@mcp.tool()
@user_errors_as_tool_errors
def get_model_info(checkpoint: str = DEFAULT_MODEL_ALIAS) -> dict:
    """Load a checkpoint and report its architecture details."""
    return inference.model_info(resolve_model(checkpoint))


@mcp.tool()
@user_errors_as_tool_errors
def embed_sequence(
    sequence: str | list[str],
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    layer_name: str = "last",
    pooling: str = "mean",
) -> dict:
    """Return embeddings for one DNA sequence, or for a list of sequences as one batch.

    pooling="mean" gives one vector per sequence; "per_token" one row per token.
    """
    m = resolve_model(checkpoint)
    is_batch = isinstance(sequence, list)
    sequences = sequence if is_batch else [sequence]
    embeddings, layer_idx, n_layers = inference.compute_embeddings(m, sequences, layer_name, pooling)
    base = {"checkpoint": m.repo_id, "layer_name": layer_idx, "num_layers": n_layers, "pooling": pooling}
    if is_batch:
        return {**base, "sequences": sequences, "embeddings": [e.tolist() for e in embeddings]}
    return {**base, "sequence": sequences[0], "embedding": embeddings[0].tolist()}


@mcp.tool()
@user_errors_as_tool_errors
def score_snp(
    sequence: str,
    alternative_allele: str,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    position: int | None = None,
) -> dict:
    """Zero-shot SNP effect: whole-sequence log-likelihood(mutated) - log-likelihood(reference).

    `position` defaults to the sequence center. Response shape matches every other
    service; `method` says how the score was computed.
    """
    m = resolve_model(checkpoint)
    return {"checkpoint": m.repo_id, **inference.score_variant(m, sequence, alternative_allele, position=position)}


@mcp.tool()
@user_errors_as_tool_errors
def compare_sequences(
    sequence_a: str,
    sequence_b: str,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    layer_name: str = "last",
) -> dict:
    """Cosine similarity between the mean-pooled embeddings of two DNA sequences."""
    m = resolve_model(checkpoint)
    similarity, layer_idx = inference.compare_sequences(m, sequence_a, sequence_b, layer_name)
    return {"checkpoint": m.repo_id, "layer_name": layer_idx, "cosine_similarity": similarity}


register_health_route(mcp)


def _build_http_app(host: str, path: str):
    """Build the streamable-HTTP ASGI app, wrapped in bearer auth if MCP_AUTH_TOKEN is set."""
    return build_http_app(mcp, host, path, auth_token_env="MCP_AUTH_TOKEN", logger=logger)


def main() -> None:
    """Entry point (stdio or HTTP via MCP_TRANSPORT)."""
    run_server(mcp, logger=logger)


if __name__ == "__main__":
    main()
