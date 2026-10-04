"""MCP server exposing Arc Institute's Evo 2 (community transformers port) as tools.

Uses the tool names, `checkpoint` parameter and response shapes shared by every
service in this repo (see docs/adding-a-model.md), so clients and the gateway
can treat all model families alike. Evo 2 is causal, so it gets
`generate_sequence` and scores SNPs by exact whole-sequence log-likelihood;
there is no `predict_masked_positions` (no mask token).
"""

import logging
from importlib.metadata import version

from dnalm_common import user_errors_as_tool_errors
from dnalm_common.http_app import build_http_app, register_health_route, run_server
from mcp.server.mcpserver import MCPServer

from . import inference
from .registry import DEFAULT_MODEL_ALIAS, MODEL_REGISTRY, resolve_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evo2-mcp")

mcp = MCPServer(
    "evo2-mcp",
    version=version("evo2-mcp"),
    instructions=(
        "Tools for running Arc Institute's Evo 2 genomic language model over raw DNA "
        "sequences: embeddings, zero-shot single-nucleotide variant scoring, and sequence "
        "generation. Evo 2 is a causal (autoregressive) StripedHyena-2 model trained on "
        "genomes from all domains of life, one token per base. Sequences use A/C/G/T/N only "
        "(case-insensitive). Call list_available_checkpoints first; '7b' (Evo 2 7B) is the "
        f"default. Inputs longer than the service limit ({inference.MAX_SEQ_LEN} bp "
        "by default, see get_model_info) are rejected, never truncated. Embeddings default to "
        "layer_name='recommended' (an intermediate layer, as in the Evo 2 paper), not 'last'. "
        "score_snp compares exact log-likelihoods of the reference and mutated sequence; "
        "negative = the mutation is less likely under the model. This runs a community "
        "pure-PyTorch port (Aquiles-ai/Evo2-transformers), not Arc's Vortex stack, so numbers "
        "can differ slightly from published ones; scores are research heuristics, not clinical "
        "predictions. LICENSE: Evo 2 weights and code are Apache-2.0 (Arc Institute); the port "
        "is Apache-2.0; this server's code is MIT."
    ),
)


@mcp.tool()
@user_errors_as_tool_errors
def list_available_checkpoints() -> list[dict]:
    """List Evo 2 checkpoints available as the `checkpoint` argument of the other tools.

    `max_tokens` is the checkpoint's native context; `max_sequence_length` is what
    this service accepts (capped by EVO2_MAX_SEQ_LEN). Any other repo id made with
    the port's converter also works as `checkpoint`.
    """
    return [
        {
            "name": alias,
            "description": spec.notes or f"Evo 2 {spec.params}.",
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "params": spec.params,
            "max_tokens": spec.max_tokens,
            "max_sequence_length": min(spec.max_tokens, inference.MAX_SEQ_LEN),
            "recommended_layer": spec.recommended_layers[0],
            "default": alias == DEFAULT_MODEL_ALIAS,
        }
        for alias, spec in MODEL_REGISTRY.items()
    ]


@mcp.tool()
@user_errors_as_tool_errors
def get_model_info(checkpoint: str = DEFAULT_MODEL_ALIAS) -> dict:
    """Load an Evo 2 checkpoint and report its architecture and limits.

    Downloads the model on first use (~14 GB for the 7B). Returns device/dtype,
    parameter count, hidden size, number of blocks, which blocks are attention
    (the rest are Hyena), native context and this service's `max_sequence_length`.
    """
    return inference.model_info(resolve_model(checkpoint))


@mcp.tool()
@user_errors_as_tool_errors
def get_embedding_layers(checkpoint: str = DEFAULT_MODEL_ALIAS, which: str = "recommended") -> dict:
    """List values usable as `layer_name` in embed_sequence / compare_sequences.

    which="recommended": the intermediate layer(s) the Evo 2 authors use for
    embeddings (7B: 'blocks.28.mlp.l3', then 'blocks.26.mlp.l3'). which="all":
    every hidden-state index (0 = token embeddings ... N = final layer) plus every
    'blocks.N.mlp.l3' (block N's MLP output, Arc's evo2 naming).
    """
    m = resolve_model(checkpoint)
    return {"checkpoint": m.repo_id, **inference.list_embedding_layers(m, which)}


@mcp.tool()
@user_errors_as_tool_errors
def embed_sequence(
    sequence: str | list[str],
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    layer_name: str = "recommended",
    pooling: str = "mean",
) -> dict:
    """Return Evo 2 embeddings for one DNA sequence, or for a list of sequences as one batch.

    Args:
        sequence: A DNA sequence, or a list of them (A/C/G/T/N; lengths may differ).
            A single string in, a single embedding out.
        checkpoint: A name from list_available_checkpoints, or a full HuggingFace repo id.
        layer_name: "recommended" (default: the intermediate layer the Evo 2 authors
            embed from, see get_embedding_layers), "last" (final layer), a hidden-state
            index such as "20", or an Evo 2 layer name such as "blocks.28.mlp.l3".
        pooling: "mean" (one vector per sequence) or "per_token" (one row per base).

    Evo 2 is causal: position t's embedding only depends on bases 0..t, so the
    mean-pooled vector weights the end of the sequence's context more.
    Each sequence runs separately, so a batch returns exactly the single results.
    """
    m = resolve_model(checkpoint)
    is_batch = isinstance(sequence, list)
    sequences = sequence if is_batch else [sequence]
    embeddings, layer_id, n_layers = inference.compute_embeddings(m, sequences, layer_name, pooling)
    base = {
        "checkpoint": m.repo_id,
        "layer_name": layer_id,
        "num_layers": n_layers,
        "pooling": pooling,
    }
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
    """Zero-shot SNP effect: exact log-likelihood(mutated) - log-likelihood(reference).

    This is Evo 2's own zero-shot variant-effect method: score both sequences with
    the causal model (sum of log P(base_t | bases before t) over bases 2..L) and
    take the difference. Negative = the alternative allele is less likely in this
    context (heuristically more disruptive). `position` defaults to the sequence
    center; use a few kb of context around the variant for best results. Arc's
    `score_sequences` reports the per-base mean instead of the sum; divide
    `score_delta` by (length - 1) to compare.
    """
    m = resolve_model(checkpoint)
    return {
        "checkpoint": m.repo_id,
        **inference.score_variant(m, sequence, alternative_allele, position=position),
    }


@mcp.tool()
@user_errors_as_tool_errors
def compare_sequences(
    sequence_a: str,
    sequence_b: str,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    layer_name: str = "recommended",
) -> dict:
    """Cosine similarity between the mean-pooled Evo 2 embeddings of two DNA sequences.

    `layer_name` as in embed_sequence (default "recommended").
    """
    m = resolve_model(checkpoint)
    similarity, layer_id = inference.compare_sequences(m, sequence_a, sequence_b, layer_name)
    return {"checkpoint": m.repo_id, "layer_name": layer_id, "cosine_similarity": similarity}


@mcp.tool()
@user_errors_as_tool_errors
def generate_sequence(
    prompt: str,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
) -> dict:
    """Autoregressively continue a DNA `prompt` with Evo 2 (one token = one base).

    Sampling is restricted to A/C/G/T. temperature=0 is greedy; top_k 1-4 keeps
    the k most likely bases (0, the default, = no filtering); `seed` makes sampling
    reproducible. max_new_tokens is capped (see get_model_info), and prompt + new
    bases must fit the length limit. Returns `prompt`, the new bases
    (`generated_sequence`) and their concatenation (`full_sequence`).
    """
    m = resolve_model(checkpoint)
    result = inference.generate(m, prompt, max_new_tokens, temperature, top_k, seed=seed)
    return {
        "checkpoint": m.repo_id,
        **result,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
    }


register_health_route(mcp)


def _build_http_app(host: str, path: str):
    """Build the streamable-HTTP ASGI app, wrapped in bearer auth if MCP_AUTH_TOKEN is set."""
    return build_http_app(mcp, host, path, auth_token_env="MCP_AUTH_TOKEN", logger=logger)


def main() -> None:
    """Entry point used by `uv run evo2-mcp` and the Docker image's CMD (stdio or HTTP via MCP_TRANSPORT)."""
    run_server(mcp, logger=logger)


if __name__ == "__main__":
    main()
