"""MCP server exposing GenerTeam's GENERator causal DNA language models as tools.

Run directly for local testing:

    uv run generator-mcp

Same tool names, `checkpoint` parameter and response shapes as the other services
(see docs/adding-a-model.md), plus `generate_sequence` (causal LM). No
`predict_masked_positions`: GENERator is causal and reads 6-mers, so masking one
base is not defined. Model-specific extras: `remainder` (6-mer policy) on the
embedding tools and `generate_sequence`, `pooling="last_token"`, and `sampling`
/ `seed` on `generate_sequence`.
"""

import logging

from dnalm_common import user_errors_as_tool_errors
from dnalm_common.http_app import build_http_app, register_health_route, run_server
from mcp.server.mcpserver import MCPServer

from . import inference
from .registry import DEFAULT_MODEL_ALIAS, MODEL_REGISTRY, resolve_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generator-mcp")

mcp = MCPServer(
    "generator-mcp",
    instructions=(
        "Tools for running GenerTeam's GENERator models -- Llama-style causal (autoregressive) DNA "
        "language models -- over raw DNA: embeddings, embedding similarity, zero-shot SNP scoring "
        "by exact log-likelihood, and sequence generation. Sequences are plain strings using "
        "A/C/G/T/N only (case-insensitive). Call list_available_checkpoints first; the default is "
        "GENERator-v2 eukaryote 1.2B. GENERator reads DNA as non-overlapping 6-mers (a 6-mer "
        "containing N becomes one unknown token) with a 16384-token (~98 kb) context; longer "
        "inputs are rejected, never truncated. embed_sequence, compare_sequences and "
        "generate_sequence need a length that is a multiple of 6: by default other lengths are "
        "rejected; pass remainder='trim_left' (the authors' recipe: drop the first len%6 bases) or "
        "remainder='pad_left' (prepend 'A's) to adjust explicitly. score_snp accepts any length. "
        "per_token embeddings and generate_sequence's max_new_tokens count 6-mer TOKENS, not bases. "
        "License: the GENERator weights are MIT licensed (per the HuggingFace model cards), as is "
        "this server's code."
    ),
)


@mcp.tool()
@user_errors_as_tool_errors
def list_available_checkpoints() -> list[dict]:
    """List GENERator checkpoints available as the `checkpoint` argument of the other tools.

    Any other HuggingFace repo id with Llama weights and GENERator's 6-mer vocab.txt
    also works as `checkpoint`, even though it won't be listed here.
    """
    return [
        {
            "name": alias,
            "description": spec.notes
            or f"GENERator {spec.generation}, {spec.domain} DNA, {spec.params} parameters, ~98 kb context.",
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "params": spec.params,
            "generation": spec.generation,
            "domain": spec.domain,
            "max_tokens": spec.max_tokens,
            "context": f"{spec.max_tokens} tokens (~{(spec.max_tokens - 2) * 6 // 1000} kb)",
            "default": alias == DEFAULT_MODEL_ALIAS,
        }
        for alias, spec in MODEL_REGISTRY.items()
    ]


@mcp.tool()
@user_errors_as_tool_errors
def get_model_info(checkpoint: str = DEFAULT_MODEL_ALIAS) -> dict:
    """Load a GENERator checkpoint and report its architecture details.

    Downloads the model on first use. Returns device/dtype, parameter count, vocab
    size, hidden size, number of hidden-state layers available for `embed_sequence`,
    the context limit in tokens and bases, and the raw model config.
    """
    return inference.model_info(resolve_model(checkpoint))


@mcp.tool()
@user_errors_as_tool_errors
def get_embedding_layers(checkpoint: str, which: str = "recommended") -> dict:
    """List hidden-state layer indices usable as `layer_name` in embed_sequence.

    `which="recommended"` is a heuristic (final layer and the one before it);
    `which="all"` lists every valid layer index.
    """
    m = resolve_model(checkpoint)
    return {"checkpoint": m.repo_id, **inference.list_embedding_layers(m, which)}


@mcp.tool()
@user_errors_as_tool_errors
def embed_sequence(
    sequence: str | list[str],
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    layer_name: str = "last",
    pooling: str = "mean",
    remainder: str = "reject",
) -> dict:
    """Return GENERator embeddings for one or more DNA sequences.

    Args:
        sequence: A single DNA sequence, or a list of sequences to embed as a batch
            (letters A/C/G/T/N only). Each must fit the 16384-token (~98 kb) context.
        checkpoint: A name from list_available_checkpoints, or a full HuggingFace repo id.
        layer_name: Hidden-state layer to read, as an integer string (e.g. "13") or
            "last". See get_embedding_layers for valid values.
        pooling: The model is fed "<s>" + 6-mers, plus a trailing "<s>" separator for
            v2 checkpoints, as in the GENERator model cards. "mean" (default) averages
            every token including the <s> markers (the cards' mean-pooling option;
            captures species-level signal); "last_token" is the final token (v2: the
            separator, v1: the last 6-mer); "per_token" returns one row per TOKEN --
            leading <s>, one row per 6-mer, then the separator for v2 -- not one row
            per base.
        remainder: What to do when a length is not a multiple of 6: "reject"
            (default, error), "trim_left" (drop the first len%6 bases, as the authors
            do) or "pad_left" (prepend 'A's). Adjusted counts are returned in
            `remainder_bases`.
    """
    m = resolve_model(checkpoint)
    is_batch = isinstance(sequence, list)
    sequences = sequence if is_batch else [sequence]
    embeddings, layer_idx, n_layers, adjusted = inference.compute_embeddings(
        m, sequences, layer_name, pooling, remainder
    )
    base = {
        "checkpoint": m.repo_id,
        "layer_name": layer_idx,
        "num_layers": n_layers,
        "pooling": pooling,
        "remainder": remainder,
    }
    if is_batch:
        return {
            **base,
            "remainder_bases": adjusted,
            "sequences": sequences,
            "embeddings": [e.tolist() for e in embeddings],
        }
    return {
        **base,
        "remainder_bases": adjusted[0],
        "sequence": sequences[0],
        "embedding": embeddings[0].tolist(),
    }


@mcp.tool()
@user_errors_as_tool_errors
def score_snp(
    sequence: str,
    alternative_allele: str,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    position: int | None = None,
) -> dict:
    """Zero-shot effect score for a single-nucleotide substitution.

    Same call and response shape as the other services; `position` defaults to the
    sequence center. `method="whole_sequence_log_prob_delta"`: the reference and
    mutated sequences each get their exact causal log-likelihood log P(seq | <s>)
    (natural log, summed over the 6-mer tokens; N-containing and partial trailing
    6-mers are marginalized exactly, so any length works). `score_delta` = mutated
    - original; negative means GENERator finds the mutated sequence less likely.
    Costs two forward passes. A rough, unvalidated heuristic -- not a clinical or
    functional prediction; not numerically comparable across services.
    """
    m = resolve_model(checkpoint)
    result = inference.score_variant(m, sequence, alternative_allele, position=position)
    return {"checkpoint": m.repo_id, **result}


@mcp.tool()
@user_errors_as_tool_errors
def compare_sequences(
    sequence_a: str,
    sequence_b: str,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    layer_name: str = "last",
    remainder: str = "reject",
) -> dict:
    """Cosine similarity between the mean-pooled GENERator embeddings of two DNA sequences.

    `remainder` is the same 6-mer policy as in embed_sequence.
    """
    m = resolve_model(checkpoint)
    similarity, layer_idx = inference.compare_sequences(
        m, sequence_a, sequence_b, layer_name, remainder
    )
    return {"checkpoint": m.repo_id, "layer_name": layer_idx, "cosine_similarity": similarity}


@mcp.tool()
@user_errors_as_tool_errors
def generate_sequence(
    prompt: str,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    sampling: str = "base",
    remainder: str = "reject",
    seed: int | None = None,
) -> dict:
    """Autoregressively continue a DNA prompt; the output is returned as bases.

    Args:
        prompt: DNA to continue (A/C/G/T/N). Its length must be a multiple of 6
            unless `remainder` says otherwise (the authors warn that a ragged end
            produces junk such as repeated AAAAAA).
        checkpoint: A name from list_available_checkpoints, or a full HuggingFace repo id.
        max_new_tokens: Number of new 6-mer TOKENS; the continuation has
            6 x max_new_tokens bases.
        temperature: Softmax temperature; 0 = greedy.
        top_k: Keep only the k most likely 6-mers (of 4096) before sampling; 0 (default)
            = no filter. Note that small values such as 4 are very restrictive here.
        sampling: "base" (default, as in GenerTeam's generate()): marginalize the 6-mer
            distribution to per-base A/C/G/T probabilities and choose each of the 6
            bases from its marginal. "token": sample whole 6-mers (standard ancestral
            sampling). Special tokens are never generated in either mode.
        remainder: 6-mer policy for the prompt: "reject" (default), "trim_left" or
            "pad_left" (prepend 'A's, the authors' first suggestion).
        seed: Optional RNG seed for reproducible sampling.

    Returns `generated_sequence` (new bases only), `full_sequence` (prompt as used +
    continuation), `num_new_tokens` and `num_new_bases`.
    """
    m = resolve_model(checkpoint)
    result = inference.generate(
        m, prompt, max_new_tokens, temperature, top_k, sampling, remainder, seed
    )
    return {
        "checkpoint": m.repo_id,
        **result,
        "temperature": temperature,
        "top_k": top_k,
        "sampling": sampling,
        "seed": seed,
    }


register_health_route(mcp)


def _build_http_app(host: str, path: str):
    """Build the streamable-HTTP ASGI app, wrapped in bearer auth if MCP_AUTH_TOKEN is set."""
    return build_http_app(mcp, host, path, auth_token_env="MCP_AUTH_TOKEN", logger=logger)


def main() -> None:
    """Entry point used by `uv run generator-mcp` and the Docker image (stdio or HTTP via MCP_TRANSPORT)."""
    run_server(mcp, logger=logger)


if __name__ == "__main__":
    main()
