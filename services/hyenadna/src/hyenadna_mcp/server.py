"""MCP server exposing HyenaDNA (HazyResearch / LongSafari) as tools.

Run directly for local testing:

    uv run hyenadna-mcp

Same tool names, `checkpoint` parameter, and response shapes as the other
services (see docs/adding-a-model.md). HyenaDNA is a causal LM with one token per
base, so it adds `generate_sequence`; it has no `predict_masked_positions` (it
was never trained to fill in masks), and `score_snp` uses the exact causal
log-likelihood of the whole sequence (`method="whole_sequence_log_prob_delta"`).
"""

import logging
from importlib.metadata import version

from dnalm_common import user_errors_as_tool_errors
from dnalm_common.http_app import build_http_app, register_health_route, run_server
from mcp.server.mcpserver import MCPServer

from . import inference
from .registry import DEFAULT_MODEL_ALIAS, MODEL_REGISTRY, resolve_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hyenadna-mcp")

mcp = MCPServer(
    "hyenadna-mcp",
    version=version("hyenadna-mcp"),
    instructions=(
        "Tools for running HyenaDNA genomic language models (HazyResearch, pretrained on the "
        "human reference genome hg38) over raw DNA sequences: embeddings, embedding "
        "similarity, zero-shot single-nucleotide variant scoring, and sequence generation. "
        "Sequences are plain strings using A/C/G/T/N only (case-insensitive). Call "
        "list_available_checkpoints first. HyenaDNA is a CAUSAL (left-to-right) model with "
        "one token per base and long context: from 1,026 bases (tiny-1k) up to ~1M bases "
        "(large-1m); longer sequences are rejected, not truncated. Because it is causal, "
        "the embedding at base t only reflects bases 0..t, and there is no masked-position "
        "prediction tool. Checkpoints are licensed BSD-3-Clause (see the model cards); this "
        "server's own code is MIT licensed."
    ),
)


@mcp.tool()
@user_errors_as_tool_errors
def list_available_checkpoints() -> list[dict]:
    """List HyenaDNA checkpoints available as the `checkpoint` argument of the other tools.

    Any other HuggingFace repo id loadable with AutoModelForCausalLM and the HyenaDNA
    character tokenizer also works as `checkpoint`, even though it won't be listed here.
    """
    return [
        {
            "name": alias,
            "description": spec.notes
            or f"HyenaDNA {spec.params} ({spec.num_layers} layers, width {spec.hidden_size}), "
            f"{spec.max_tokens}-base context.",
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
    """Load a HyenaDNA checkpoint and report its architecture details.

    Downloads the model on first use. Returns device/dtype, parameter count, vocab
    size, hidden size, number of hidden-state layers available for `embed_sequence`,
    the context limit in tokens (= bases), any lower server cap (`max_bases`), and
    the raw model config.
    """
    return inference.model_info(resolve_model(checkpoint))


@mcp.tool()
@user_errors_as_tool_errors
def get_embedding_layers(checkpoint: str = DEFAULT_MODEL_ALIAS, which: str = "recommended") -> dict:
    """List hidden-state layer indices usable as `layer_name` in embed_sequence.

    0 is the token embedding, 1..n the output of each Hyena block, n+1 ("last") the
    final LayerNorm output. `which="recommended"` is a heuristic (the last two);
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
) -> dict:
    """Return HyenaDNA embeddings for one or more DNA sequences.

    Args:
        sequence: A single DNA sequence, or a list of sequences to embed as a batch
            (letters A/C/G/T/N only; lengths may differ). Each must fit the
            checkpoint's context limit. Batch results equal single-sequence results.
        checkpoint: A name from list_available_checkpoints, or a full HuggingFace repo id.
        layer_name: Hidden-state layer to read, as an integer string (e.g. "4") or
            "last". See get_embedding_layers for valid values.
        pooling: "mean" (default) averages over all base positions into one vector per
            sequence. HyenaDNA is causal, so position t only summarizes bases 0..t: the
            mean is an average of prefix summaries (weighted toward the start), and only
            the final row of "per_token" has seen the whole sequence. "per_token" returns
            one row per base, in order (no special tokens).
    """
    m = resolve_model(checkpoint)
    is_batch = isinstance(sequence, list)
    sequences = sequence if is_batch else [sequence]
    embeddings, layer_idx, n_layers = inference.compute_embeddings(
        m, sequences, layer_name, pooling
    )
    base = {
        "checkpoint": m.repo_id,
        "layer_name": layer_idx,
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
    """Zero-shot effect score for a single-nucleotide substitution.

    Same call and response shape as the other services; `position` defaults to the
    sequence center. `method="whole_sequence_log_prob_delta"`: the reference and
    mutated sequences each get their EXACT causal log-likelihood,
    sum_t log P(base_t | bases_<t) (natural log; the first base has no context and is
    not scored), from one forward pass each. `score_delta` = mutated - original;
    negative means HyenaDNA finds the mutated sequence less likely. Bases after the
    variant contribute too, so longer downstream context changes the score. A rough,
    unvalidated heuristic -- not a clinical or functional prediction. Not numerically
    comparable with other models' scores.
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
) -> dict:
    """Compute cosine similarity between the mean-pooled HyenaDNA embeddings of two DNA sequences."""
    m = resolve_model(checkpoint)
    similarity, layer_idx = inference.compare_sequences(m, sequence_a, sequence_b, layer_name)
    return {"checkpoint": m.repo_id, "layer_name": layer_idx, "cosine_similarity": similarity}


@mcp.tool()
@user_errors_as_tool_errors
def generate_sequence(
    prompt: str,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 4,
    seed: int | None = None,
) -> dict:
    """Autoregressively extend a DNA prompt with HyenaDNA (one token = one base).

    Args:
        prompt: Non-empty DNA prefix (A/C/G/T/N).
        checkpoint: A name from list_available_checkpoints, or a full HuggingFace repo id.
        max_new_tokens: Bases to generate (1..HYENADNA_MAX_NEW_TOKENS, default cap 2048).
            prompt + new bases must fit the context limit.
        temperature: Sampling temperature; 0 = greedy (argmax).
        top_k: Keep only the k most likely bases before sampling; 0 or 4 = no filtering.
        seed: Optional RNG seed for reproducible sampling.

    Only A/C/G/T are sampled (N and special tokens are excluded). There is no cache, so
    each new base costs one forward pass over the whole prefix.
    """
    m = resolve_model(checkpoint)
    result = inference.generate(m, prompt, max_new_tokens, temperature, top_k, seed=seed)
    return {"checkpoint": m.repo_id, **result}


register_health_route(mcp)


def _build_http_app(host: str, path: str):
    """Build the streamable-HTTP ASGI app, wrapped in bearer auth if MCP_AUTH_TOKEN is set."""
    return build_http_app(mcp, host, path, auth_token_env="MCP_AUTH_TOKEN", logger=logger)


def main() -> None:
    """Entry point used by `uv run hyenadna-mcp` and the Docker image (stdio or HTTP via MCP_TRANSPORT)."""
    run_server(mcp, logger=logger)


if __name__ == "__main__":
    main()
