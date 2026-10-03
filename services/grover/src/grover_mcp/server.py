"""MCP server exposing PoetschLab's GROVER DNA language model as tools.

Run directly for local testing:

    uv run grover-mcp

Same tool names, `checkpoint` parameter, and response shapes as the other
services (see docs/adding-a-model.md), minus `predict_masked_positions`:
GROVER's BPE tokenizer means one sequence position is not one token, so
single-position masked prediction isn't well defined. `score_snp` uses the
tokenizer-agnostic whole-sequence method (reported in its `method` field).
No `generate_sequence`: GROVER is a masked (bidirectional) model.
"""

import logging

from dnalm_common import user_errors_as_tool_errors
from dnalm_common.http_app import build_http_app, register_health_route, run_server
from mcp.server.mcpserver import MCPServer

from . import inference
from .registry import DEFAULT_MODEL_ALIAS, MODEL_REGISTRY, resolve_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grover-mcp")

mcp = MCPServer(
    "grover-mcp",
    instructions=(
        "Tools for running GROVER (PoetschLab, TU Dresden; Sanabria et al., Nature Machine "
        "Intelligence 2024), a BERT masked DNA language model trained on the human genome, "
        "over raw DNA sequences: sequence embeddings, embedding similarity, and zero-shot "
        "single-nucleotide variant scoring. Sequences are plain strings using A/C/G/T/N only "
        "(case-insensitive); each 'N' becomes one unknown token. Call "
        "list_available_checkpoints first. GROVER tokenizes DNA with byte-pair encoding "
        "(tokens of 1-16 bases, ~3.5-4 on average) and has a hard 512-token context (~1.9 kb, "
        "depending on the sequence) -- longer sequences are rejected, not truncated. The "
        "model card notes that sequences shorter than ~50 bp tokenize differently from the "
        "genome GROVER was trained on, and suggests adding ~100 bp of genomic flank on both "
        "sides. GROVER is a masked (bidirectional) model: there is no sequence generation "
        "and no single-position masked prediction tool here. License: the HuggingFace model "
        "card states none; PoetschLab's Zenodo release of the same weights "
        "(doi:10.5281/zenodo.8373117) is CC-BY-4.0 (attribution required). This server's own "
        "code is MIT licensed."
    ),
)


@mcp.tool()
@user_errors_as_tool_errors
def list_available_checkpoints() -> list[dict]:
    """List GROVER checkpoints available as the `checkpoint` argument of the other tools.

    Any other HuggingFace repo id loadable with AutoModelForMaskedLM and a GROVER-style
    BPE tokenizer (e.g. a fine-tune) also works as `checkpoint`, even though it won't be
    listed here.
    """
    return [
        {
            "name": alias,
            "description": spec.notes or f"GROVER {spec.params}, {spec.max_tokens}-token context.",
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
    """Load a GROVER checkpoint and report its architecture details.

    Downloads the model on first use. Returns device/dtype, parameter count, vocab
    size, hidden size, number of hidden-state layers available for `embed_sequence`,
    the context limit in tokens, and the raw model config.
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
) -> dict:
    """Return GROVER embeddings for one or more DNA sequences.

    Args:
        sequence: A single DNA sequence, or a list of sequences to embed as a batch
            (letters A/C/G/T/N only). Each must fit the checkpoint's token limit.
        checkpoint: A name from list_available_checkpoints, or a full HuggingFace repo id.
        layer_name: Hidden-state layer to read, as an integer string (e.g. "6") or
            "last". See get_embedding_layers for valid values.
        pooling: "mean" (default) averages over all non-padding tokens ([CLS] and [SEP]
            included) into one vector per sequence. "per_token" returns one row per
            BPE TOKEN -- [CLS], one row per BPE token (1-16 bases each; one per 'N'),
            then [SEP] -- not one row per nucleotide as in ntv3-mcp.
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

    Same call and response shape as ntv3-mcp's `score_snp`; `position` defaults to
    the sequence center. Method `whole_sequence_log_prob_delta`: GROVER's BPE tokens
    don't map one-to-one onto bases (and a SNP can change the tokenization around
    it), so the reference and mutated sequences are tokenized independently and each
    gets a masked-LM pseudo-log-likelihood (mask every token in turn, sum the
    log-probability of the true token; 'N' tokens are skipped). `original_score`/
    `mutated_score` are those two sums and `score_delta` = mutated - original;
    negative means GROVER finds the mutated sequence less likely. Because the two
    sums can cover different token counts, the delta mixes "how likely" with "how it
    tokenizes". Costs ~2x (number of tokens) forward passes, batched. A rough,
    unvalidated heuristic -- not a clinical or functional prediction. Not numerically
    comparable to other services' scores.
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
    """Compute cosine similarity between the mean-pooled GROVER embeddings of two DNA sequences."""
    m = resolve_model(checkpoint)
    similarity, layer_idx = inference.compare_sequences(m, sequence_a, sequence_b, layer_name)
    return {"checkpoint": m.repo_id, "layer_name": layer_idx, "cosine_similarity": similarity}


register_health_route(mcp)


def _build_http_app(host: str, path: str):
    """Build the streamable-HTTP ASGI app, wrapped in bearer auth if MCP_AUTH_TOKEN is set."""
    return build_http_app(mcp, host, path, auth_token_env="MCP_AUTH_TOKEN", logger=logger)


def main() -> None:
    """Entry point used by `uv run grover-mcp` and the Docker image (stdio or HTTP via MCP_TRANSPORT)."""
    run_server(mcp, logger=logger)


if __name__ == "__main__":
    main()
