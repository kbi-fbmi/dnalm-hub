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

Post-trained checkpoints (`*-post*`) are conditioned on a species and add two
NTv3-only capabilities: genome annotation (`annotate_sequence`) and functional
track prediction (`predict_tracks`), with `list_species` / `list_tracks` to
discover valid inputs.
"""

import logging

from dnalm_common import user_errors_as_tool_errors
from dnalm_common.http_app import build_http_app, register_health_route, run_server
from mcp.server.mcpserver import MCPServer

from . import inference
from .registry import DEFAULT_MODEL_ALIAS, DEFAULT_POST_MODEL_ALIAS, MODEL_REGISTRY, resolve_model

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
        "models. Post-trained checkpoints (stage='post', e.g. '100m-post') are "
        "conditioned on a `species` (default 'human'; see list_species) and can also "
        "annotate genomic elements (annotate_sequence) and predict experimental signal "
        "tracks (predict_tracks); pre-trained ones are species-agnostic. NTv3 is a "
        "masked (bidirectional) DNA language model, not a generative one -- there is "
        "no sequence-generation tool here. NOTE: NTv3 model weights are distributed "
        "under a NON-COMMERCIAL license by InstaDeep and are gated on HuggingFace (see "
        "README); check the relevant model card before production or commercial use. "
        "This server's own code is MIT licensed."
    ),
)

@mcp.tool()
@user_errors_as_tool_errors
def list_available_checkpoints() -> list[dict]:
    """List NTv3 checkpoints available as the `checkpoint` argument of the other tools.

    Same shape as evo2-mcp's tool of the same name: a list of {"name", "description"}
    dicts (plus extra NTv3-specific metadata: repo_id, params, stage, context,
    pad_multiple). stage="pre": DNA-only masked LM; stage="post": additionally
    species-conditioned with annotation/track heads. Any other HuggingFace repo id
    implementing the NTv3 trust_remote_code architecture -- e.g. your own
    fine-tuned checkpoint -- also works as `checkpoint` even though it won't be listed here.
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
@user_errors_as_tool_errors
def get_model_info(checkpoint: str = DEFAULT_MODEL_ALIAS) -> dict:
    """Load an NTv3 checkpoint and report its architecture details.

    Downloads the model on first use (cached afterwards). Returns device/dtype,
    parameter count, vocab size, hidden size, number of hidden-state layers available
    for `embed_sequence`, and the raw model config. For post-trained checkpoints also
    the supported species and annotation element names.
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    return inference.model_info(repo_id, pad_multiple)


@mcp.tool()
@user_errors_as_tool_errors
def get_embedding_layers(checkpoint: str, which: str = "recommended") -> dict:
    """List hidden-state layer indices usable as `layer_name` in embed_sequence.

    Mirrors evo2-mcp's tool of the same name. NTv3 is U-Net shaped: only the first
    and last layers have one position per base; the middle ones are downsampled
    (`bases_per_position` says by how much). `which="recommended"` returns the final
    layer; `which="all"` lists every valid layer index.
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    result = inference.list_embedding_layers(repo_id, pad_multiple, which)
    return {"checkpoint": repo_id, **result}


@mcp.tool()
@user_errors_as_tool_errors
def embed_sequence(
    sequence: str | list[str],
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    layer_name: str = "last",
    pooling: str = "mean",
    species: str | None = None,
) -> dict:
    """Return NTv3 feature representations (embeddings) for one or more DNA sequences.

    Args:
        sequence: A single DNA sequence, or a list of sequences to embed as one batch
            (letters A/C/G/T/N only; lengths may differ). A single string in, a single
            embedding out -- matching evo2-mcp's `embed_sequence` call shape.
        checkpoint: A name from list_available_checkpoints, or a full HuggingFace repo id.
        layer_name: Hidden-state layer to read, as an integer string (e.g. "6") or
            "last". See get_embedding_layers for valid values.
        pooling: "mean" averages over the real (non-padding) positions into one vector
            per sequence (default). "per_token" returns the per-position matrix
            (one row per base at the full-resolution layers), trimmed to the
            sequence's real length -- can be large for long sequences.
        species: Organism the DNA comes from -- only for post-trained checkpoints
            (default 'human'; see list_species). Must be omitted for pre-trained ones.
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    is_batch = isinstance(sequence, list)
    sequences = sequence if is_batch else [sequence]
    embeddings, layer_idx, n_layers = inference.compute_embeddings(
        repo_id, sequences, layer_name, pooling, pad_multiple, species=species
    )
    base = {
        "checkpoint": repo_id,
        "layer_name": layer_idx,
        "num_layers": n_layers,
        "pooling": pooling,
    }
    if species:
        base["species"] = species.strip().lower()
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
    species: str | None = None,
) -> dict:
    """Zero-shot effect score for a single-nucleotide substitution.

    Mirrors evo2-mcp's `score_snp`: pass a reference `sequence` and the
    `alternative_allele` to test. `position` defaults to the sequence center (same
    default evo2-mcp always uses); pass it explicitly to score any other site.
    `species` applies to post-trained checkpoints only (default 'human').

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
    result = inference.score_variant(
        repo_id, sequence, alternative_allele, pad_multiple, position=position, species=species
    )
    response = {
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
    if result["species"]:
        response["species"] = result["species"]
    return response


@mcp.tool()
@user_errors_as_tool_errors
def predict_masked_positions(
    sequence: str,
    positions: list[int] | None = None,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    top_k: int = 4,
    species: str | None = None,
) -> dict:
    """Predict nucleotide probabilities at masked positions in a DNA sequence.

    NTv3-specific: has no evo2-mcp equivalent (evo2 is generative and would instead
    use `generate_sequence`, which has no NTv3 analog since NTv3 cannot generate
    novel sequence). If `positions` is omitted, every 'N' character in `sequence`
    is treated as a position to fill in. Returns the top_k most likely nucleotides
    (softmax probabilities restricted to A/C/G/T/N) for each requested position.
    `species` applies to post-trained checkpoints only (default 'human').
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    results = inference.predict_masked(repo_id, sequence, positions or [], top_k, pad_multiple, species=species)
    return {"checkpoint": repo_id, "predictions": results}


@mcp.tool()
@user_errors_as_tool_errors
def compare_sequences(
    sequence_a: str,
    sequence_b: str,
    checkpoint: str = DEFAULT_MODEL_ALIAS,
    layer_name: str = "last",
    species: str | None = None,
) -> dict:
    """Compute cosine similarity between the mean-pooled NTv3 embeddings of two DNA sequences.

    NTv3-specific: has no evo2-mcp equivalent. `species` applies to post-trained
    checkpoints only (default 'human').
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    similarity, layer_idx = inference.compare_sequences(
        repo_id, sequence_a, sequence_b, layer_name, pad_multiple, species=species
    )
    return {"checkpoint": repo_id, "layer_name": layer_idx, "cosine_similarity": similarity}


@mcp.tool()
@user_errors_as_tool_errors
def list_species(checkpoint: str = DEFAULT_POST_MODEL_ALIAS) -> dict:
    """List the species a post-trained checkpoint is conditioned on, and what it can predict.

    Returns every valid `species` value with its number of predictable experimental
    tracks (0 = annotation/embeddings only, no predict_tracks), plus the genomic
    element names annotate_sequence reports. Reads only the model config (fast).
    """
    repo_id, _ = resolve_model(checkpoint)
    return {"checkpoint": repo_id, **inference.species_overview(repo_id)}


@mcp.tool()
@user_errors_as_tool_errors
def list_tracks(
    species: str = inference.DEFAULT_SPECIES,
    checkpoint: str = DEFAULT_POST_MODEL_ALIAS,
    contains: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """List the experimental track ids predict_tracks accepts for a species (paginated).

    The model ships no track descriptions, only ids; the prefix tells the source,
    where assay type, tissue and cell type are documented:
    `ENCSR...` ENCODE experiments (encodeproject.org; most human/mouse tracks),
    `CNhs...` FANTOM5 CAGE libraries, `GSM...` GEO samples, `GTEX-...` GTEx
    RNA-seq samples, `SRX/ERX/DRX...` SRA/ENA experiments (plants); a few groups
    (`kai*`, `yangli*`, fly `window_*`) are dataset-internal names.
    `contains` filters by substring (e.g. contains="ENCSR" or "CNhs").
    """
    repo_id, _ = resolve_model(checkpoint)
    return {"checkpoint": repo_id, **inference.list_track_ids(repo_id, species, contains, limit, offset)}


@mcp.tool()
@user_errors_as_tool_errors
def annotate_sequence(
    sequence: str,
    checkpoint: str = DEFAULT_POST_MODEL_ALIAS,
    species: str = inference.DEFAULT_SPECIES,
    elements: list[str] | None = None,
    bin_size: int = 1,
) -> dict:
    """Predict genomic elements along a DNA sequence (post-trained checkpoints only).

    Returns, per position, the probability that it belongs to each element:
    protein_coding_gene, lncRNA, exon, intron, splice_donor/acceptor, promoter and
    enhancer (tissue-specific/-invariant), 5'/3' UTR (+/- strand), start/stop codon,
    CTCF-bound, polyA_signal, ORF, ... (see list_species for the full list).

    Only the CENTRAL 37.5% of the input window gets predictions (the rest is
    context): `region_start`/`region_end` give the covered positions. Use long
    windows (tens of kb, ideally a multiple of 128 bp) centred on the region of
    interest. `elements` restricts the output; `bin_size` averages over bins of
    that many bases to keep the response small.
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    result = inference.annotate(repo_id, sequence, pad_multiple, species=species, elements=elements, bin_size=bin_size)
    return {"checkpoint": repo_id, **result}


@mcp.tool()
@user_errors_as_tool_errors
def predict_tracks(
    sequence: str,
    track_ids: list[str],
    checkpoint: str = DEFAULT_POST_MODEL_ALIAS,
    species: str = inference.DEFAULT_SPECIES,
    bin_size: int = 1,
) -> dict:
    """Predict experimental signal (RNA-seq, ChIP-seq, ATAC/DNase, ...) for selected tracks.

    Post-trained checkpoints only. `track_ids` (max 64) come from list_tracks for
    the same species; only species with tracks work (human, mouse, fly, plants;
    see list_species). Values are the model's predicted coverage (non-negative,
    model output scale, not calibrated to a specific experiment's read depth).
    As in annotate_sequence, only the central 37.5% of the window
    (`region_start`..`region_end`) is predicted; `bin_size` averages bins.
    """
    repo_id, pad_multiple = resolve_model(checkpoint)
    result = inference.predict_tracks(repo_id, sequence, track_ids, pad_multiple, species=species, bin_size=bin_size)
    return {"checkpoint": repo_id, **result}


register_health_route(mcp)


def _build_http_app(host: str, path: str):
    """Build the streamable-HTTP ASGI app, wrapped in bearer auth if MCP_AUTH_TOKEN is set."""
    return build_http_app(mcp, host, path, auth_token_env="MCP_AUTH_TOKEN", logger=logger)


def main() -> None:
    """Entry point used by `uv run ntv3-mcp` and the Docker image's CMD.

    MCP_TRANSPORT selects the transport:
      - "stdio" (default): standard transport for local desktop MCP clients.
      - "http": streamable-HTTP transport for server/container deployments,
        exposing the MCP endpoint at http://MCP_HOST:MCP_PORT/MCP_PATH plus an
        unauthenticated GET /health for liveness/readiness checks.
    """
    run_server(mcp, logger=logger)


if __name__ == "__main__":
    main()
