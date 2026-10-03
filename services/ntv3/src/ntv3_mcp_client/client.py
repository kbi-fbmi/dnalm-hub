"""Python client for an ntv3-mcp HTTP deployment.

Thin subclass of `dnalm_client.GenericMcpClient` -- all session
management, SSE parsing, and tool-wrapper methods live there since every
dnalm-hub service (ntv3-mcp and siblings) exposes the same tool names and
shapes. This subclass exists for a family-specific name/docstring.

Example:
    >>> from ntv3_mcp_client import Ntv3McpClient
    >>> with Ntv3McpClient("http://kbi-cs2.fbmi.cvut.cz:8000", auth_token="your-token") as client:
    ...     client.health()
    ...     result = client.embed_sequence("ACGTACGTACGT", checkpoint="100m-pre")
    ...     sorted(result.keys())
    'ok'
    ['checkpoint', 'embedding', 'layer_name', 'num_layers', 'pooling', 'sequence']
"""

from __future__ import annotations

from typing import Any

from dnalm_client import GenericMcpClient


class Ntv3McpClient(GenericMcpClient):
    """Client for an ntv3-mcp server. See `GenericMcpClient` for all available methods.

    Supports every tool ntv3-mcp registers, including `predict_masked_positions`
    (NTv3 uses a single-nucleotide tokenizer, so single-position masking is valid)
    -- there is no `generate_sequence`, since NTv3 is a masked, not generative, model.

    The methods below exist only on ntv3-mcp and need a post-trained checkpoint
    (default "100m-post"); pass `species` (default "human") to the shared methods
    too when using one.
    """

    def list_species(self, checkpoint: str = "100m-post") -> dict[str, Any]:
        """Species the post-trained model accepts, tracks per species, and annotation element names."""
        return self._call_structured("list_species", {"checkpoint": checkpoint})

    def list_tracks(
        self,
        species: str = "human",
        checkpoint: str = "100m-post",
        contains: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Track ids (ENCODE ENCSR..., FANTOM5 CNhs..., GEO GSM..., GTEx, SRA ...) usable in `predict_tracks`, paginated."""
        arguments: dict[str, Any] = {"species": species, "checkpoint": checkpoint, "limit": limit, "offset": offset}
        if contains:
            arguments["contains"] = contains
        return self._call_structured("list_tracks", arguments)

    def annotate_sequence(
        self,
        sequence: str,
        checkpoint: str = "100m-post",
        species: str = "human",
        elements: list[str] | None = None,
        bin_size: int = 1,
    ) -> dict[str, Any]:
        """Per-position probabilities of genomic elements (exon, intron, promoter, ...) in the central window."""
        arguments: dict[str, Any] = {"sequence": sequence, "checkpoint": checkpoint, "species": species, "bin_size": bin_size}
        if elements:
            arguments["elements"] = elements
        return self._call_structured("annotate_sequence", arguments)

    def predict_tracks(
        self,
        sequence: str,
        track_ids: list[str],
        checkpoint: str = "100m-post",
        species: str = "human",
        bin_size: int = 1,
    ) -> dict[str, Any]:
        """Predicted experimental signal for the given track ids in the central window."""
        return self._call_structured(
            "predict_tracks",
            {"sequence": sequence, "track_ids": track_ids, "checkpoint": checkpoint, "species": species, "bin_size": bin_size},
        )
