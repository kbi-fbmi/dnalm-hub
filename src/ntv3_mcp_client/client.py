"""Python client for an ntv3-mcp HTTP deployment.

Thin subclass of `genomic_mcp_client.GenericMcpClient` -- all session
management, SSE parsing, and tool-wrapper methods live there since every
genomic-mcp-* service (ntv3-mcp and siblings) exposes the same tool names and
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

from genomic_mcp_client import GenericMcpClient


class Ntv3McpClient(GenericMcpClient):
    """Client for an ntv3-mcp server. See `GenericMcpClient` for all available methods.

    Supports every tool ntv3-mcp registers, including `predict_masked_positions`
    (NTv3 uses a single-nucleotide tokenizer, so single-position masking is valid)
    -- there is no `generate_sequence`, since NTv3 is a masked, not generative, model.
    """
