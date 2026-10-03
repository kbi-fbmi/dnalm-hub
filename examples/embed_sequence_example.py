"""Example: send a DNA sequence to ntv3-mcp and print embedding metadata.

Usage (from the repo root; uses the ntv3 service's environment for `ntv3_mcp_client`):
    uv run --project services/ntv3 python examples/embed_sequence_example.py \
        --server http://kbi-cs2.fbmi.cvut.cz:8000 --sequence ACGTACGTACGT

Optional auth token:
    set MCP_AUTH_TOKEN=...   (Windows)
    export MCP_AUTH_TOKEN=... (Linux/macOS)
"""

from __future__ import annotations

import argparse
import os

from ntv3_mcp_client import Ntv3McpClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Call ntv3-mcp embed_sequence over HTTP MCP.")
    parser.add_argument("--server", default="http://kbi-cs2.fbmi.cvut.cz:8000", help="Base server URL")
    parser.add_argument("--sequence", default="ACGTACGTACGT", help="DNA sequence (A/C/G/T/N)")
    parser.add_argument("--checkpoint", default="100m-pre", help="Checkpoint alias or full HF repo id")
    parser.add_argument("--token", default=os.getenv("MCP_AUTH_TOKEN"), help="Bearer token for /mcp")
    args = parser.parse_args()

    with Ntv3McpClient(base_url=args.server, auth_token=args.token) as client:
        print(f"health: {client.health().strip()}")

        result = client.embed_sequence(sequence=args.sequence, checkpoint=args.checkpoint)

        embedding = result.get("embedding", [])
        print(f"checkpoint used: {result.get('checkpoint')}")
        print(f"pooling: {result.get('pooling')}")
        print(f"embedding length: {len(embedding)}")
        print(f"embedding preview (first 8): {embedding[:8]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
