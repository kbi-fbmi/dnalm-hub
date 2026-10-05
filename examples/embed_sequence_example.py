"""Example: embed a DNA sequence with any dnalm-hub service and print a summary.

Usage (from the repo root):
    uv run --project packages/dnalm-client python examples/embed_sequence_example.py \
        --server http://localhost:8001 --sequence ACGTACGTACGT --checkpoint 50m

Without --checkpoint the service's default model is used. The token is read from
MCP_AUTH_TOKEN (or pass --token).
"""

from __future__ import annotations

import argparse
import os

from dnalm_client import GenericMcpClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Call embed_sequence on a dnalm-hub service.")
    parser.add_argument(
        "--server", default="http://localhost:8000", help="service URL, e.g. :8001 for ntv2"
    )
    parser.add_argument("--sequence", default="ACGTACGTACGT", help="DNA sequence (A/C/G/T/N)")
    parser.add_argument("--checkpoint", help="checkpoint alias or HF repo id (default: server's)")
    parser.add_argument("--token", default=os.getenv("MCP_AUTH_TOKEN"), help="bearer token")
    args = parser.parse_args()

    with GenericMcpClient(base_url=args.server, auth_token=args.token) as client:
        print(f"health: {client.health().strip()}")
        result = client.embed_sequence(args.sequence, checkpoint=args.checkpoint)

    embedding = result["embedding"]
    print(f"checkpoint: {result['checkpoint']}")
    print(f"layer: {result['layer_name']} of {result['num_layers']}")
    print(f"dimension: {len(embedding)}")
    print(f"first values: {[round(x, 4) for x in embedding[:5]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
