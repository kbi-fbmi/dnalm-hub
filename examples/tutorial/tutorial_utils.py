"""Shared setup for the tutorial notebooks: connect to the services, test sequences, tables.

Settings come from environment variables:
    MCP_HOST        server address, default http://localhost
    MCP_AUTH_TOKEN  shared bearer token (needed when the server sets one)
"""

from __future__ import annotations

import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dnalm_client import GenericMcpClient
except ImportError:  # not running in the repo's dnalm-client environment: use the source tree
    _repo = next(p for p in Path(__file__).resolve().parents if (p / "packages").is_dir())
    sys.path.insert(0, str(_repo / "packages" / "dnalm-client" / "src"))
    from dnalm_client import GenericMcpClient

HOST = os.environ.get("MCP_HOST", "http://localhost").rstrip("/")
TOKEN = os.environ.get("MCP_AUTH_TOKEN")
TIMEOUT_S = 900  # the first call to a model downloads it and loads it onto the GPU

PORTS = {
    "ntv3": 8000,
    "ntv2": 8001,
    "dnabert2": 8002,
    "hyenadna": 8003,
    "evo2": 8004,
    "generator": 8005,
    "genalm": 8006,
    "grover": 8007,
}
GATEWAY = f"{HOST}:8080"

# First 90 bp of the human HBB coding sequence (beta-globin, exon 1). 90 is a multiple of 6,
# which GENERator's 6-mer tokenizer requires by default.
HBB = "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGC"
# Sickle-cell variant: codon 6 GAG -> GTG (Glu -> Val), A -> T at 0-based position 19.
SNP_POS, SNP_ALT = 19, "T"
HBB_SICKLE = HBB[:SNP_POS] + SNP_ALT + HBB[SNP_POS + 1 :]
RANDOM_SEQ = "".join(random.Random(42).choice("ACGT") for _ in HBB)


def connect(
    services: list[str] | None = None, *, quiet: bool = False
) -> dict[str, GenericMcpClient]:
    """Clients for the services that answer /health, keyed by service name."""
    clients = {}
    for name in services or list(PORTS):
        client = GenericMcpClient(f"{HOST}:{PORTS[name]}", auth_token=TOKEN, timeout=TIMEOUT_S)
        try:
            client.health()
        except (urllib.error.URLError, OSError) as exc:
            if not quiet:
                print(f"{name}: not reachable ({exc}), skipped")
            continue
        clients[name] = client
    if not quiet:
        print(f"server {HOST} | token set: {bool(TOKEN)} | up: {', '.join(clients) or 'nothing'}")
    return clients


def default_checkpoints(clients: dict[str, GenericMcpClient]) -> dict[str, str]:
    """The default checkpoint of each service."""
    return {
        name: next(c["name"] for c in client.list_available_checkpoints() if c.get("default"))
        for name, client in clients.items()
    }


def gateway(path: str, payload: dict | None = None) -> dict:
    """GET (no payload) or POST JSON to the OpenAI-compatible gateway."""
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(GATEWAY + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.load(resp)


def fetch_hg38(chrom: str, start: int, end: int) -> str:
    """Reference sequence from the UCSC API (0-based, end-exclusive), uppercase."""
    url = (
        "https://api.genome.ucsc.edu/getData/sequence"
        f"?genome=hg38;chrom={chrom};start={start};end={end}"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)["dna"].upper()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0


def show_table(rows: list[dict], columns: list[str] | None = None) -> None:
    """Print a list of dicts as an aligned text table (floats with 4 decimals)."""
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))

    def cell(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if v is None else str(v)

    table = [[cell(r.get(c)) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in table)) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
