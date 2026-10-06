"""Embedding throughput of a running dnalm-hub: one-by-one vs. concurrent vs. batched requests.

    uv run --project packages/dnalm-client python scripts/benchmark.py
    uv run --project packages/dnalm-client python scripts/benchmark.py ntv2 evo2 --json out.json

For each service's default checkpoint and each sequence length, embeds the same N random
sequences in several ways and reports sequences/s:

    single      one embed_sequence call per sequence, one after another
    threads8    one call per sequence, 8 concurrent clients
    batch8      embed_sequence with lists of 8 sequences
    batchN      one embed_sequence call with all N sequences
    gw-float    one gateway /v1/embeddings request with all N (JSON floats)
    gw-b64      the same with encoding_format=base64

Each service is warmed up first (model load is not timed). Lengths a model can't take are
skipped. Also `make benchmark`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from dnalm_client import GenericMcpClient

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
# (length in bp, number of sequences). Lengths are multiples of 6 (GENERator); 1500 bp of
# random sequence fits the 512-token BPE models; 30 kb only the long-context ones.
WORKLOADS = [(600, 64), (1500, 32), (30_000, 8)]
MODES = ["single", "threads8", "batch8", "batchN", "gw-float", "gw-b64"]


def random_sequences(length: int, n: int) -> list[str]:
    rng = random.Random(length)
    return ["".join(rng.choice("ACGT") for _ in range(length)) for _ in range(n)]


def chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class Gateway:
    def __init__(self, url: str, token: str | None) -> None:
        self.url = url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def embed(self, model: str, inputs: list[str], encoding: str) -> int:
        """POST /v1/embeddings; return the response size in bytes."""
        body = json.dumps({"model": model, "input": inputs, "encoding_format": encoding})
        req = urllib.request.Request(
            f"{self.url}/v1/embeddings", data=body.encode(), headers=self.headers
        )
        with urllib.request.urlopen(req, timeout=1800) as resp:
            raw = resp.read()
        data = json.loads(raw)["data"]
        if len(data) != len(inputs):
            raise AssertionError(f"gateway returned {len(data)} vectors for {len(inputs)} inputs")
        return len(raw)


def run_mode(mode, make_client, checkpoint, model_id, gateway, seqs) -> tuple[float, int | None]:
    """Embed `seqs` in `mode`; return (seconds, response bytes for gateway modes)."""
    t0 = time.perf_counter()
    size = None
    if mode == "single":
        client = make_client()
        for s in seqs:
            client.embed_sequence(s, checkpoint=checkpoint)
    elif mode == "threads8":
        clients = [make_client() for _ in range(8)]
        for c in clients:  # open the 8 MCP sessions outside the timed part
            c.list_tools()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(8) as pool:
            list(
                pool.map(
                    lambda i: clients[i % 8].embed_sequence(seqs[i], checkpoint=checkpoint),
                    range(len(seqs)),
                )
            )
    elif mode in ("batch8", "batchN"):
        client = make_client()
        for part in chunks(seqs, 8 if mode == "batch8" else len(seqs)):
            out = client.embed_sequence(part, checkpoint=checkpoint)
            if len(out["embeddings"]) != len(part):
                raise AssertionError("wrong number of embeddings")
    elif mode in ("gw-float", "gw-b64"):
        size = gateway.embed(model_id, seqs, "float" if mode == "gw-float" else "base64")
    else:
        raise ValueError(mode)
    return time.perf_counter() - t0, size


def fits(client: GenericMcpClient, checkpoint: str, seqs: list[str]) -> str | None:
    """None if the model takes `seqs` in one call; otherwise the server's error message."""
    try:
        client.embed_sequence(seqs, checkpoint=checkpoint)
        return None
    except RuntimeError as exc:
        return str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("services", nargs="*", help="subset of services (default: all)")
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "http://localhost"))
    parser.add_argument("--token", default=os.environ.get("MCP_AUTH_TOKEN"))
    parser.add_argument("--modes", default=",".join(MODES), help=f"comma list from {MODES}")
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args()
    host = args.host.rstrip("/")
    modes = args.modes.split(",")
    gateway = Gateway(f"{host}:8080", args.token)
    results = []

    for name in args.services or list(PORTS):
        url = f"{host}:{PORTS[name]}"

        def make_client(url=url):
            return GenericMcpClient(url, auth_token=args.token, timeout=1800)

        probe = make_client()
        try:
            probe.health()
        except (urllib.error.URLError, OSError):
            print(f"-- {name}: not reachable, skipped")
            continue
        checkpoint = next(c["name"] for c in probe.list_available_checkpoints() if c.get("default"))
        model_id = f"{name}/{checkpoint}"
        print(f"== {model_id}", flush=True)
        try:
            probe.embed_sequence("ACGT" * 30, checkpoint=checkpoint)  # warm-up: load the model
        except RuntimeError as exc:
            print(f"   warm-up failed, skipped: {str(exc)[:200]}")
            continue

        for length, n in WORKLOADS:
            seqs = random_sequences(length, n)
            error = fits(probe, checkpoint, seqs)  # also the warm-up at this length
            if error:
                reason = "too long for this model" if "limit" in error else error[:200]
                print(f"   {length:>6} bp: skipped ({reason})")
                continue
            row = {"service": name, "checkpoint": checkpoint, "length": length, "n": n}
            for mode in modes:
                try:
                    seconds, size = run_mode(mode, make_client, checkpoint, model_id, gateway, seqs)
                except Exception as exc:  # noqa: BLE001 - record and continue
                    row[mode] = None
                    row[f"{mode}_error"] = str(exc)[:200]
                    continue
                row[mode] = n / seconds
                if size is not None:
                    row[f"{mode}_bytes"] = size
            results.append(row)
            cells = "  ".join(f"{m}={row[m]:7.1f}" if row.get(m) else f"{m}=   fail" for m in modes)
            print(f"   {length:>6} bp x{n:<3} seq/s: {cells}", flush=True)
            for m in modes:
                if f"{m}_error" in row:
                    print(f"      {m} failed: {row[f'{m}_error']}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
