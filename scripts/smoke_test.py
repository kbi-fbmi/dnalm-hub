"""Smoke test of a running dnalm-hub: every model service over MCP, plus the gateway.

    uv run --project packages/dnalm-client python scripts/smoke_test.py
    uv run --project packages/dnalm-client python scripts/smoke_test.py --host http://gpu-server ntv2 evo2

Uses each service's default checkpoint (the first call downloads and loads it, so allow a
few minutes) and checks: health, list_available_checkpoints, embed_sequence (single and
batch agree), compare_sequences, score_snp, generate_sequence on causal models, and the
gateway's /v1/models and /v1/embeddings. Services that aren't reachable are reported and
skipped. Exit code 1 if anything that is up fails. Also `make smoke`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request

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
GATEWAY_PORT = 8080

rng = random.Random(0)
SEQ_A = "".join(rng.choice("ACGT") for _ in range(600))  # multiple of 6 (GENERator)
SEQ_B = "".join(rng.choice("ACGT") for _ in range(600))
SNP_POS = 150
ALT = "A" if SEQ_A[SNP_POS] != "A" else "C"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def check_service(client: GenericMcpClient) -> tuple[str, list[str]]:
    """Run the checks against one service; return its default checkpoint and one line per check."""
    lines = []
    checkpoints = client.list_available_checkpoints()
    default = next(c["name"] for c in checkpoints if c.get("default"))
    lines.append(f"{len(checkpoints)} checkpoints, default {default}")

    t = time.time()
    batch = client.embed_sequence([SEQ_A, SEQ_B], checkpoint=default)["embeddings"]
    single = client.embed_sequence(SEQ_A, checkpoint=default)["embedding"]
    agree = cosine(batch[0], single)
    if agree < 0.999:
        raise AssertionError(f"batch and single embeddings differ (cosine {agree:.5f})")
    lines.append(
        f"embed: {len(single)} dim, batch == single (cos {agree:.6f}), {time.time() - t:.1f}s"
    )

    mutated = SEQ_A[:SNP_POS] + ALT + SEQ_A[SNP_POS + 1 :]
    sim = client.compare_sequences(SEQ_A, mutated, checkpoint=default)["cosine_similarity"]
    lines.append(f"compare 1-SNP: {sim:.5f}")

    snp = client.score_snp(SEQ_A[:300], ALT, checkpoint=default, position=SNP_POS)
    lines.append(f"score_snp: {snp['score_delta']:+.3f} ({snp['method']})")

    tools = {t["name"] for t in client.list_tools()}
    if "generate_sequence" in tools:
        out = client.generate_sequence(SEQ_A[:120], checkpoint=default, max_new_tokens=8)
        lines.append(f"generate: +{out['num_new_bases']} bases {out['generated_sequence'][:24]}")
    return default, lines


def gateway_request(url: str, token: str | None, payload: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.load(resp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("services", nargs="*", help=f"subset of {', '.join(PORTS)} (default: all)")
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "http://localhost"))
    parser.add_argument("--token", default=os.environ.get("MCP_AUTH_TOKEN"))
    args = parser.parse_args()
    host = args.host.rstrip("/")
    names = args.services or list(PORTS)
    unknown = set(names) - set(PORTS)
    if unknown:
        parser.error(f"unknown service(s): {', '.join(sorted(unknown))}")

    failed, defaults = [], {}
    for name in names:
        client = GenericMcpClient(f"{host}:{PORTS[name]}", auth_token=args.token, timeout=900)
        try:
            client.health()
        except (urllib.error.URLError, OSError) as exc:
            print(f"-- {name}: not reachable ({exc}), skipped")
            continue
        print(f"== {name}")
        try:
            defaults[name], lines = check_service(client)
            for line in lines:
                print(f"   {line}")
        except Exception as exc:  # noqa: BLE001 - report and continue with the next service
            print(f"   FAIL: {type(exc).__name__}: {str(exc)[:300]}")
            failed.append(name)

    gw = f"{host}:{GATEWAY_PORT}"
    try:
        version = gateway_request(f"{gw}/version", None)["version"]
    except (urllib.error.URLError, OSError) as exc:
        print(f"-- gateway: not reachable ({exc}), skipped")
    else:
        print(f"== gateway {version}")
        try:
            models = {m["id"] for m in gateway_request(f"{gw}/v1/models", args.token)["data"]}
            print(f"   {len(models)} models")
            for name, default in defaults.items():
                model = f"{name}/{default}"
                if model not in models:
                    raise AssertionError(f"{model} missing from /v1/models")
                out = gateway_request(
                    f"{gw}/v1/embeddings", args.token, {"model": model, "input": [SEQ_A, SEQ_B]}
                )
                print(f"   {out['model']}: {len(out['data'])} x {len(out['data'][0]['embedding'])}")
        except Exception as exc:  # noqa: BLE001
            print(f"   FAIL: {type(exc).__name__}: {str(exc)[:300]}")
            failed.append("gateway")

    print("\nOK" if not failed else f"\nFAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
