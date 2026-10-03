# dnalm-hub

One server for many DNA language models (NTv3, NTv2, and soon DNABERT-2, HyenaDNA, Evo2),
reachable over [MCP](https://modelcontextprotocol.io). Each model family runs as its own
service with the same tool names and response shapes, so a client can switch
between models by changing only the URL and the `checkpoint`.

| Service | Models | Port | Status |
|---|---|---|---|
| [`services/ntv3`](services/ntv3) | InstaDeep Nucleotide Transformer v3 (1 token per base) | 8000 | running |
| [`services/ntv2`](services/ntv2) | InstaDeep Nucleotide Transformer v2 + v1 2.5B (6-mer tokens) | 8001 | running |
| [`services/gateway`](services/gateway) | OpenAI-compatible `/v1/embeddings` over all of the above | 8080 | running |
| `services/dnabert2` | DNABERT-2 (BPE) | | planned |
| `services/hyenadna` | HyenaDNA (causal, up to 1M context) | | planned |
| `services/evo2t` | Evo2 (community PyTorch port, experimental) | | planned |

Tools available on every service: `list_available_checkpoints`, `get_model_info`,
`get_embedding_layers`, `embed_sequence` (one sequence or a batch), `compare_sequences`
and `score_snp`. Model-specific tools: `predict_masked_positions` (NTv3), genome annotation and
track prediction for species-conditioned NTv3 `post` checkpoints (`annotate_sequence`, `predict_tracks`,
`list_species`, `list_tracks`), and `generate_sequence` (causal models, planned). Call model-specific
tools from Python with `client.call("tool_name", **arguments)`.

**OpenAI API:** the gateway embeds DNA with any model through the standard OpenAI
embeddings API, e.g. `OpenAI(base_url="http://localhost:8080/v1", api_key=MCP_AUTH_TOKEN)
.embeddings.create(model="ntv3/100m-pre", input=["ACGT..."])`. See
[services/gateway](services/gateway).

**Try it:** [examples/model_overview.ipynb](examples/model_overview.ipynb) asks each
service for its models, picks one per type, and tests embeddings (single, batch and
per-token), similarity and SNP scoring.

## Layout

```
packages/
  dnalm-common/         shared server code: device/dtype, model cache, HTTP+auth, scoring, errors
  dnalm-client/         stdlib-only Python client (GenericMcpClient)
services/
  ntv3/  ntv2/          one directory per model family: pyproject.toml, uv.lock,
                        Dockerfile, src/<name>_mcp/{registry,inference,server}.py, tests/
  gateway/              OpenAI-compatible /v1/embeddings that routes to the model services
  _template/            skeleton used by `make new-service`
docs/                   adding-a-model.md
examples/               notebook and scripts for API users
scripts/                shell/PowerShell smoke tests against a running server
compose.yaml            GPU deployment of all services (compose.cpu.yaml: NTv3 + gateway on CPU)
Makefile                development tasks across the separate projects
```

**Why not one uv workspace?** A uv workspace resolves all members into one lockfile.
Model families need incompatible dependencies: NTv2's remote code needs
`transformers<5` while NTv3 runs on 5.x, and DNABERT-2 must not have Triton while Evo2
needs it. So each service has its own environment, and the root `Makefile` makes
them feel like one project (uv has no task runner of its own).

## Common tasks

```bash
# deploy (plain Docker Compose)
cp .env.example .env             # HF_TOKEN, MCP_AUTH_TOKEN, optionally MODELS_DIR
mkdir -p models                  # model weights on the host (default MODELS_DIR=./models)
docker compose up -d --build     # start / update all services
docker compose ps                # status;  docker compose logs -f ntv2;  docker compose down

# develop (the Makefile loops over the separate uv projects)
make sync                        # create venvs (make sync CLEAN=1 after moving/renaming the repo)
make test                        # every package and service (or: make test S=ntv2)
make lint                        # ruff check + format check (make format fixes; config: ruff.toml)
make lock                        # after editing a pyproject.toml
make new-service NAME=hyenadna   # add a model family -> docs/adding-a-model.md
```

`make test` needs a recent [uv](https://docs.astral.sh/uv/) (the lockfiles use the
current format; uv 0.5 is too old). To use a specific binary, run
`make test UV=/path/to/uv`.

## License

The code is MIT licensed. Model weights have their own licenses; for example, the
NTv2 and NTv3 weights are **non-commercial**. Check each service's README and the
model cards.
