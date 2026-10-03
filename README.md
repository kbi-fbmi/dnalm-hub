# dnalm-hub

One server for many DNA language models (NTv3, NTv2, DNABERT-2, HyenaDNA, Evo 2, GENERator,
GENA-LM, GROVER),
reachable over [MCP](https://modelcontextprotocol.io). Each model family runs as its own
service with the same tool names and response shapes, so a client can switch
between models by changing only the URL and the `checkpoint`.

Developed at the Department of Biomedical Informatics, Faculty of Biomedical Engineering,
Czech Technical University in Prague (Kladno, Czech Republic). If you use it in your
work, please [cite our paper](#citation).

| Service | Models | Port | Status |
|---|---|---|---|
| [`services/ntv3`](services/ntv3) | InstaDeep Nucleotide Transformer v3 (1 token per base) | 8000 | running |
| [`services/ntv2`](services/ntv2) | InstaDeep Nucleotide Transformer v2 + v1 2.5B (6-mer tokens) | 8001 | running |
| [`services/dnabert2`](services/dnabert2) | DNABERT-2 117M (BPE + ALiBi masked LM, 512 tokens ≈ 2.4 kb) | 8002 | running |
| [`services/hyenadna`](services/hyenadna) | HyenaDNA tiny-1k ... large-1m (causal, 1 token per base, up to 1M bases) | 8003 | running |
| [`services/evo2`](services/evo2) | Evo 2 7B (causal StripedHyena 2, 1 token per base, up to 32 kb by default; community PyTorch port, experimental) | 8004 | running |
| [`services/generator`](services/generator) | GENERator v1/v2 1.2B/3B (causal Llama, 6-mer tokens, ~98 kb) | 8005 | running |
| [`services/genalm`](services/genalm) | GENA-LM BERT base/large + BigBird (BPE masked LM, 512/4096 tokens) | 8006 | running |
| [`services/grover`](services/grover) | GROVER (human-genome BERT, BPE, 512 tokens ≈ 1.9 kb) | 8007 | running |
| [`services/gateway`](services/gateway) | OpenAI-compatible `/v1/embeddings` over all of the above | 8080 | running |

Tools available on every service: `list_available_checkpoints`, `get_model_info`,
`get_embedding_layers`, `embed_sequence` (one sequence or a batch), `compare_sequences`
and `score_snp`. Model-specific tools: `predict_masked_positions` (NTv3), genome annotation and
track prediction for species-conditioned NTv3 `post` checkpoints (`annotate_sequence`, `predict_tracks`,
`list_species`, `list_tracks`), and `generate_sequence` (causal models: HyenaDNA, Evo 2, GENERator). Call model-specific
tools from Python with `client.call("tool_name", **arguments)`.

**OpenAI API:** the gateway embeds DNA with any model through the standard OpenAI
embeddings API, e.g. `OpenAI(base_url="http://localhost:8080/v1", api_key=MCP_AUTH_TOKEN)
.embeddings.create(model="ntv3/100m-pre", input=["ACGT..."])`. See
[services/gateway](services/gateway).

**Try it:** [examples/model_overview.ipynb](examples/model_overview.ipynb) asks each
service for its models, picks one per type, and tests embeddings (single, batch and
per-token), similarity and SNP scoring.

**GPU sharing:** all services share one GPU. Each loads a checkpoint on first use and
frees it after `DNALM_IDLE_UNLOAD_SECONDS` without requests (default 900 in compose.yaml;
0 keeps models loaded). The biggest models need roughly 16 GB (Evo 2 7B, bf16) and
14 GB (GENERator 3B), so on a 32 GB card don't expect every family to be resident at
once. Start only the services you need if the GPU is smaller.

## Layout

```
packages/
  dnalm-common/         shared server code: device/dtype, model cache, HTTP+auth, scoring, errors
  dnalm-client/         stdlib-only Python client (GenericMcpClient)
services/
  ntv3/ ntv2/ dnabert2/ hyenadna/ evo2/ generator/ genalm/ grover/
                        one directory per model family: pyproject.toml, uv.lock,
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
docker compose up -d --build     # start / update all services (or name some: ... up -d ntv3 gateway)
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

## Citation

If you use dnalm-hub, please cite:

> Krupička, R.; Komárková, M.; Dvorský, B.; Kollinová, K.; Klempíř, O.
> Benchmarking genomic foundation models for binary classification of gene fusion
> breakpoints from DNA sequences. *BioData Mining*. 2026, 19(1), 41. ISSN 1756-0381.
> [doi:10.1186/s13040-026-00553-1](https://doi.org/10.1186/s13040-026-00553-1)

```bibtex
@article{krupicka2026gfm_fusion,
  author  = {Krupi{\v{c}}ka, Radim and Kom{\'a}rkov{\'a}, Mariana and Dvorsk{\'y}, Bohuslav
             and Kollinov{\'a}, Kate{\v{r}}ina and Klemp{\'i}{\v{r}}, Ond{\v{r}}ej},
  title   = {Benchmarking genomic foundation models for binary classification of gene
             fusion breakpoints from DNA sequences},
  journal = {BioData Mining},
  year    = {2026},
  volume  = {19},
  number  = {1},
  pages   = {41},
  issn    = {1756-0381},
  doi     = {10.1186/s13040-026-00553-1}
}
```
## License

The code is MIT licensed. Model weights have their own licenses; for example, the
NTv2 and NTv3 weights are **non-commercial**. Check each service's README and the
model cards.
