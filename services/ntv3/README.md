# ntv3-mcp

An [MCP](https://modelcontextprotocol.io) (Model Context Protocol) server that wraps
InstaDeep's **[Nucleotide Transformer v3 (NTv3)](https://huggingface.co/collections/InstaDeepAI/nucleotide-transformer-v3)**
genomic language models, so any MCP-compatible client or service (Claude Desktop,
Claude Code, Cursor, your own agent, ...) can call them as plain tools -- no
JAX/Haiku setup, no bespoke inference code, just DNA sequences in, structured
results out.

NTv3 is a family of DNA foundation models using single-nucleotide tokenization and
a U-Net-like architecture, trained on ~9 trillion base pairs across thousands of
genomes, supporting context lengths up to ~1 Mb. This server loads checkpoints
straight from the HuggingFace Hub via `transformers` (`trust_remote_code=True`).

> Part of the [dnalm-hub](../../README.md) repo (sibling services: [ntv2](../ntv2), ...).

> **License notice:** This repository's code is MIT licensed (see [LICENSE](../../LICENSE)).
> The NTv3 **model weights themselves are distributed by InstaDeep under a separate,
> non-commercial license** ("other" on HuggingFace: no commercial use, no training
> competing commercial models). Check the model card for the checkpoint you use
> (e.g. [NTv3_100M_pre](https://huggingface.co/InstaDeepAI/NTv3_100M_pre)) before any
> production or commercial deployment.

## What it does

| Tool | Purpose |
| --- | --- |
| `list_available_checkpoints` | List available NTv3 checkpoints (size, pre/post-training stage, context length). No download. |
| `get_model_info` | Load a checkpoint and report architecture details (hidden size, layer count, params, device/dtype). |
| `get_embedding_layers` | List valid hidden-state layer indices for `embed_sequence`. |
| `embed_sequence` | Compute feature representations (embeddings) for one or more DNA sequences. |
| `score_snp` | Zero-shot effect score for a single-nucleotide substitution (masked-LM based variant scoring). |
| `predict_masked_positions` | Predict top-k nucleotide probabilities at masked / unknown (`N`) positions. NTv3-specific, no evo2-mcp equivalent. |
| `compare_sequences` | Cosine similarity between two sequences' embeddings. NTv3-specific, no evo2-mcp equivalent. |
| `list_species` | Post-trained only: valid `species` values, tracks per species, annotation element names. |
| `list_tracks` | Post-trained only: searchable, paginated track ids for `predict_tracks`. |
| `annotate_sequence` | Post-trained only: per-base probability of genes, exons, introns, promoters, UTRs, ... |
| `predict_tracks` | Post-trained only: predicted RNA-seq / ChIP / ATAC / CAGE signal for chosen tracks. |

All sequences are plain strings using IUPAC letters `A`/`C`/`G`/`T`/`N` (case-insensitive).
Tool names and the `checkpoint` parameter are deliberately aligned with
[evo2-mcp](https://evo2-mcp.readthedocs.io/) -- see
[Compatibility with evo2-mcp](#compatibility-with-evo2-mcp) below.

## Getting access to the model weights (required, one-time)

NTv3 repos on HuggingFace are **gated**: you must accept InstaDeep's license before
you can download any checkpoint, or every tool call will fail with a `401
Client Error` / "You are trying to access a gated repo" message.

1. Log into HuggingFace and open the checkpoint's page, e.g.
   [InstaDeepAI/NTv3_100M_pre](https://huggingface.co/InstaDeepAI/NTv3_100M_pre),
   and accept the license/request access (usually granted instantly).
2. Authenticate the machine running this server, either:
   - `uv run huggingface-cli login` (interactive), or
   - set an `HF_TOKEN` environment variable to a token from
     https://huggingface.co/settings/tokens.

Repeat step 1 for each checkpoint alias you plan to use (access is granted per repo).

## Installation

Requires Python >= 3.11.

Run these from `services/ntv3/`:

```bash
# with uv (recommended)
uv sync

# or with pip, editable install
pip install -e .
```

The first call to any tool that needs a model will download it from the
HuggingFace Hub (cached under the usual `HF_HOME` / `~/.cache/huggingface`) and
keep it in memory for the life of the server process.

## Running

```bash
uv run ntv3-mcp
```

This starts the server on stdio, the standard MCP transport for local/desktop
clients. For interactive debugging, use the official MCP inspector:

```bash
uv run mcp dev src/ntv3_mcp/server.py
```

## Using it from an MCP client

Add an entry to your client's MCP server config (Claude Desktop's
`claude_desktop_config.json`, Claude Code's `.mcp.json`, Cursor's `mcp.json`, etc.).
See [examples/mcp_client_config.json](examples/mcp_client_config.json):

```json
{
  "mcpServers": {
    "ntv3": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/ntv3-mcp", "ntv3-mcp"]
    }
  }
}
```

Or, once published to PyPI, via `uvx` without a local checkout:

```json
{
  "mcpServers": {
    "ntv3": {
      "command": "uvx",
      "args": ["--from", "ntv3-mcp", "ntv3-mcp"]
    }
  }
}
```

## Deploying with Docker

Locally, an MCP server talks stdio to one client on the same machine. On a
server, you instead want a long-running process that MCP clients reach over the
network -- this server supports both via `MCP_TRANSPORT`.

### Quick start

From the **repo root**:

```bash
cp .env.example .env   # fill in HF_TOKEN (required) and MCP_AUTH_TOKEN (recommended)
docker compose up -d --build   # GPU: all services (compose.yaml)
curl http://localhost:8000/health   # -> ok
```

This builds the GPU image ([Dockerfile](Dockerfile)) and starts it with
`MCP_TRANSPORT=http` (already the image's default) on port 8000. It also keeps a
named volume, so downloaded checkpoints survive container restarts. Without
Compose, the equivalent is (again from the repo root, since the build needs
`packages/`):

```bash
docker build -f services/ntv3/Dockerfile -t ntv3-mcp:gpu .   # or: make build S=ntv3
docker run -d --gpus all --name ntv3-mcp -p 8000:8000 \
  -e HF_TOKEN=hf_xxx -e MCP_AUTH_TOKEN=change-me \
  -v ntv3-hf-cache:/home/appuser/.cache/huggingface \
  ntv3-mcp:gpu
```

The MCP endpoint is then `http://<server>:8000/mcp` (streamable-HTTP transport).
Point a remote-capable MCP client at it, e.g. Claude Code's `.mcp.json`:

```json
{
  "mcpServers": {
    "ntv3": {
      "type": "http",
      "url": "http://<server>:8000/mcp",
      "headers": { "Authorization": "Bearer <your MCP_AUTH_TOKEN>" }
    }
  }
}
```

### Securing the endpoint

`streamable-HTTP` has no built-in auth. Set `MCP_AUTH_TOKEN` to a long random
value and the server requires `Authorization: Bearer <token>` on every request
except `GET /health` (used for container health checks). **If `MCP_AUTH_TOKEN`
is unset, the endpoint is completely open** -- the server logs a warning on
startup in that case. This is a single shared secret, not real per-user
auth/OAuth; for anything beyond a private/trusted deployment, put a reverse
proxy (nginx, Caddy, Traefik) in front for TLS termination and put this
container on an internal network / behind a VPN rather than exposing it
directly to the internet.

### CPU-only

The default deployment image is the GPU one ([Dockerfile](Dockerfile)), tested on
an RTX PRO 4500 (Blackwell). It needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and defaults to `NTV3_DEVICE=cuda` and `NTV3_DTYPE=bfloat16`
(`compose.yaml` overrides this to `float32`). Without a GPU, use
[Dockerfile.cpu](Dockerfile.cpu). It avoids pulling several GB of CUDA packages
(see `[tool.uv.sources]` in [pyproject.toml](pyproject.toml)):

```bash
docker compose -f compose.cpu.yaml up -d --build   # from the repo root
```

## Configuration

Set via environment variables (in the client's `env` block, docker-compose's
`environment:`, or your shell):

| Variable | Default | Description |
| --- | --- | --- |
| `MCP_TRANSPORT` | `stdio` (`http` in the Docker image) | `stdio` for local desktop clients; `http` to serve streamable-HTTP for remote/server deployments. |
| `MCP_HOST` | `127.0.0.1` (`0.0.0.0` in the Docker image) | Bind address for the `http` transport. |
| `MCP_PORT` | `8000` | Bind port for the `http` transport. |
| `MCP_PATH` | `/mcp` | URL path for the MCP endpoint under the `http` transport. |
| `MCP_AUTH_TOKEN` | unset (open) | Shared-secret bearer token required on the `http` transport. See "Securing the endpoint" above. |
| `NTV3_DEVICE` | auto (`cuda` > `mps` > `cpu`) | Force a specific torch device. |
| `NTV3_DTYPE` | `float32` | One of `float32`, `bfloat16`, `float16`. `bfloat16` needs a supported GPU. |
| `NTV3_MAX_RESIDENT_MODELS` | `2` | How many checkpoints stay loaded (LRU; evicted ones free their VRAM). |
| `NTV3_MAX_OUTPUT_VALUES` | `500000` | Cap on numbers returned by `annotate_sequence`/`predict_tracks`; above it the error says which `bin_size` to use. |
| `HF_HOME` | `~/.cache/huggingface` | Where model weights are cached. |
| `HF_TOKEN` | unset | **Required** for the gated NTv3 repos unless you've already run `huggingface-cli login`. See "Getting access" above. |

## Model registry

Names accepted by the `checkpoint` argument (default: `100m-pre`). Any other
HuggingFace repo id implementing the NTv3 `trust_remote_code` architecture --
including your own fine-tuned checkpoint -- also works.

| Alias | Repo id | Params | Stage | Context |
| --- | --- | --- | --- | --- |
| `8m-pre` | `InstaDeepAI/NTv3_8M_pre` | 7.7M | pre | ~1Mb |
| `8m-pre-8kb` | `InstaDeepAI/NTv3_8M_pre_8kb` | 7.7M | pre | 8kb |
| `100m-pre` (default) | `InstaDeepAI/NTv3_100M_pre` | 0.1B | pre | ~1Mb |
| `100m-pre-8kb` | `InstaDeepAI/NTv3_100M_pre_8kb` | 0.1B | pre | 8kb |
| `100m-post` | `InstaDeepAI/NTv3_100M_post` | 0.1B | post | ~1Mb |
| `100m-post-131kb` | `InstaDeepAI/NTv3_100M_post_131kb` | 0.1B | post | 131kb |
| `650m-pre` | `InstaDeepAI/NTv3_650M_pre` | 0.7B | pre | ~1Mb |
| `650m-pre-8kb` | `InstaDeepAI/NTv3_650M_pre_8kb` | 0.7B | pre | 8kb |
| `650m-post` | `InstaDeepAI/NTv3_650M_post` | 0.7B | post | ~1Mb |
| `650m-post-131kb` | `InstaDeepAI/NTv3_650M_post_131kb` | 0.7B | post | 131kb |
| `5ds-pre` | `InstaDeepAI/NTv3_5downsample_pre` | 0.6B | pre | ~1Mb |
| `5ds-pre-8kb` | `InstaDeepAI/NTv3_5downsample_pre_8kb` | 0.6B | pre | 8kb |
| `5ds-post` | `InstaDeepAI/NTv3_5downsample_post` | 0.6B | post | ~1Mb |
| `5ds-post-131kb` | `InstaDeepAI/NTv3_5downsample_post_131kb` | 0.6B | post | 131kb |

Run `list_available_checkpoints` for this table at runtime, straight from
[`registry.py`](src/ntv3_mcp/registry.py).

### Post-trained checkpoints: species, annotation, tracks

`pre` checkpoints learned from DNA alone (masked-nucleotide prediction across many
genomes) and need no other input. `post` checkpoints are the same backbone trained
further on functional genomics data, which differs between organisms. So they are
**conditioned on a species** (24 supported: `human`, `mouse`, `rattus_norvegicus`,
`danio_rerio`, `drosophila_melanogaster`, `caenorhabditis_elegans`, `arabidopsis_thaliana`,
`zea_mays`, ... -- see `list_species`). The species token modulates the whole network
and selects species-specific output heads.

- Every shared tool (`embed_sequence`, `compare_sequences`, `score_snp`,
  `predict_masked_positions`) accepts an optional `species`. It defaults to `human` on
  `post` checkpoints and must be omitted on `pre` checkpoints.
- `annotate_sequence` gives the per-base probability of 21 genomic elements: gene,
  exon, intron, splice donor/acceptor, promoter, enhancer, 5'/3' UTR per strand,
  start/stop codon, CTCF, polyA, ORF, ...
- `predict_tracks` gives the predicted experimental signal for chosen track ids. There
  are 7362 for human, 2450 for mouse and fewer for fly and plants. Browse them with
  `list_tracks`. The ids are ENCODE `ENCSR`, FANTOM5 `CNhs`, GEO `GSM`, GTEx, SRA, ...
- Both heads predict only the **central 37.5%** of the (128-padded) input window; the
  rest is context. Pass long windows (tens of kb) centred on the region of interest.
  `bin_size` averages bins to keep responses small.

Sanity check on a 32 kb hg38 window centred on *HBB*: mean exon probability on its
three exons is 0.85-0.90, intron probability in intron 2 is 0.96, the 3'UTR signal
sits on exon 3 (minus strand), and the exon probability 5 kb away is 0.03.

## Notes on `score_snp`

`score_snp` masks a single position and compares NTv3's predicted log-probability
of the alternate allele against the reference allele -- a common zero-shot
heuristic for variant effect prediction used in the Nucleotide Transformer papers.
It is **not** a validated clinical or functional prediction; treat it as a rough
signal, not ground truth. See "Compatibility with evo2-mcp" below for how its
method differs from evo2-mcp's `score_snp` despite the matching name/call shape.

## Compatibility with evo2-mcp

[evo2-mcp](https://evo2-mcp.readthedocs.io/) wraps Arc Institute's **Evo 2**, a
causal (autoregressive) DNA foundation model, as an MCP server. NTv3 is a
**masked** (bidirectional) DNA foundation model -- a different paradigm -- so the
two servers are **not drop-in replacements for each other**: tool availability and
exact semantics differ. They are, however, both plain MCP servers over stdio, so
nothing stops a client or agent from running both side by side and picking
whichever tool fits.

This server's tool names and the `checkpoint` parameter are aligned with
evo2-mcp's conventions wherever NTv3's architecture allows an equivalent
operation, to make switching between the two easier:

| evo2-mcp | ntv3-mcp | Compatible? |
| --- | --- | --- |
| `list_available_checkpoints()` | `list_available_checkpoints()` | Same name/shape (list of `{name, description, ...}`). |
| `get_embedding_layers(checkpoint, which)` | `get_embedding_layers(checkpoint, which)` | Same name/params. NTv3's "recommended" layers are a generic heuristic (see docstring), not an InstaDeep-published recommendation like evo2's may be. |
| `embed_sequence(sequence, checkpoint, layer_name)` | `embed_sequence(sequence, checkpoint, layer_name, pooling)` | Same call shape for a single `sequence` string. `sequence` may also be a list here (NTv3-mcp batching extension); `pooling` is an added option ("mean" default vs. evo2's implicit per-token/2D output -- pass `pooling="per_token"` to match that shape). |
| `score_snp(sequence, alternative_allele, checkpoint, reduce_method)` | `score_snp(sequence, alternative_allele, checkpoint, position)` | Same required args and defaults (position defaults to sequence center, like evo2-mcp always uses). **Method differs**: evo2-mcp compares whole-sequence log-likelihoods (valid for its causal model); this compares the model's log-probability of the two alleles from a single masked-position forward pass (the standard approach for masked DNA LMs). Output field names match (`original_score`, `mutated_score`, `score_delta`, etc.) but are computed differently -- see the tool's docstring. No `reduce_method` (not meaningful for a single masked position). |
| `score_sequence(sequence, checkpoint, reduce_method)` | -- | Not implemented. A masked-model analog (pseudo-log-likelihood over all positions) would need one forward pass per position and hasn't been added yet. |
| `generate_sequence(prompt, checkpoint, n_tokens, temperature, top_k)` | -- | **No NTv3 analog.** NTv3 is not autoregressive and cannot generate novel sequence continuations. Closest available capability is `predict_masked_positions` (fill in specific masked positions, not open-ended generation). |
| -- | `predict_masked_positions(sequence, positions, checkpoint, top_k)` | NTv3-specific addition; no evo2-mcp equivalent. |
| -- | `compare_sequences(sequence_a, sequence_b, checkpoint, layer_name)` | NTv3-specific addition; no evo2-mcp equivalent. |

Other differences worth knowing before running both in the same deployment:

| | evo2-mcp | ntv3-mcp |
| --- | --- | --- |
| Install | `pip install evo2-mcp` | `uv sync` / `pip install -e .` (this repo) |
| Run | `python -m evo2_mcp.main` | `uv run ntv3-mcp` |
| Python | 3.12+ | 3.11+ |
| Heavy deps | CUDA, `transformer-engine`, `flash-attn`, `evo2` (effectively GPU-only) | `transformers` + `torch` only (CPU or GPU) |
| Weight license | Evo 2 (Apache-2.0) | NTv3 ("other"/non-commercial, HF-gated -- see above) |

## Development

```bash
make test S=ntv3        # from the repo root; or inside services/ntv3: uv run pytest
```

Tests under [`tests/`](tests/) only cover pure logic (sequence validation, model
registry resolution) and do not download any model weights.

## Architecture

```
src/ntv3_mcp/
  registry.py     # known NTv3 checkpoints + alias resolution
  inference.py    # tokenizer/model loading (cached), embeddings, variant scoring,
                  # masked prediction -- all torch/transformers logic lives here
  server.py       # MCPServer app + tool wrappers + stdio/HTTP transport + auth
src/ntv3_mcp_client/  # thin Python client (Ntv3McpClient) on top of dnalm-client
Dockerfile        # GPU deployment image (see "Deploying with Docker")
Dockerfile.cpu    # CPU-only image
```

Shared code (device selection, model cache, HTTP/auth, ...) lives in
[packages/dnalm-common](../../packages/dnalm-common).

## Contributing

Issues and PRs welcome. This is an independent, community wrapper and is not
affiliated with or endorsed by InstaDeep.

## Acknowledgements

- [InstaDeep](https://www.instadeep.com/) for training and open-sourcing the
  Nucleotide Transformer models.
- [InstaDeepAI/nucleotide-transformer-v3 collection](https://huggingface.co/collections/InstaDeepAI/nucleotide-transformer-v3) on HuggingFace.
- [instadeepai/nucleotide-transformer](https://github.com/instadeepai/nucleotide-transformer) (native JAX implementation).
- Built on the [Model Context Protocol Python SDK](https://github.com/modelcontextprotocol/python-sdk).
