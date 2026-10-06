# Changelog

All notable changes to dnalm-hub. The whole repository (shared packages, every
service and the gateway) shares one version and is released together; versions follow
[Semantic Versioning](https://semver.org) and this file follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). How to cut a release:
[docs/releasing.md](docs/releasing.md).

## [Unreleased]

### Added
- Tutorial: six notebooks in `examples/tutorial` (getting started, embeddings with a small
  classifier, variant effects with a mutation scan, generation, NTv3 annotation and tracks,
  the OpenAI gateway), replacing `examples/model_overview.ipynb`.
- `scripts/smoke_test.py` (`make smoke`): checks every running service and the gateway.

### Fixed
- Concurrent requests to one service could fail or hang: the model code isn't
  thread-safe (e.g. NTv2's rotary cache), and MCP runs tools in worker threads. Tools
  that run a model now execute one at a time per service (`one_call_at_a_time`).
- `GenericMcpClient` reused one JSON-RPC id for every request, so two threads sharing a
  client could get each other's response, or wait forever. Each request now gets its own
  id, and session creation is locked.
- Services shared the GPU poorly: PyTorch kept freed memory cached, so after one long batch
  a larger model (Evo 2) could no longer load. After each call, cached memory above
  `DNALM_CUDA_CACHE_LIMIT_MB` (default 1024) is now released, and running out of GPU
  memory returns a readable error instead of "Error executing tool".

### Changed
- `GenericMcpClient` sends `checkpoint` only when given, so each server's default applies
  (it used to send NTv3's `100m-pre` to every service).
- `examples/mcp_client_config.json` configures all services over HTTP;
  `examples/embed_sequence_example.py` works with any service.
- README: weight licenses in the service table.

### Removed
- The NTv3-only smoke-test scripts (`scripts/test_mcp_*`), the `ntv3_mcp_client` package
  (a thin subclass of `GenericMcpClient`) and `services/ntv3/main.py`.

## [0.9.0] - 2026-10-03

First public release. Close to complete; 1.0.0 follows once the open items below are
checked.

### Added
- Model services, all with the same MCP tools (`list_available_checkpoints`,
  `get_model_info`, `get_embedding_layers`, `embed_sequence`, `compare_sequences`,
  `score_snp`) and response shapes:
  - `ntv3` (port 8000): InstaDeep Nucleotide Transformer v3, pre- and post-trained
    checkpoints, with `predict_masked_positions` and, for post-trained checkpoints,
    `annotate_sequence`, `predict_tracks`, `list_species`, `list_tracks`.
  - `ntv2` (8001): Nucleotide Transformer v2 50M-500M and v1 2.5B.
  - `dnabert2` (8002): DNABERT-2 117M.
  - `hyenadna` (8003): HyenaDNA tiny-1k to large-1m.
  - `evo2` (8004): Evo 2 7B via the community PyTorch port `Aquiles-ai/Evo2-7B`.
  - `generator` (8005): GENERator v1/v2, eukaryote and prokaryote, 1.2B and 3B.
  - `genalm` (8006): GENA-LM BERT base/large and BigBird.
  - `grover` (8007): GROVER.
- `generate_sequence` on the causal models (HyenaDNA, Evo 2, GENERator) with a common
  response shape.
- `gateway` (8080): OpenAI-compatible `POST /v1/embeddings` and `GET /v1/models` over all
  services; non-OpenAI request fields are passed to `embed_sequence`; `GET /version`.
- Shared packages `dnalm-common` (device/dtype, LRU model cache with idle unload via
  `DNALM_IDLE_UNLOAD_SECONDS`, validation, SNP scoring, HTTP + bearer auth) and
  `dnalm-client` (stdlib-only MCP client).
- One Docker Compose deployment (`compose.yaml`, GPU; `compose.cpu.yaml`, NTv3 + gateway
  on CPU) with model weights in a shared host directory (`MODELS_DIR`).
- `make new-service` scaffold, `docs/adding-a-model.md`, `examples/model_overview.ipynb`.
- Versioning: `scripts/version.py`, `make version` / `check-version` / `bump`,
  `CITATION.cff`, CI and release workflows.

### Known limitations (to resolve before 1.0.0)
- Evo 2 runs through a community port, not Arc's `evo2` package; numerical agreement
  with the official implementation is not verified. Inputs are capped at 32 kb by
  default (`EVO2_MAX_SEQ_LEN`). The 1B checkpoint is unusable in bf16 and not registered.
- DNABERT-2 accepts at most 512 tokens (~2.4 kb), so the 10 kbp inputs used in the
  paper are rejected.
- GENERator needs inputs whose length is a multiple of 6 unless `remainder` is given.
- The gateway returns only mean-pooled embeddings.

[Unreleased]: https://github.com/kbi-fbmi/dnalm-hub/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/kbi-fbmi/dnalm-hub/releases/tag/v0.9.0
