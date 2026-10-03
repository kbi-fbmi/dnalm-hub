# Adding a model family

Every model family is a separate service under `services/<name>/` with its own
environment. The steps below take a new family (e.g. HyenaDNA) from nothing to a
running container.

## 1. Scaffold

```bash
make new-service NAME=hyenadna        # lowercase letters/digits only
```

This copies `services/_template/` to `services/hyenadna/`, renames the package to
`hyenadna_mcp`, sets env-var prefixes to `HYENADNA_*`, and creates `uv.lock`.
The result already passes its own tests (`make test S=hyenadna`). The model calls
are stubs that raise `NotImplementedError`.

## 2. Fill in the model

| File | What to do |
|---|---|
| `pyproject.toml` | Pin the dependencies this model really needs (e.g. `transformers<5`, no `triton`). Then run `uv lock`. |
| `src/<pkg>/registry.py` | List the checkpoints: alias, HF repo id, **pinned `revision`**, params, context limit. |
| `src/<pkg>/inference.py` | Implement `load`, `model_info`, `compute_embeddings`, `log_likelihood`. |
| `src/<pkg>/server.py` | Write the `instructions` text, and add or remove tools (see below). |
| `tests/test_server_tools.py` | Keep `EXPECTED_TOOLS` in sync with the tools you register. |
| `README.md` | Checkpoints, model-specific behavior, env vars, license. |

Reuse `packages/dnalm-common` instead of copying code. It provides:

- device/dtype selection (`select_device`, `select_dtype`),
- the bounded model cache with VRAM release (`LRUModelCache`),
- sequence validation (`validate_sequence`),
- SNP scoring (`score_snp_whole_sequence`, `masked_lm_pseudo_log_likelihood`),
- the single-nucleotide tokenizer check (`is_single_nucleotide_tokenizer`),
- HTTP, auth and `/health` (`http_app`),
- readable tool errors (`user_errors_as_tool_errors`).

If two services need the same new helper, add it there once, with tests.

## 3. Choose the tool set

All services share tool names, the `checkpoint` parameter and the response shapes.
Clients and the planned `/v1/embeddings` gateway rely on that.

| Tool | Register when |
|---|---|
| `list_available_checkpoints`, `get_model_info`, `embed_sequence`, `compare_sequences`, `score_snp` | always |
| `get_embedding_layers` | the model exposes per-layer hidden states |
| `predict_masked_positions` | masked LM **and** one token per base (`is_single_nucleotide_tokenizer`) |
| `generate_sequence` | causal (autoregressive) LM |

`embed_sequence` must accept one string or a list of strings (a batch).
`score_snp` must set `method`: `single_position_masked_lm_log_prob` (NTv3 only) or
`whole_sequence_log_prob_delta` (everything else).

Raise `ValueError` for bad input: invalid bases, an over-long sequence, an unknown
checkpoint. The `@user_errors_as_tool_errors` decorator passes these messages to
the client; any other exception reaches the client only as "Error executing tool".
Never truncate silently.

## 4. Verify on the GPU

Unit tests never download models. Before you deploy, do a real smoke test:

```bash
make build S=hyenadna
docker run --rm --gpus all -p 8010:8000 -e MCP_AUTH_TOKEN=test hyenadna-mcp:gpu
```

Then run `examples/model_overview.ipynb` against it. Add the service to `SERVICES`
in the notebook's first code cell.

## 5. Deploy

Add a block to `compose.yaml` in the repo root. Copy an existing one and give it:

- a new host port,
- its own `<name>-hf-cache` volume.

Then run `docker compose up -d --build`.
