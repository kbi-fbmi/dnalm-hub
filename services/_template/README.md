# __NAME__-mcp

MCP server for the __NAME__ genomic language model(s). Created from `services/_template`
(`make new-service NAME=__NAME__`); follow the checklist in
[docs/adding-a-model.md](../../docs/adding-a-model.md).

## Checkpoints

TODO: table of registered aliases, parameters, context limit, license.

## Tools

`list_available_checkpoints`, `get_model_info`, `embed_sequence` (single sequence or a batch),
`score_snp`, `compare_sequences`. TODO: list anything model-specific.

## Build and run

From the repo root:

```bash
make build S=__NAME__
make test S=__NAME__
```

| Env var | Default | |
|---|---|---|
| `__ENV___DEVICE` | auto (`cuda` in the image) | |
| `__ENV___DTYPE` | `float32` | |
| `__ENV___MAX_RESIDENT_MODELS` | `2` | LRU limit on loaded checkpoints |
| `MCP_AUTH_TOKEN`, `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT`, `MCP_PATH` | | same for all services |
