# genomic-mcp-common

Shared internals for the genomic MCP server family (`ntv3-mcp` and its siblings:
`ntv2-mcp`, `dnabert2-mcp`, `hyenadna-mcp`, `evo2t-mcp`). Not published standalone;
consumed as a local path dependency by each service.

- `device.py` -- device/dtype selection from an env var, same logic every service used to duplicate.
- `validation.py` -- DNA sequence validation shared by every masked/causal backend.
- `model_cache.py` -- `LRUModelCache`, a bounded tokenizer/model cache with real VRAM eviction
  (replaces the unbounded per-service caches that existed before this package).
- `http_app.py` -- the `/health` route, bearer-token auth middleware, and `run_server()` transport
  selection (stdio vs streamable-HTTP), shared verbatim across services.
- `tokenization.py` -- `is_single_nucleotide_tokenizer` / `assert_single_nucleotide_tokenizer`, a
  guard against running single-position masked scoring against a k-mer/BPE tokenizer where that
  position math is not valid.
