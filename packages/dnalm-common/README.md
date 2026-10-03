# dnalm-common

Shared internals for the genomic MCP services in [`services/`](../../services). Not
published standalone; each service depends on it as a local path dependency
(`../../packages/dnalm-common`). Add helpers here, with tests in `tests/`, when
more than one service needs them.

- `device.py` -- device/dtype selection from an env var, same logic every service used to duplicate.
- `validation.py` -- DNA sequence validation shared by every masked/causal backend.
- `model_cache.py` -- `LRUModelCache`, a bounded tokenizer/model cache with real VRAM eviction
  (replaces the unbounded per-service caches that existed before this package).
- `http_app.py` -- the `/health` route, bearer-token auth middleware, and `run_server()` transport
  selection (stdio vs streamable-HTTP), shared verbatim across services.
- `tokenization.py` -- `is_single_nucleotide_tokenizer` / `assert_single_nucleotide_tokenizer`, a
  guard against running single-position masked scoring against a k-mer/BPE tokenizer where that
  position math is not valid.
- `scoring.py` -- tokenizer-agnostic SNP scoring (`score_snp_whole_sequence`): score the reference
  and mutated sequences independently and report the log-likelihood delta. Includes the masked-LM
  pseudo-log-likelihood used by NTv2/DNABERT-2; causal backends plug in their own scorer.
- `errors.py` -- `user_errors_as_tool_errors`, so `ValueError` messages (bad input) reach MCP
  clients instead of the SDK's generic "Error executing tool <name>".
