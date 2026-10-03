# ntv2-mcp

MCP server for InstaDeep's [Nucleotide Transformer v2](https://huggingface.co/collections/InstaDeepAI/nucleotide-transformer-65099cdde13ff96230f2e592)
DNA language models. Sibling of [`ntv3-mcp`](../ntv3): same tool names, `checkpoint`
parameter, response shapes, auth, and `/health`, built on the shared [`packages/`](../../packages).

## Tools

| Tool | Notes |
|---|---|
| `list_available_checkpoints` | `50m`, `100m`, `250m`, `500m` (default), `2.5b-v1`; each pinned to a Hub commit |
| `get_model_info`, `get_embedding_layers` | as in ntv3-mcp, plus `max_tokens` |
| `embed_sequence` | `per_token` rows are **tokens** (`<cls>` + 6-mers), not bases |
| `compare_sequences` | cosine similarity of mean-pooled embeddings |
| `score_snp` | `method="whole_sequence_log_prob_delta"` -- see below |

Not available, by design: `predict_masked_positions` (a 6-mer token is not one base, so
single-position masking is ill-defined) and `generate_sequence` (masked LM).

**`score_snp`** tokenizes the reference and mutated sequences independently and reports
the difference of their masked-LM pseudo-log-likelihoods (mask each token in turn, sum the
log-probability of the true token). It costs about `2 x length/6` forward passes, batched
by `NTV2_PLL_BATCH_SIZE` (default 16). Scores are not comparable with ntv3-mcp's
single-position method.

**Context limit:** 2048 tokens (~12 kb) for v2, 1000 tokens (~6 kb) for `2.5b-v1`, counting
the leading `<cls>` and one token per `N`. Longer inputs are rejected with an explicit
error, never truncated.

## Build and run

From the **repo root** (the build needs `packages/`):

```bash
make build S=ntv2      # = docker build -f services/ntv2/Dockerfile -t ntv2-mcp:gpu .
docker run --rm --gpus all -p 8001:8000 -e MCP_AUTH_TOKEN=change-me \
  -v "$PWD/models":/models ntv2-mcp:gpu
```

NTv2 weights are public (no `HF_TOKEN` needed) but **non-commercial** licensed.

| Env var | Default (image) | |
|---|---|---|
| `NTV2_DEVICE` | `cuda` | |
| `NTV2_DTYPE` | `float32` | `bfloat16` works (via autocast) but is noticeably noisier and saves little VRAM at these sizes |
| `NTV2_MAX_RESIDENT_MODELS` | `2` | LRU limit on loaded checkpoints |
| `NTV2_PLL_BATCH_SIZE` | `16` | masked copies per forward pass in `score_snp` |
| `MCP_AUTH_TOKEN`, `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT`, `MCP_PATH` | | same as ntv3-mcp |

Peak VRAM measured on an RTX PRO 4500 (fp32, max-length input): 50m 0.8 GiB, 500m 2.9 GiB,
2.5b-v1 10.1 GiB.

## Why a separate environment

NTv2's remote `modeling_esm.py` imports `find_pruneable_heads_and_indices`, which
transformers 5.x removed, so this service pins `transformers>=4.57,<5` while ntv3-mcp
runs 5.x.

## Tests

No downloads, no GPU: `make test S=ntv2` from the repo root (or `uv run pytest` here).
