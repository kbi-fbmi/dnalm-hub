# dnabert2-mcp

MCP server for [DNABERT-2](https://huggingface.co/zhihan1996/DNABERT-2-117M)
([paper](https://arxiv.org/abs/2306.15006)), a 117M-parameter multi-species masked DNA
language model with a BPE tokenizer and ALiBi attention. Sibling of [`ntv2-mcp`](../ntv2):
same tool names, `checkpoint` parameter, response shapes, auth, and `/health`, built on the
shared [`packages/`](../../packages).

## Checkpoints

| Alias | Repo | Revision | Params | Context |
|---|---|---|---|---|
| `117m` (default) | `zhihan1996/DNABERT-2-117M` | `7bce263b` | 117M | 512 tokens (~2.4 kb) |

Only DNABERT-2-117M is registered (DNABERT-S is deliberately left out). Any other repo id
with the same architecture (a fine-tune that keeps the masked-LM head) can be passed as
`checkpoint`, unpinned.

## Tools

| Tool | Notes |
|---|---|
| `list_available_checkpoints` | the one pinned checkpoint above |
| `get_model_info`, `get_embedding_layers` | 13 hidden-state layers: 0 = embedding block, 12 = final |
| `embed_sequence` | one string or a list (batch); `per_token` rows are **BPE tokens** (`[CLS]`, tokens, `[SEP]`), not bases |
| `compare_sequences` | cosine similarity of mean-pooled embeddings |
| `score_snp` | `method="whole_sequence_log_prob_delta"`, see below |

Not available, by design: `predict_masked_positions` (a BPE token spans 1 to ~16 bases, so
single-position masking is ill-defined) and `generate_sequence` (masked LM).

**Embeddings.** Mean pooling averages every non-padding token, `[CLS]`/`[SEP]` included, as
on the model card. The remote `BertModel` ignores `output_hidden_states` and runs its encoder
on unpadded tokens, so intermediate layers are captured with a forward hook and scattered
back into a padded batch. Batching does not change results (checked: batch vs single
max abs diff ~1e-7).

**`score_snp`** tokenizes the reference and mutated sequences independently and reports the
difference of their masked-LM pseudo-log-likelihoods (mask each token in turn, sum the
log-probability of the true token; `[UNK]` tokens for `N` are skipped). A SNP can change
how its neighborhood is tokenized, so the two sums may cover different token counts. It
costs about `2 x number of tokens` forward passes, batched by `DNABERT2_PLL_BATCH_SIZE`
(default 16). Scores are not comparable with other services' scores.

**Context limit.** ALiBi has no hard limit, but DNABERT-2 was pre-trained on short
(~128-token) inputs and its config sizes the ALiBi table to 512. This server caps inputs at
512 tokens including `[CLS]`/`[SEP]`: about 2.4 kb of non-repetitive sequence at ~4.7 bp per
token (more for repetitive DNA, less with `N`s, one token each). Longer inputs are rejected
with an explicit error, never truncated.

## Remote-code quirks handled here

- **transformers `>=4.57,<5`.** 5.x builds models on the meta device, where the remote ALiBi
  setup crashes, and the remote config lacks attributes 5.x expects. The config is loaded as
  transformers' own `BertConfig` and passed in explicitly (the known workaround for
  transformers > 4.28).
- **No triton.** torch's CUDA wheels pull in triton, and the remote code would then use its
  bundled Triton flash-attention kernel, written for the pre-2.0 Triton API. `pyproject.toml`
  excludes triton (`override-dependencies`), the loader skips transformers' import check for
  it, and the remote kernel handle is forced to `None`, so attention always runs in PyTorch.
  The remote code warns "Unable to import Triton; defaulting ... to pytorch" at load time,
  which is expected.
- **Half precision** needs autocast (the ALiBi bias is float32); `DNABERT2_DTYPE=bfloat16`
  runs under `torch.autocast`.

## Build and run

From the **repo root** (the build needs `packages/`):

```bash
make build S=dnabert2   # = docker build -f services/dnabert2/Dockerfile -t dnabert2-mcp:gpu .
docker run --rm --gpus all -p 8002:8000 -e MCP_AUTH_TOKEN=change-me \
  -v "$MODELS_DIR":/models dnabert2-mcp:gpu
```

DNABERT-2 weights are public (no `HF_TOKEN` needed) and released under the
**Apache-2.0** license (`LICENSE` in the model repo).

| Env var | Default (image) | |
|---|---|---|
| `DNABERT2_DEVICE` | `cuda` | auto-detected outside the image |
| `DNABERT2_DTYPE` | `float32` | `bfloat16` works via autocast |
| `DNABERT2_MAX_RESIDENT_MODELS` | `2` | LRU limit on loaded checkpoints |
| `DNABERT2_PLL_BATCH_SIZE` | `16` | masked copies per forward pass in `score_snp` |
| `MCP_AUTH_TOKEN`, `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT`, `MCP_PATH` | | same for all services |

Measured on an RTX PRO 4500 (fp32, GPU shared with other jobs): cold load 1.8 s; embedding a
batch of 8 x 2.3 kb takes 0.08 s; `score_snp` takes 0.5 s for 500 bp and 8-10 s for 2.3 kb
(~490 tokens). Peak process VRAM was 2.9 GiB, CUDA context included. Random sequence of
2.41-2.45 kb fits in 512 tokens.

## Tests

No downloads, no GPU: `make test S=dnabert2` from the repo root (or `uv run pytest` here).
