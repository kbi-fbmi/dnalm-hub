# hyenadna-mcp

MCP server for [HyenaDNA](https://github.com/HazyResearch/hyena-dna) (Nguyen et al., 2023),
long-context **causal** DNA language models from HazyResearch, pretrained on the human
reference genome (hg38) with one token per base. Sibling of [`ntv2-mcp`](../ntv2) and
[`ntv3-mcp`](../ntv3): same tool names, `checkpoint` parameter, response shapes, auth and
`/health`, built on the shared [`packages/`](../../packages).

## Checkpoints

The transformers ports (`-hf` repos) under [LongSafari](https://huggingface.co/LongSafari),
each pinned to a Hub commit in [`registry.py`](src/hyenadna_mcp/registry.py):

| Alias | Repo | Params | Layers x width | Context (bases) |
|---|---|---|---|---|
| `tiny-1k` | `LongSafari/hyenadna-tiny-1k-seqlen-hf` | 0.44M | 2 x 128 | 1,026 |
| `small-32k` | `LongSafari/hyenadna-small-32k-seqlen-hf` | 3.3M | 4 x 256 | 32,770 |
| `medium-160k` (default) | `LongSafari/hyenadna-medium-160k-seqlen-hf` | 6.6M | 8 x 256 | 160,002 |
| `medium-450k` | `LongSafari/hyenadna-medium-450k-seqlen-hf` | 6.6M | 8 x 256 | 450,002 |
| `large-1m` | `LongSafari/hyenadna-large-1m-seqlen-hf` | 6.6M | 8 x 256 | 1,000,002 |

The context is the config's `max_seq_len` (pretraining length + 2). This service adds no
special tokens, so the limit is in bases. Longer inputs are rejected with an explicit error,
never truncated. Any other HyenaDNA repo id (e.g. a fine-tune) works as `checkpoint` too,
unpinned.

## Tools

| Tool | Notes |
|---|---|
| `list_available_checkpoints` | the table above |
| `get_model_info`, `get_embedding_layers` | layers: 0 = token embedding, 1..n = each Hyena block's output, n+1 = final LayerNorm (`"last"`) |
| `embed_sequence` | one string or a list (batch; lengths may differ). `mean` or `per_token` (one row per base) |
| `compare_sequences` | cosine similarity of mean-pooled embeddings |
| `score_snp` | `method="whole_sequence_log_prob_delta"`, exact causal log-likelihood -- see below |
| `generate_sequence` | autoregressive sampling: `prompt`, `max_new_tokens`, `temperature` (0 = greedy), `top_k`, optional `seed` |

There is no `predict_masked_positions`, because HyenaDNA is causal and was never trained to fill in masks.

**Causal embeddings.** The embedding at base t depends only on bases 0..t. `mean` pooling
averages these prefix summaries, so it is weighted toward the start of the sequence. Only the
last `per_token` row has seen the whole input. The tokenizer's trailing `[SEP]` is not used.

**Batching.** Hyena has no attention mask. Its convolutions are causal, so right padding would
leave the real positions mathematically unchanged, but the FFT size and therefore the floats
would change. The server never pads. It groups a batch by exact length and caps each forward
pass at `HYENADNA_MAX_BATCH_BASES`, so batch results equal single-sequence results.

**`score_snp`** computes `log P(mutated) - log P(reference)`. Each is the exact autoregressive
log-likelihood `sum_t log P(x_t | x_<t)` (natural log), from one forward pass. The first base
has no context and is not scored, matching pretraining, which used no BOS token. Bases
downstream of the variant contribute too, so the amount of flanking sequence changes the
score. Scores are not comparable across models.

**`generate_sequence`** samples only A/C/G/T. The HF port has no recurrent/cached inference
mode, so each new base costs a full forward pass over the prefix. `prompt + max_new_tokens`
must fit the context.

## Build and run

From the **repo root** (the build needs `packages/`):

```bash
make build S=hyenadna      # = docker build -f services/hyenadna/Dockerfile -t hyenadna-mcp:gpu .
docker run --rm --gpus all -p 8003:8000 -e MCP_AUTH_TOKEN=change-me \
  -v "$MODELS_DIR":/models hyenadna-mcp:gpu
```

The checkpoints are public, so `HF_TOKEN` is not needed.

| Env var | Default (image) | |
|---|---|---|
| `HYENADNA_DEVICE` | `cuda` | |
| `HYENADNA_DTYPE` | `float32` | The remote code runs its FFT convolutions and residual stream in float32 anyway. The models are under 7M parameters, so half precision saves nothing worthwhile. |
| `HYENADNA_MAX_RESIDENT_MODELS` | `2` | LRU limit on loaded checkpoints |
| `HYENADNA_MAX_BASES` | unset | optional server-wide cap on bases per sequence, below the checkpoint limit (VRAM grows linearly with length) |
| `HYENADNA_MAX_BATCH_BASES` | `262144` | bases per forward pass when stacking equal-length sequences |
| `HYENADNA_RELEASE_CACHE_ABOVE_BASES` | `32768` | after inputs at least this long, return PyTorch's cached VRAM to the driver so the shared GPU gets it back |
| `HYENADNA_FFT_CHANNEL_CHUNK`, `HYENADNA_POINTWISE_CHUNK` | `32`, `65536` | chunk sizes of the memory-lean forward pass (see below) |
| `HYENADNA_MAX_NEW_TOKENS` | `2048` | upper bound for `generate_sequence`'s `max_new_tokens` |
| `MCP_AUTH_TOKEN`, `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT`, `MCP_PATH` | | same for all services |

**Memory.** The remote modeling code needs about 21 GiB of activations for a 1M-base forward
pass. It runs FFTs over all 256 channels at once and a 4x-wide MLP over the whole sequence.
This service runs the same blocks itself and reuses the remote `fftconv`, in channel and
position chunks, keeping only the requested hidden state. The outputs are bit-identical to
the remote forward at short lengths and agree to float32 rounding at 70k bases, at the same
speed. Peak memory is 2-3x lower. After large inputs the server returns cached VRAM to the
driver, and an OOM comes back as a readable error.

Measured on an RTX PRO 4500 (fp32, one sequence; the GPU was shared, so timings are noisy):

| Input | Peak allocated | embed / log-likelihood |
|---|---|---|
| 100k bases, `medium-160k` | 1.1 GiB | ~0.5-0.9 s |
| 160k bases, `medium-160k` | 1.8 GiB | ~0.6-2 s |
| 450k bases, `medium-450k` | 4.9 GiB | ~2-6 s |
| 1M bases, `large-1m` | 10.7 GiB (~12.4 GB process incl. CUDA context) | ~6 s |

A 50-base `generate_sequence` from a 100-base prompt takes about 5 s.

## Dependencies

The pinned remote code loads unchanged on current transformers. Its outputs were compared
bit for bit on 4.57.6 and 5.18.0, and trained filter frequencies load correctly on both.
The service therefore pins only `transformers>=4.57,<6`. The lock file and image use 5.x.

## License

The HyenaDNA weights and code are **BSD-3-Clause** (see the model cards). This server's own
code is MIT licensed.

## Tests

No downloads, no GPU: `make test S=hyenadna` from the repo root (or `uv run pytest` here).
