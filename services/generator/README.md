# generator-mcp

MCP server for GenerTeam's [GENERator](https://github.com/GenerTeam/GENERator) DNA language
models: Llama-architecture **causal** (autoregressive) models with a 6-mer tokenizer and a
16384-token (~98 kb) context ([v1 paper](https://arxiv.org/abs/2502.07272),
[v2 report](https://www.biorxiv.org/content/10.64898/2026.01.27.702015v1)). Same tool names,
`checkpoint` parameter, response shapes, auth and `/health` as the other services, built on
the shared [`packages/`](../../packages).

## Checkpoints

All checkpoints are public (not gated) and pinned to a Hub commit. The Hub stores fp32 weights:
about 4.6 GB for each 1.2B model and 12 GB for each 3B model.

| Alias | Repo | Params | Hidden / layers |
|---|---|---|---|
| `v2-eukaryote-1.2b` (default) | [GenerTeam/GENERator-v2-eukaryote-1.2b-base](https://huggingface.co/GenerTeam/GENERator-v2-eukaryote-1.2b-base) | 1.2B | 2048 / 26 |
| `v2-eukaryote-3b` | GenerTeam/GENERator-v2-eukaryote-3b-base | 3B | 3072 / 30 |
| `v2-prokaryote-1.2b` | GenerTeam/GENERator-v2-prokaryote-1.2b-base | 1.2B | 2048 / 26 |
| `v2-prokaryote-3b` | GenerTeam/GENERator-v2-prokaryote-3b-base | 3B | 3072 / 30 |
| `eukaryote-1.2b` | GenerTeam/GENERator-eukaryote-1.2b-base (v1) | 1.2B | 2048 / 26 |
| `eukaryote-3b` | GenerTeam/GENERator-eukaryote-3b-base (v1) | 3B | 3072 / 30 |

**No remote code.** The repos ship a `GENERatorForCausalLM` subclass of `LlamaForCausalLM`
and a `DNAKmerTokenizer`, both `trust_remote_code`. This server loads the same weights into
the stock `LlamaForCausalLM` and re-implements the tokenizer from `vocab.txt` in
[`kmer.py`](src/generator_mcp/kmer.py). On the GPU it produced the same token ids as the
remote tokenizer on 300 random sequences containing `N`.

## Tools

| Tool | Notes |
|---|---|
| `list_available_checkpoints` | the table above, plus `generation`, `domain`, `max_tokens` |
| `get_model_info`, `get_embedding_layers` | as in the other services; also `max_tokens`, `max_bases` |
| `embed_sequence` | `pooling`: `mean` / `last_token` / `per_token`; `remainder` policy (see below) |
| `compare_sequences` | cosine similarity of mean-pooled embeddings; `remainder` policy |
| `score_snp` | `method="whole_sequence_log_prob_delta"`, exact causal log-likelihood; any length |
| `generate_sequence` | continuation returned as bases; `sampling`, `seed`, `remainder` |

There is no `predict_masked_positions`. GENERator is causal and reads 6-mers, so masking a
single base is not defined.

### The 6-mer policy

GENERator reads DNA as non-overlapping 6-mers from the left. A 6-mer containing `N` becomes the
single `<oov>` token, as in the authors' tokenizer. The stock tokenizer would also turn a
leftover of 1-5 bases into `<oov>`, and the model cards warn against this: they left-truncate
(`seq[len(seq) % 6:]`) or left-pad with `A`. This server never drops bases silently:

- `embed_sequence`, `compare_sequences` and `generate_sequence` take `remainder`:
  - `"reject"` (default): a length that is not a multiple of 6 gets a clear error.
  - `"trim_left"`: the authors' recipe, which drops the first `len % 6` bases.
  - `"pad_left"`: prepends `A` bases.

  The number of bases trimmed or padded is returned in `remainder_bases`.
- `score_snp` accepts any length with no adjustment. The partial last 6-mer is marginalized
  exactly (see below).

### Embeddings

Embeddings follow the model cards. The model reads `<s>` + 6-mers. v2 checkpoints also get a
trailing `<s>` separator, as in the v2 card; v1 checkpoints don't. Batches are right-padded
and use an attention mask. The default layer is the final one (`hidden_states[-1]`, after the
last RMSNorm).

| `pooling` | Result |
|---|---|
| `mean` (default) | Average of all real tokens, including the `<s>` markers. This is the cards' mean-pooling option ("species-level information"). |
| `last_token` | The final token: the separator for v2, the last 6-mer for v1. This is the cards' option 1. |
| `per_token` | One row per **token**: `<s>`, one row per 6-mer, then the separator for v2. Rows are not bases: 90 bp gives 17 rows on v2 and 16 on v1. |

### `score_snp`

The reference and mutated sequences each get their exact log P(seq | `<s>`). This is the sum,
over tokens, of the full-vocabulary log-softmax at the true 6-mer. Two cases are marginalized:

- A 6-mer containing `N` scores the summed probability of all matching 6-mers. It is fed back
  to the model as `<oov>`.
- A trailing partial 6-mer of r bases scores the summed probability of all 6-mers that start
  with those r bases.

The LM head runs in fp32 on top of the bf16 decoder, because bf16 logits are too coarse for
small likelihood deltas. Each SNP costs two forward passes (1.2 s for 96 kb on the 1.2B
model). The authors' `score_sequence` helper reports per-base marginals instead, so its
numbers differ from these.

### `generate_sequence`

`generate_sequence` uses a KV-cached sampling loop. Only the 4096 DNA 6-mers can be
generated, never special tokens. The parameters work as follows:

- `max_new_tokens` counts **6-mer tokens**, so the continuation has 6 × that many bases. The
  maximum is `GENERATOR_MAX_NEW_TOKENS` (default 2048).
- `sampling="base"` (default) is the authors' `generate()`: the 6-mer distribution is reduced
  to per-base A/C/G/T marginals, and each base is chosen from its marginal.
  `sampling="token"` samples whole 6-mers.
- `temperature=0` gives greedy decoding.
- `top_k` keeps the k most likely 6-mers (0 means no filter). `GenericMcpClient` sends
  `top_k=4` by default, which is restrictive with 4096 tokens.

## Build and run

From the **repo root** (the build needs `packages/`):

```bash
make build S=generator   # = docker build -f services/generator/Dockerfile -t generator-mcp:gpu .
docker run --rm --gpus all -p 8005:8000 -e MCP_AUTH_TOKEN=change-me \
  -v "$MODELS_DIR":/models generator-mcp:gpu
```

| Env var | Default (image) | |
|---|---|---|
| `GENERATOR_DEVICE` | `cuda` | |
| `GENERATOR_DTYPE` | `bfloat16` | The authors' dtype. `float32` doubles VRAM (4.6 / 12 GB of weights). |
| `GENERATOR_MAX_RESIDENT_MODELS` | `1` | LRU limit on loaded checkpoints (a 3B takes about 6 GB in bf16) |
| `GENERATOR_MAX_NEW_TOKENS` | `2048` | upper bound for `generate_sequence` |
| `MCP_AUTH_TOKEN`, `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT`, `MCP_PATH` | | same for all services |

These numbers were measured on an RTX PRO 4500 Blackwell in bf16:

| Model | Peak VRAM (96 kb input) | Embed 96 kb | `score_snp` 96 kb | Generate 500 tokens (3 kb) |
|---|---|---|---|---|
| 1.2B | 6.7 GB | 0.6 s | 1.2 s | 11 s |
| 3B | 13.8 GB | 2.8 s | 5.7 s | 20 s |

**Precision.** In bf16, batch and single embeddings agree to cosine ≥ 0.99999. In fp32 they
are identical (relative error 2e-7).

## License

The GENERator weights are **MIT** licensed (HuggingFace model cards), and so is this
server's code.

## Tests

`make test S=generator` needs no downloads and no GPU. The tests use a tiny random-weight
Llama with GENERator's vocabulary. They check the exact log-likelihood (including `N` and
partial 6-mers), batch == single, per-token shapes, generation and the remainder policy.
