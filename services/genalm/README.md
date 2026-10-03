# genalm-mcp

MCP server for AIRI Institute's [GENA-LM](https://github.com/AIRI-Institute/GENA_LM)
DNA language models: BERT / BigBird masked LMs with a 32k BPE tokenizer, trained on the
T2T human genome (plus 1000 Genomes SNP augmentation, and multispecies genomes for one
variant). Same tool names, `checkpoint` parameter, response shapes, auth, and `/health`
as the other services, built on the shared [`packages/`](../../packages).

## Checkpoints

| Alias | Repo (`AIRI-Institute/...`) | Params | Context | Notes |
|---|---|---|---|---|
| `bert-large-t2t` (default) | `gena-lm-bert-large-t2t` | 336M | 512 tokens (~4.5 kb) | BERT-24L; best GENA-LM in the authors' downstream benchmarks |
| `bert-base-t2t` | `gena-lm-bert-base-t2t` | 110M | 512 tokens | BERT-12L, no final layernorm |
| `bert-base-lastln-t2t` | `gena-lm-bert-base-lastln-t2t` | 110M | 512 tokens | BERT-12L with final layernorm |
| `bert-base-t2t-multi` | `gena-lm-bert-base-t2t-multi` | 110M | 512 tokens | BERT-12L, human + multispecies training data |
| `bigbird-base-t2t` | `gena-lm-bigbird-base-t2t` | 110M | 4096 tokens (~36 kb) | transformers' native BigBird, block-sparse attention |

Each alias is pinned to a Hub commit (`registry.py`). Any other GENA-LM-style repo id
(remote-code BERT or native BigBird with an MLM head) can be passed as `checkpoint`.

Not registered: `gena-lm-bigbird-base-sparse(-t2t)` needs DeepSpeed sparse-attention ops
(Triton kernels, fp16 only), and `bigbird-base-t2t` already covers the 4096-token context.
`gena-lm-bert-base` is the authors' preliminary model, replaced by the `-t2t` ones. AIRI's newer
[ModernGENA](https://huggingface.co/AIRI-Institute/moderngena-base) is a different
architecture (ModernBERT) and is not part of this service.

**License:** the GENA-LM model cards declare no license. The
[GENA_LM code repository](https://github.com/AIRI-Institute/GENA_LM) is MIT licensed.
Check with the authors before commercial use. This server's code is MIT licensed.

## Tools

| Tool | Notes |
|---|---|
| `list_available_checkpoints` | aliases above, with `architecture` and `max_tokens` |
| `get_model_info`, `get_embedding_layers` | as in the other services, plus `max_tokens` |
| `embed_sequence` | one string or a list (batch); `per_token` rows are **BPE tokens**, not bases |
| `compare_sequences` | cosine similarity of mean-pooled embeddings |
| `score_snp` | `method="whole_sequence_log_prob_delta"`, see below |

Not available, by design: `predict_masked_positions` (a BPE token is not one base, so
single-position masking is ill-defined) and `generate_sequence` (masked LM).

**Tokens.** BPE tokens cover 1 to ~60 bases (~9 on average for human DNA, fewer for
high-entropy sequence). Each sequence gets `[CLS]` and `[SEP]`, and the tokenizer collapses
any run of 10 or more `N` into a single `-` token. `per_token` returns `[CLS]`, then one row
per token, then `[SEP]`. Mean pooling averages all of those rows (padding excluded).

**Context limit:** 512 tokens for BERT, 4096 for BigBird, `[CLS]`/`[SEP]` included.
Longer inputs are rejected with an explicit error, never truncated.

**`score_snp`** tokenizes the reference and mutated sequences independently (a SNP can
change how neighboring bases are tokenized) and reports the difference of their masked-LM
pseudo-log-likelihoods (mask each token in turn, sum the log-probability of the true token).
It costs about `2 x num_tokens` forward passes, batched as `GENALM_PLL_BATCH_TOKENS // num_tokens`
masked copies per pass. The MLM head runs only at the masked position, so full-vocabulary
logits are never materialized.

**Embedding layers.** `bert-base-t2t` and `bert-base-t2t-multi` are pre-layernorm models
*without* a final layernorm, so their last hidden state is the raw residual stream
(large magnitudes). Mean-pooled cosine similarity still works; normalize if you feed
the vectors to a distance-sensitive method.

**BigBird.** transformers' BigBird uses block-sparse attention only when the padded input is
longer than 704 tokens, and full attention otherwise. In a padded batch, a short sequence
would get a different attention pattern than on its own. So this service runs BigBird
sequences one per forward pass: batch results equal single results, but a batch is not faster.

## Build and run

From the **repo root** (the build needs `packages/`):

```bash
make build S=genalm      # = docker compose build genalm
docker run --rm --gpus all -p 8006:8000 -e MCP_AUTH_TOKEN=change-me \
  -v "$MODELS_DIR":/models genalm-mcp:gpu
```

The weights are public (no `HF_TOKEN` needed).

| Env var | Default (image) | |
|---|---|---|
| `GENALM_DEVICE` | `cuda` | |
| `GENALM_DTYPE` | `float32` | |
| `GENALM_MAX_RESIDENT_MODELS` | `2` | LRU limit on loaded checkpoints |
| `GENALM_PLL_BATCH_TOKENS` | `16384` | token budget per `score_snp` forward pass (32 x 512 for BERT, 4 x 4096 for BigBird) |
| `MCP_AUTH_TOKEN`, `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT`, `MCP_PATH` | | same for all services |

Measured on an RTX PRO 4500 (fp32). Embedding a max-length input takes under 0.2 s.
`score_snp` takes about 21 s for a ~500-token input on `bert-large-t2t` and about 9.5 min
for a ~2650-token (15 kb) input on `bigbird-base-t2t`. Peak VRAM is about 2.8 GiB for
`bert-large-t2t` and about 8.4 GiB for `bigbird-base-t2t` during long `score_snp` runs.

## Why a separate environment

GENA-LM's remote `modeling_bert.py` is code from the transformers 4.6 era. It imports
`find_pruneable_heads_and_indices` and `transformers.file_utils`, which transformers 5.x
removed. So this service pins `transformers>=4.57,<5`. 4.57 is the last 4.x release.

## Tests

No downloads, no GPU: `make test S=genalm` from the repo root (or `uv run pytest` here).
