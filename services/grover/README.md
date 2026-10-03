# grover-mcp

MCP server for [GROVER](https://huggingface.co/PoetschLab/GROVER) (PoetschLab, TU Dresden;
Sanabria et al., [Nature Machine Intelligence 2024](https://doi.org/10.1038/s42256-024-00872-0)):
a BERT-base masked DNA language model trained on the human genome (hg19) with a
byte-pair-encoding vocabulary from 600 merge cycles. Sibling of [`ntv2-mcp`](../ntv2): same
tool names, `checkpoint` parameter, response shapes, auth, and `/health`, built on the shared
[`packages/`](../../packages).

## Checkpoints

| Alias | Repo | Revision | Params | Context |
|---|---|---|---|---|
| `grover` (default) | `PoetschLab/GROVER` | `6b223110f0d6` | 87M | 512 tokens (~1.9 kb) |

PoetschLab publishes only this one pre-trained checkpoint (the same weights are on
[Zenodo](https://doi.org/10.5281/zenodo.8373117)). Any other repo id that loads with
`AutoModelForMaskedLM` and a GROVER-style BPE tokenizer (e.g. a fine-tune) can be passed as
`checkpoint`, unpinned.

## Tools

| Tool | Notes |
|---|---|
| `list_available_checkpoints` | the single `grover` alias, pinned to a Hub commit |
| `get_model_info`, `get_embedding_layers` | 13 hidden-state layers (0 = embeddings, 12 = last), hidden size 768 |
| `embed_sequence` | `per_token` rows are **BPE tokens** (`[CLS]`, tokens of 1-16 bases, `[SEP]`), not bases |
| `compare_sequences` | cosine similarity of mean-pooled embeddings |
| `score_snp` | `method="whole_sequence_log_prob_delta"` -- see below |

Not available, by design: `predict_masked_positions` (a BPE token is not one base, so
single-position masking is ill-defined) and `generate_sequence` (masked LM).

**Mean pooling** averages every non-padding token, `[CLS]` and `[SEP]` included.

**`score_snp`** tokenizes the reference and mutated sequences independently and reports
the difference of their masked-LM pseudo-log-likelihoods (mask each token in turn, sum the
log-probability of the true token). A SNP can change how the surrounding bases are merged into
tokens, so the two sums may cover different token counts. Costs about `2 x num_tokens`
forward passes, batched by `GROVER_PLL_BATCH_SIZE` (default 32). Scores are not comparable with
other services.

**Context limit:** 512 tokens including `[CLS]` and `[SEP]`. BPE compression depends on the
sequence (random sequence: ~3.7 bp/token, so ~1.9 kb fits; repeats compress better). Longer
inputs are rejected with an explicit error, never truncated.

**Input:** A/C/G/T/N, case-insensitive (input is uppercased; GROVER's vocabulary is uppercase
only). Every `N` becomes one `[UNK]` token: it costs a token, gets its own `per_token` row, and
is skipped by the pseudo-log-likelihood.

**Short sequences and edges:** the model card warns that BPE tokenization of sequences shorter
than ~50 bp differs from the genome-wide tokenization GROVER was trained on, and that edges are
affected too. It advises adding ~100 bp of genomic flank on each side. The server does not add
flanks for you.

## Build and run

From the **repo root** (the build needs `packages/`):

```bash
make build S=grover    # = docker build -f services/grover/Dockerfile -t grover-mcp:gpu .
docker run --rm --gpus all -p 8007:8000 -e MCP_AUTH_TOKEN=change-me \
  -v "$MODELS_DIR":/models grover-mcp:gpu
```

GROVER is public (no `HF_TOKEN` needed).

| Env var | Default (image) | |
|---|---|---|
| `GROVER_DEVICE` | `cuda` | |
| `GROVER_DTYPE` | `float32` | the model is small, so half precision saves little |
| `GROVER_MAX_RESIDENT_MODELS` | `2` | LRU limit on loaded checkpoints |
| `GROVER_PLL_BATCH_SIZE` | `32` | masked copies per forward pass in `score_snp` |
| `MCP_AUTH_TOKEN`, `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT`, `MCP_PATH` | | same for all services |

Measured on an RTX PRO 4500 (fp32, shared GPU): peak VRAM 0.86 GiB for the whole process;
embedding a max-length (512-token) sequence ~0.08 s (also for a batch of 8); `score_snp` on a
max-length sequence ~5 s.

## Environment

Stock `BertForMaskedLM` plus a `tokenizers` BPE `tokenizer.json`: no `trust_remote_code`, and
it runs on transformers 5.x (like ntv3-mcp). Two quirks are handled in `inference.py`:

- the Hub's `special_tokens_map.json` omits `[SEP]` and `[UNK]`, so transformers 5 leaves them
  out of `all_special_ids`. The service passes them explicitly when loading the tokenizer.
- the weights include an unused BERT pooler (`bert.pooler.*`), which transformers reports as
  "UNEXPECTED" keys on load. This is harmless.

## License

The HuggingFace model card states no license. PoetschLab's Zenodo release of the same weights
([10.5281/zenodo.8373117](https://doi.org/10.5281/zenodo.8373117)) is **CC-BY-4.0**:
attribution required, commercial use allowed. Cite Sanabria et al. 2024. This server's code is
MIT licensed.

## Tests

No downloads, no GPU: `make test S=grover` from the repo root (or `uv run pytest` here).
