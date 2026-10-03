# evo2-mcp

MCP server for Arc Institute's [Evo 2](https://github.com/ArcInstitute/evo2), a causal
(autoregressive) DNA language model trained on genomes from all domains of life, one
token per base. Sibling of [`ntv3-mcp`](../ntv3) and [`ntv2-mcp`](../ntv2): same tool
names, `checkpoint` parameter, response shapes, auth and `/health`, built on the shared
[`packages/`](../../packages).

It runs the community **pure-PyTorch transformers port**
[Aquiles-ai/Evo2-transformers](https://github.com/Aquiles-ai/Evo2-transformers)
(weights converted unchanged from Arc's checkpoints), not Arc's Vortex stack. No
Transformer Engine, FP8, FlashAttention or custom kernels are needed, so it runs on
any recent NVIDIA GPU, including Blackwell (sm_120). In exchange it is slower than
Vortex and memory hungry on long inputs, hence the length cap below. Treat it as
**experimental**: numbers can differ slightly from Arc's published ones.

> **License:** Evo 2 code and weights are Apache-2.0 (Arc Institute); the port and its
> converted weights on the Hub are Apache-2.0 too. Both are public (no `HF_TOKEN`).
> This server's code is MIT.

## Checkpoints

| Alias | Repo (pinned revision) | Params | Native context | Accepted here | Notes |
|---|---|---|---|---|---|
| `7b` (default) | [Aquiles-ai/Evo2-7B](https://huggingface.co/Aquiles-ai/Evo2-7B) @ `0838c72` | 6.58B | 1M | `EVO2_MAX_SEQ_LEN` (32,768) | = `evo2_7b`; robust in BF16 |

The port publishes two of Arc's eight checkpoints. Only the 7B is registered:

- [Aquiles-ai/Evo2-1B-Base](https://huggingface.co/Aquiles-ai/Evo2-1B-Base) (`evo2_1b_base`,
  8k) is **not registered**. Arc's README says the 1B (and 40B) need FP8 for accurate
  results, and this port has no FP8 path. Measured in BF16 on a 2 kb human HBB window,
  it scored -1.35 nats/base, barely better than the shuffled sequence (-1.39) and close to
  uniform (-1.386). The 7B scored -1.16. Its SNP deltas were also not meaningful. You can
  still pass the repo id as `checkpoint`, at your own risk.
- `evo2_40b`/`evo2_40b_base` (~80 GB in BF16) do not fit a 32 GB GPU.
- `evo2_7b_base`, `evo2_7b_262k` and `evo2_20b` have no published conversion. The port's
  `convert_evo2_vortex_to_hf.py` can make one, and any such repo id works as `checkpoint`.

## Tools

| Tool | Notes |
|---|---|
| `list_available_checkpoints` | aliases above, with `max_sequence_length` and `recommended_layer` |
| `get_model_info` | dtype, params, blocks (5 attention + 27 Hyena in the 7B), limits |
| `get_embedding_layers` | `recommended`: `blocks.28.mlp.l3`, `blocks.26.mlp.l3` (7B) |
| `embed_sequence` | one string or a list; `mean` or `per_token` (one row per base) |
| `compare_sequences` | cosine similarity of mean-pooled embeddings |
| `score_snp` | exact causal log-likelihood delta, `method="whole_sequence_log_prob_delta"` |
| `generate_sequence` | `prompt`, `max_new_tokens`, `temperature`, `top_k` (default 0 = off), optional `seed` |

No `predict_masked_positions`: Evo 2 is causal and has no mask token.

**Layers.** `layer_name` accepts `"recommended"` (the default for `embed_sequence` and
`compare_sequences`, unlike the other services' `"last"`), `"last"` (final layer after
the last RMSNorm), a hidden-state index (`0` = token embeddings, `i` = output of block
`i-1`, `32` = last), an Evo 2 layer name `blocks.N.mlp.l3` (block N's MLP output before
the residual add, the naming of Arc's `evo2` package). The Evo 2
authors embed from intermediate layers, not the last one: Arc's README uses
`blocks.28.mlp.l3` for `evo2_7b`, and the paper's sparse autoencoders read layer 26. That is
why `"recommended"` (= `blocks.28.mlp.l3` on the 7B) is the default here. The forward pass stops at the
requested layer, so earlier layers are also cheaper.

**score_snp** is Evo 2's own zero-shot variant-effect method: sum of
`log P(base_t | bases before t)` over bases 2..L for the mutated and the reference
sequence, and their difference. Negative = the alternative allele is less likely in
context. Arc's `score_sequences` reports the per-base mean; divide `score_delta` by
`length - 1` to compare. Give a few kb of context around the variant.

**generate_sequence** samples with the port's decoding cache, restricted to A/C/G/T.
`temperature=0` is greedy, and `top_k` 1-4 keeps the k most likely bases (0 = no
filtering). It returns `prompt`, `generated_sequence` (the new bases), `full_sequence`,
`num_new_tokens` / `num_new_bases` (always equal for Evo 2), `temperature`, `top_k` and `seed`.
Greedy decoding often falls into homopolymer runs (as with any DNA LM), so sample with
`temperature` around 1 for realistic sequence.

**Batches** run one sequence at a time, so a batch returns exactly the single results.
See "Port quirks" for why padding is avoided.

## Limits and measurements

All measured with the 7B in BF16 on an RTX PRO 4500 Blackwell (32 GB, sm_120), one
forward pass, PyTorch peak allocation (weights alone: 12.3 GiB):

| Length | log-likelihood | embedding (`blocks.28`) | peak VRAM |
|---|---|---|---|
| 2 kb | 0.3 s | 0.1 s | ~13 GiB (est.) |
| 16 kb | 2.9 s | 2.6 s | 15.3 GiB |
| 32 kb (default cap) | 6.4 s | 5.8 s | 18.2 GiB |
| 48 kb | 10.0 s | 9.0 s | 21.1 GiB |
| 64 kb | 13.4 s | 12.0 s | 23.9 GiB |
| 96 kb | OOM | | |

The default `EVO2_MAX_SEQ_LEN=32768` leaves room for other services on a shared GPU
(nvidia-smi showed ~21 GiB for the whole container at 32 kb). On a dedicated 32 GB GPU,
65536 works. `score_snp` runs two forward passes (12.8 s at 32 kb). Generation runs at
about 30 ms per new base after the prefill (100 bases from a 1 kb prompt: 3.2 s).

Sanity checks (2 kb human HBB window centered on the start codon):

- Mean log-likelihood: -1.16 nats/base (coding exon 1: -0.30), against -1.37 for the
  shuffled sequence and -1.386 for uniform over A/C/G/T.
- `score_snp` deltas: start codon ATG>ACG -28.4, sickle-cell HbS c.20A>T -20.4,
  synonymous CTG>CTA -10.7, four upstream/intronic substitutions between -1.0 and +1.1.
- Sampling from the start of exon 1 reproduces the real exon (`GTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGG...`)
  almost base for base.
- Cached decoding matches full recomputation: greedy outputs agree 64/64 with the
  argmax of a full forward pass.

**BF16 noise.** Per-base log-probabilities in BF16 move by up to ~0.4 nats with
mathematically irrelevant changes (sequence padding, FFT length, summation order). So
`score_snp` deltas below ~1 nat (sums over a whole window) are within noise. The
service runs each sequence alone without padding, so results are deterministic:
identical inputs give identical outputs, and a batch equals the single calls.

## Port quirks this service works around

- **Attention mask.** With an `attention_mask`, the port's attention calls SDPA with
  `is_causal=False`, so every position also sees the future (verified: an all-ones mask
  changes the log-probabilities). The service never passes a mask to a full forward
  pass. Without a mask, every mixer is causal. `model.generate()` always builds a mask
  for the prompt, so `generate_sequence` uses its own decoding loop: an unmasked prefill,
  then one-token steps with explicit `position_ids` and an all-ones mask over the cached
  keys.
- **fp32 tensors.** The checkpoint keeps the Hyena IIR poles/residues and the RoPE
  `inv_freq` in fp32, but `from_pretrained(dtype=bfloat16)` casts them to bf16. The
  service copies them back from the safetensors file in fp32 after loading.
- **Long-input memory.** The port builds each IIR long filter through an
  `(channels, 16, L)` fp32 temporary (8 GiB at 32 kb) and runs every FFT convolution
  over all 4096 channels at once. The service swaps in a term-by-term filter and a
  channel-chunked FFT (`EVO2_FFT_CHANNEL_CHUNK`, same math, covered by unit tests).
  Without these, 32 kb ran out of memory. The cached-generation prefill also loops over
  the prompt in Python for the IIR state, so long prompts take a while.
- **LM head precision.** The port's bf16 LM head rounds logits to steps of 0.5 (up to
  ±0.25 nats per base). Scoring and sampling use the same head in fp32 instead.

## Build and run

From the **repo root** (the build needs `packages/`):

```bash
make build S=evo2      # = docker build -f services/evo2/Dockerfile -t evo2-mcp:gpu .
docker run --rm --gpus all -p 8004:8000 -e MCP_AUTH_TOKEN=change-me \
  -v "$MODELS_DIR":/models evo2-mcp:gpu
```

The first call downloads the 7B (~14 GB) into `MODELS_DIR`.

| Env var | Default (image) | |
|---|---|---|
| `EVO2_DEVICE` | `cuda` | |
| `EVO2_DTYPE` | `bfloat16` | fp32 tensors stay fp32 either way; `float32` needs ~28 GB for the 7B weights |
| `EVO2_MAX_SEQ_LEN` | `32768` | longest accepted input (bases); longer is rejected, never truncated |
| `EVO2_MAX_NEW_TOKENS` | `2048` | upper bound for `generate_sequence` |
| `EVO2_MAX_OUTPUT_VALUES` | `16777216` | cap on `per_token` floats per call (4096 per base for the 7B) |
| `EVO2_FFT_CHANNEL_CHUNK` | `512` | channels per FFT convolution call (lower = less memory) |
| `EVO2_MAX_RESIDENT_MODELS` | `1` | LRU limit on loaded checkpoints |
| `MCP_AUTH_TOKEN`, `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT`, `MCP_PATH` | | same as the other services |

## Why a separate environment

The port's remote code uses the transformers 5.x hybrid cache API
(`cache_utils.LinearAttentionLayer`) and was written against 5.17, so this service pins
`transformers>=5.17,<6` (NTv2 needs `<5`).

## Tests

No downloads, no GPU: `make test S=evo2` from the repo root (or `uv run pytest` here).
The tests use a tiny stand-in model with the port's module names.
