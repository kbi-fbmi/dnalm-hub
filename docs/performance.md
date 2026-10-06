# Performance: batching sequences

The biggest speedup for embeddings is to send many sequences in one call.
Pass a list to `embed_sequence`, or several `input` items to the gateway's
`/v1/embeddings`. Sending many concurrent single-sequence requests to one
service does not help. Each service runs its model calls one at a time
(`one_call_at_a_time`), so concurrent calls only wait in a queue.

```python
client.embed_sequence(["ACGT...", "TTGA...", ...])   # one call, many sequences
```

## Measurements

`make benchmark [S=ntv2]` (`scripts/benchmark.py`) embeds the same random sequences in
several ways and reports sequences per second. The default checkpoint of each family was
measured on an RTX PRO 4500 Blackwell (32 GB), 2026-10-05. Each service had the GPU to
itself (the other services were restarted before each run). Mean pooling, default layer.

| Model | Length | single | 8 threads | batches of 8 | one batch | gateway, float | gateway, base64 |
|---|---|---:|---:|---:|---:|---:|---:|
| ntv3/100m-pre | 600 bp × 64 | 56 | 23 | 235 | **516** | 213 | 454 |
| | 1.5 kb × 32 | 45 | 20 | 179 | **253** | 209 | 236 |
| | 30 kb × 8 | 15 | 14 | 15 | 15 | 15 | 15 |
| ntv2/500m | 600 bp × 64 | 36 | 18 | 141 | **164** | 145 | 166 |
| | 1.5 kb × 32 | 35 | 21 | 69 | 66 | 62 | 65 |
| dnabert2/117m | 600 bp × 64 | 79 | 28 | 208 | **396** | 296 | 377 |
| | 1.5 kb × 32 | 78 | 33 | 130 | **176** | 153 | 169 |
| hyenadna/medium-160k | 600 bp × 64 | 41 | 20 | 243 | **752** | 591 | 694 |
| | 1.5 kb × 32 | 42 | 17 | 166 | **329** | 276 | 320 |
| | 30 kb × 8 | 16 | 16 | 14 | 14 | 14 | 14 |
| evo2/7b | 600 bp × 64 | 11 | 11 | 11 | 11 | 10 | 10 |
| | 1.5 kb × 32 | 4.4 | 4.6 | 4.7 | 4.7 | 4.7 | 4.7 |
| | 30 kb × 8 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| generator/v2-eukaryote-1.2b | 600 bp × 64 | 32 | 18 | 172 | **257** | 162 | 257 |
| | 1.5 kb × 32 | 29 | 20 | 109 | **126** | 98 | 121 |
| | 30 kb × 8 | 7.0 | 7.1 | 6.5 | 6.5 | 6.4 | 6.4 |
| genalm/bert-large-t2t | 600 bp × 64 | 52 | 22 | 177 | **222** | 170 | 207 |
| | 1.5 kb × 32 | 50 | 25 | 89 | 81 | 74 | 80 |
| grover/grover | 600 bp × 64 | 106 | 28 | 140 | **300** | 257 | 352 |
| | 1.5 kb × 32 | 97 | 30 | 148 | **205** | 180 | 200 |

The 30 kb rows are missing for the 512-token models (NTv2, DNABERT-2, GENA-LM, GROVER):
those sequences are too long for them.

## What follows

- **Short sequences: batch.** At 600 bp, one call with 64 sequences is 3–18× faster
  than 64 single calls. The gain is largest for small models (HyenaDNA 18×, NTv3 9×,
  GENERator 8×). A single call spends most of its time on HTTP/MCP overhead and an
  underused GPU. Batches of 8 already give most of the gain for NTv2 and GENA-LM.
- **Long sequences: no gain.** At 30 kb one sequence already fills the GPU, so batching
  changes nothing. It also doesn't hurt, as long as the batch fits in GPU memory.
- **Evo 2 doesn't speed up.** Evo 2 7B runs each sequence as its own forward pass (see
  `services/evo2/src/evo2_mcp/inference.py`). Even so, at 600 bp it already computes at
  roughly 90 TFLOP/s, near the GPU's limit, so batching would not help either. Shorter
  sequences are the only way to make it faster.
- **Don't send concurrent requests to one service.** Eight threads are about 2× *slower*
  than one: the calls run one at a time anyway, and the extra connections only add
  overhead. Running different services in parallel is fine, as long as they fit in GPU
  memory together.
- **Gateway: use `encoding_format="base64"`.** JSON floats cost up to 2× for short
  sequences, where serializing the response dominates (NTv3: 213 vs. 454 seq/s).
  Base64 is about as fast as calling the service directly.

## GPU memory

All services share one GPU. A loaded model stays in memory until
`DNALM_IDLE_UNLOAD_SECONDS` (default 900 s) without calls. After each call, PyTorch's
cached memory above `DNALM_CUDA_CACHE_LIMIT_MB` (default 1024) is returned to the GPU.
Without that, one 30 kb batch could leave several GB reserved, and Evo 2 (about 13 GB)
would no longer fit. When the GPU is full, a call fails with a readable
"Out of GPU memory" error: send fewer or shorter sequences per call, or try again later.

GPU memory used after each benchmark run (model loaded, cache released):

| ntv3 | ntv2 | dnabert2 | hyenadna | evo2 | generator | genalm | grover |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.1 GB | 2.5 GB | 1.3 GB | 0.7 GB | 13.2 GB | 3.0 GB | 1.9 GB | 1.1 GB |
