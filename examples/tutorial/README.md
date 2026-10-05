# dnalm-hub tutorial

Six notebooks that walk through the hub from first contact to downstream use. They are
clients only: the models run on the server (`docker compose up -d` in the repo root).

| Notebook | What you learn |
|---|---|
| [01 Getting started](01_getting_started.ipynb) | services, tools, checkpoints, a first embedding, errors, plain HTTP |
| [02 Embeddings](02_embeddings.ipynb) | batches, per-token output, layers, mean vs. single-position embeddings, a small classifier |
| [03 Variant effects](03_variant_effects.ipynb) | zero-shot SNP scores with every model, masked prediction, a mutation scan |
| [04 Generation](04_generation.ipynb) | continuing DNA with HyenaDNA, Evo 2 and GENERator; temperature; 6-mer lengths |
| [05 NTv3 species models](05_ntv3_annotation.ipynb) | genome annotation and experimental-track prediction around *HBB* |
| [06 OpenAI gateway](06_openai_gateway.ipynb) | embeddings with the OpenAI SDK or plain HTTP, model-specific options, errors |

## Running

From the repo root:

```bash
export MCP_HOST=http://localhost          # or your GPU server
export MCP_AUTH_TOKEN=...                 # the token from the server's .env
uv run --project packages/dnalm-client --with jupyterlab --with matplotlib --with openai \
    jupyter lab examples/tutorial
```

`tutorial_utils.py` holds the shared setup (one client per service, the test sequences,
a table printer). Notebooks 02, 05 fetch reference sequence from the UCSC Genome Browser
API, so they need internet access. The first call to a model downloads and loads it on the
server, which can take a few minutes for the large ones (Evo 2 7B, GENERator).

The saved outputs come from a run against all eight services on one RTX PRO 4500 (32 GB).
