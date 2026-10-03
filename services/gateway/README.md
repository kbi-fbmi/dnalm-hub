# dnalm-gateway

OpenAI-compatible embeddings API over every dnalm-hub model service. Any client
that speaks the OpenAI embeddings API (the official SDKs, LangChain, LlamaIndex, ...)
can embed DNA with any model by changing only the `model` string.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="<MCP_AUTH_TOKEN>")
resp = client.embeddings.create(model="ntv3/100m-pre", input=["ACGTACGT...", "TTGACC..."])
vectors = [d.embedding for d in resp.data]
```

```bash
curl http://localhost:8080/v1/embeddings -H "Authorization: Bearer $MCP_AUTH_TOKEN" \
  -H "Content-Type: application/json" -d '{"model": "ntv2/500m", "input": "ACGTACGT"}'
```

## Endpoints

| Endpoint | |
|---|---|
| `POST /v1/embeddings` | one mean-pooled vector per input sequence |
| `GET /v1/models` | every checkpoint of every reachable service |
| `GET /v1/models/{id}` | one model |
| `GET /health` | liveness, no auth |

## Model ids

`<service>/<checkpoint>`, e.g. `ntv3/100m-pre`, `ntv3/100m-post`, `ntv2/2.5b-v1`. The
checkpoint part is passed to the service unchanged, so a full HuggingFace repo id works
too (`ntv2/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`). A bare alias or
repo id (`500m`, `InstaDeepAI/NTv3_100M_pre`) works when exactly one service lists it.

## Request fields

| Field | |
|---|---|
| `model`, `input` | `input` is one DNA sequence or a list of them (A/C/G/T/N). Token-id arrays are rejected. |
| `encoding_format` | `float` or `base64` (the OpenAI SDKs ask for base64 and decode it for you) |
| `dimensions` | not supported; vectors have the model's hidden size |
| `layer` | extra: hidden-state layer, `"last"` (default) or an index (see `get_embedding_layers`) |
| `species` | extra: for NTv3 post-trained checkpoints only (default `human`) |

Send the extra fields with `extra_body={"species": "mouse"}` in the OpenAI SDK.
`usage.prompt_tokens` counts bases, not model tokens.

Errors use the OpenAI error shape: 400 for invalid input (the service's own message,
e.g. invalid bases or a sequence over the context limit), 404 for an unknown model,
401 for a wrong key, 502 when a service is down. Sequences are never truncated.

## Configuration

| Env var | Default | |
|---|---|---|
| `GATEWAY_BACKENDS` | (required) | `name=url` pairs, e.g. `ntv3=http://ntv3:8000,ntv2=http://ntv2:8000`; the name is the model-id prefix |
| `MCP_AUTH_TOKEN` | | API key clients must send; also forwarded to the services |
| `GATEWAY_BACKEND_TOKEN` | `MCP_AUTH_TOKEN` | token for the services, if it differs |
| `GATEWAY_MAX_BATCH` | `32` | sequences per `embed_sequence` call; bigger inputs are split |
| `GATEWAY_TIMEOUT` | `600` | seconds per service call |
| `GATEWAY_HOST`, `GATEWAY_PORT` | `0.0.0.0`, `8080` (image) | |

`compose.yaml` runs it on port 8080 with every service configured. To add a model
family, append it to `GATEWAY_BACKENDS` there.

## Tests

```bash
make test S=gateway
```

The tests use fake services (no network, no weights), including a round trip through
the official `openai` SDK.
