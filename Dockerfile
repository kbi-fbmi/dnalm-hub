# syntax=docker/dockerfile:1
#
# CPU deployment image for ntv3-mcp. Builds a locked, reproducible environment
# with uv, then copies just the venv + source into a slim runtime image.
#
# On Linux, pyproject.toml's [tool.uv.sources] redirects the `torch` dependency
# to the CPU-only wheel index (see pyproject.toml), so this image does NOT pull
# the several-GB of CUDA runtime packages that a plain `pip install torch` would
# on Linux. For GPU inference, use Dockerfile.gpu instead.
#
# Build:  docker build -t ntv3-mcp .
# Run:    docker run --rm -p 8000:8000 -e HF_TOKEN=hf_xxx -e MCP_AUTH_TOKEN=change-me \
#           -v ntv3-hf-cache:/home/appuser/.cache/huggingface ntv3-mcp

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Install dependencies first so this (slow, large) layer is only rebuilt when
# pyproject.toml / uv.lock actually change, not on every source edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Now install the project itself.
COPY src ./src
COPY main.py README.md LICENSE ./
RUN uv sync --frozen --no-dev


FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/main.py /app/README.md /app/LICENSE ./

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    MCP_PATH=/mcp \
    NTV3_DEVICE=cpu \
    NTV3_DTYPE=float32

RUN mkdir -p "$HF_HOME" && chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 8000
VOLUME ["/home/appuser/.cache/huggingface"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('MCP_PORT','8000') + '/health', timeout=3)"]

ENTRYPOINT ["ntv3-mcp"]
