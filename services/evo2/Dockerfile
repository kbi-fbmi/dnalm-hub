# syntax=docker/dockerfile:1
#
# GPU image for evo2-mcp (NVIDIA CUDA runtime + PyTorch's CUDA build). Build from
# the REPO ROOT so the shared packages/ are in the build context:
#
# Build:  docker build -f services/evo2/Dockerfile -t evo2-mcp:gpu .   (or: make build S=evo2)
# Run:    docker run --rm --gpus all -p 8004:8000 -e MCP_AUTH_TOKEN=change-me \
#           -v "$MODELS_DIR":/models evo2-mcp:gpu
#
# The checkpoints are public (no HF_TOKEN needed). The 7B needs ~16 GB of VRAM
# for weights + short inputs; see README for longer ones.

FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Mirror the repo layout (/app/services/evo2 next to /app/packages) so the
# relative path sources in pyproject.toml resolve unchanged.
WORKDIR /app/services/evo2
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY services/evo2/pyproject.toml services/evo2/uv.lock ./
COPY packages /app/packages
# --no-sources-package torch: drop the Linux -> CPU-wheel override so torch
# resolves from PyPI with CUDA (same as services/ntv3/Dockerfile).
RUN uv sync --no-install-project --no-dev --no-sources-package torch

COPY services/evo2/src ./src
COPY services/evo2/README.md ./
RUN uv sync --no-dev --no-sources-package torch

RUN if getent passwd 1000 >/dev/null; then \
        existing_user="$(getent passwd 1000 | cut -d: -f1)"; \
        usermod --login appuser --home /home/appuser --move-home "$existing_user"; \
    else \
        useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser; \
    fi

# torch>=2.14 routes some ops (e.g. outer-product bmm) to Triton kernels JIT-built
# at first use, which needs a C compiler + Python headers this runtime image lacks
# ("Failed to find C compiler"). Disable that and use the stock CUDA kernels.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    MCP_PATH=/mcp \
    EVO2_DEVICE=cuda \
    EVO2_DTYPE=bfloat16 \
    TORCH_DISABLE_NATIVE_JIT=1

RUN mkdir -p "$HF_HOME" && chown -R appuser:$(id -g appuser) /home/appuser "$HF_HOME"
USER appuser

EXPOSE 8000
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python3", "-c", "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('MCP_PORT','8000') + '/health', timeout=3)"]

ENTRYPOINT ["evo2-mcp"]
