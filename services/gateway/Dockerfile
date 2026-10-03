# syntax=docker/dockerfile:1
#
# OpenAI-compatible embeddings gateway (no GPU, no model weights). Build from the
# REPO ROOT so the shared packages/ are in the build context:
#
# Build:  docker build -f services/gateway/Dockerfile -t dnalm-gateway .   (or: make build S=gateway)
# Run:    docker run --rm -p 8080:8080 -e MCP_AUTH_TOKEN=change-me \
#           -e GATEWAY_BACKENDS=ntv3=http://host.docker.internal:8000 dnalm-gateway

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Mirror the repo layout (/app/services/gateway next to /app/packages) so the
# relative path sources in pyproject.toml resolve unchanged.
WORKDIR /app/services/gateway
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY services/gateway/pyproject.toml services/gateway/uv.lock ./
COPY packages/dnalm-client /app/packages/dnalm-client
RUN uv sync --frozen --no-install-project --no-dev

COPY services/gateway/src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8080

USER appuser
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python3", "-c", "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('GATEWAY_PORT','8080') + '/health', timeout=3)"]

ENTRYPOINT ["dnalm-gateway"]
