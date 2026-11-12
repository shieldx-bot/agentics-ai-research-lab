# Build from repository root:
# docker build -f docker/sandbox.Dockerfile -t agentics-sandbox:py311 .
FROM python:3.11-slim-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/agentics
COPY pyproject.toml README.md ./
COPY agentics ./agentics
RUN python -m pip install --no-cache-dir . "pytest>=8.0"

# The runner uses a rootless Docker daemon, drops every Linux capability,
# enables no-new-privileges, and runs as UID 0 inside the rootless user namespace.
# The repository workspace is mounted read-write at /workspace at runtime.
WORKDIR /workspace
CMD ["python", "-m", "pytest", "-q", "-x"]
