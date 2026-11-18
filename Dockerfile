# Build stage: resolve dependencies from uv.lock into a virtual environment
FROM python:3.12-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies change less often than the code, so they get their own layer
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


# Runtime stage: the virtual environment, configuration, and entry scripts only
FROM python:3.12-slim-trixie

RUN useradd --create-home --uid 1000 digitgan

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY config ./config
COPY scripts ./scripts

# Writable locations for MNIST, training runs, reports, and generated pictures. The API
# serves runs/cgan_seed42/generator.pt, so mount a runs directory or train inside first.
RUN mkdir -p data runs reports outputs \
    && chown -R digitgan:digitgan /app

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DIGITGAN_API_HOST=0.0.0.0

USER digitgan
EXPOSE 8000

CMD ["python", "scripts/run_api.py"]
