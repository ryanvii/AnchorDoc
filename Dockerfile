FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY anchordoc ./anchordoc
COPY adapters ./adapters

RUN pip install --no-cache-dir ".[mcp]" \
    && mkdir -p /data \
    && useradd --create-home --shell /usr/sbin/nologin mcp \
    && chown -R mcp:mcp /app /data

USER mcp

ENTRYPOINT ["anchordoc", "mcp"]
CMD ["--root", "/data"]
