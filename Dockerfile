# Base image
FROM debian:bookworm-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/
ENV LIB_NAME='quoi'
RUN apt-get update && apt-get upgrade -y && apt-get clean && apt-get autoremove
COPY . ./src
WORKDIR /src
RUN uv sync --locked

# Lint and test
FROM base AS test
RUN uv run ruff check ${LIB_NAME}/
RUN uv run pytest

# Demo notebooks
FROM base AS demo
RUN uv pip install -e .
EXPOSE 8888
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
