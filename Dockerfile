FROM debian:bookworm-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/
ENV LIB_NAME='quoi'

RUN apt-get update && apt-get upgrade -y && apt-get clean && apt-get autoremove

COPY . ./src
WORKDIR /src

RUN uv sync --locked

RUN uv run ruff check ${LIB_NAME}/
RUN uv run pytest