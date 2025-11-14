FROM ghcr.io/astral-sh/uv:bookworm-slim

ADD . /app

WORKDIR /app
RUN uv sync --locked

CMD ["uv", "run", "main.py"]
