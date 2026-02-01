FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install age for runtime dataset decryption
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -Lo /tmp/age.tar.gz https://github.com/FiloSottile/age/releases/download/v1.2.0/age-v1.2.0-linux-amd64.tar.gz \
    && tar -xzf /tmp/age.tar.gz -C /tmp \
    && mv /tmp/age/age /usr/local/bin/ \
    && rm -rf /tmp/age* \
    && apt-get remove -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY prompts/ /home/agent/prompts/
COPY mutation_data/ /home/agent/mutation_data/
COPY src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009