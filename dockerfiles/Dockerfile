# syntax=docker/dockerfile:1
# FROM ghcr.io/iterative/cml:0-dvc2-base1-gpu
FROM ghcr.io/iterative/cml@sha256:ad10a563de25311241f10d9d5509cecab6bc754b6b2c90b61e309e34fe80911e

WORKDIR ${CML_RUNNER_PATH}
COPY . pyrovelocity/
ENV PYENV_VERSION=3.10.10
RUN cd pyrovelocity && \
    ./.github/install-pyenv.sh $PYENV_VERSION && \
    export PYENV_ROOT="$HOME/.pyenv" && \
    export PATH="$PYENV_ROOT/bin:$PATH" && \
    eval "$(pyenv init -)" && \
    pip install --upgrade pip && \
    pip install -e .

LABEL org.opencontainers.image.title="pyrovelocity" \
      org.opencontainers.image.authors="pyrovelocity team" \
      org.opencontainers.image.description="This image contains the pyrovelocity library." \
      org.opencontainers.image.url="https://github.com/pinellolab/pyrovelocity" \
      org.opencontainers.image.licenses="AGPL-3.0-only"
