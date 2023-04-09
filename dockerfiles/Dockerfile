# syntax=docker/dockerfile:1
FROM ghcr.io/iterative/cml:0-dvc2-base1-gpu

LABEL maintainer="pyrovelocity team"
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
