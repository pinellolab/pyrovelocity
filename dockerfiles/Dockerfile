# syntax=docker/dockerfile:1
FROM ghcr.io/iterative/cml:0-dvc2-base1-gpu

LABEL maintainer="pyrovelocity team"
WORKDIR ${CML_RUNNER_PATH}
COPY . pyrovelocity/
RUN cd pyrovelocity && \
    pip install -e .
