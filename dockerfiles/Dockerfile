FROM ghcr.io/iterative/cml:latest-gpu

LABEL maintainer="pyrovelocity team"
WORKDIR ${CML_RUNNER_PATH}
COPY . pyrovelocity/
RUN cd pyrovelocity && \
    pip install -e .
