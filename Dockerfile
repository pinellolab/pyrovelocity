FROM ghcr.io/iterative/cml:latest-gpu

WORKDIR ${CML_RUNNER_PATH}
COPY . pyrovelocity/
RUN cd pyrovelocity && \
    pip install -e .
