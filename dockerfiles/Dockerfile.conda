# Dockerfile

# Build container
FROM quay.io/condaforge/mambaforge:latest as conda

ADD conda/conda-lock.yml /locks/conda-lock.yml
RUN mamba info --envs && \
    mamba install -y conda-lock && \
    which conda-lock
RUN conda-lock install -p /opt/env --copy /locks/conda-lock.yml

# Primary container

FROM ghcr.io/iterative/cml:latest-gpu

COPY --from=conda /opt/env /opt/env
