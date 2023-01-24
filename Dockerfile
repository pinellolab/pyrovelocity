# Dockerfile

# Build container
FROM quay.io/condaforge/mambaforge:latest as conda

ADD conda/conda-lock.yml /locks/conda-lock.yml
# RUN conda create -p /opt/env --copy --file /locks/conda-lock.yml
RUN mamba info --envs
RUN mamba install -y conda-lock
RUN which conda-lock
RUN conda-lock install -p /opt/env --copy --file /locks/conda-lock.yml

# Primary container

FROM ghcr.io/iterative/cml:latest-gpu

COPY --from=conda /opt/env /opt/env
