# Dockerfile

# Build container
FROM quay.io/condaforge/mambaforge:latest as conda

ADD conda/conda-lock.yml /locks/conda-lock.yml
RUN conda create -p /opt/env --copy --file /locks/conda-lock.yml

# Primary container

FROM ghcr.io/iterative/cml:latest-gpu

COPY --from=conda /opt/env /opt/env
