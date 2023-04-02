# syntax=docker/dockerfile:1
FROM condaforge/mambaforge:22.11.1-4
LABEL maintainer="pyrovelocity team"


WORKDIR /pyrovelocity
# COPY . .
COPY pyproject.toml poetry.lock README.md ./
COPY pyrovelocity pyrovelocity
COPY app app

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && \
    apt-get -qq install -y \
    build-essential \
    gfortran
RUN mamba init
RUN mamba env update -n base -f app/environment.yml
RUN pip install -e .

# install development packages not in environment.yml
# RUN mamba info && \
#     mamba install \
#     -c conda-forge \
#     -c bioconda \
#     -c nodefaults \
#     --yes \
#     python=3.8 && \
#     mamba info

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
