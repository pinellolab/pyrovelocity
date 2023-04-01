# syntax=docker/dockerfile:1
FROM condaforge/mambaforge:22.11.1-4
LABEL maintainer="pyrovelocity team"

WORKDIR /pyrovelocity
COPY app/ .

RUN mamba init
RUN mamba env update -n base -f environment.yml

# install development packages not in environment.yml
# RUN mamba info && \
#     mamba install \
#     -c conda-forge \
#     -c bioconda \
#     -c nodefaults \
#     --yes \
#     python=3.10 && \
#     mamba info

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
