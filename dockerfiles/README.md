# Dockerfiles

## packages

### ubuntu

If you need to install system packages in a container derived from an ubuntu image

```Dockerfile
# install additional ubuntu system packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && \
    apt-get -qq install -y \
    build-essential \
    gfortran
```

### conda

If you need to install additional packages into a mamba environment

```Dockerfile
# install development packages not in environment.yml
RUN mamba info && \
    mamba install \
    -c conda-forge \
    -c bioconda \
    -c nodefaults \
    --yes \
    python=3.10 && \
    mamba info
```
