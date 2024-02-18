FROM r-base:4.2.3

WORKDIR /root
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -q && \
    apt-get install -yq \
    build-essential \
    zlib1g-dev \
    libglpk-dev \
    libgmp-dev \
    libfontconfig1-dev \
    libxml2-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    curl \
    wget \
    git \
    htop \
    make \
    neovim \
    time \
    tree \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/dyngen .
RUN Rscript dyngen_install.R
