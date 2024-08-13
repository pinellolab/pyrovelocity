FROM python:3.10.13-slim

WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -q && \
    apt-get install -yq \
    build-essential \
    zlib1g-dev \
    curl \
    git \
    make \
    tree \
    && rm -rf /var/lib/apt/lists/*

ENV VENV /opt/venv

RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

COPY pyproject.toml poetry.lock requirements-main.txt /root/

# requirements.txt auto-generated by poetry export
# see `make -n export_pip_requirements`
RUN pip install --upgrade pip && \
    pip install -r requirements-main.txt

COPY . /root

# development
RUN pip install --no-deps -e .
# distribution
# RUN pip install pyrovelocity==0.3.0b3

ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
