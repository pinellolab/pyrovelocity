# syntax=docker/dockerfile:1
FROM condaforge/mambaforge:23.1.0-1
LABEL maintainer="pyrovelocity team"


WORKDIR /pyrovelocity
# COPY . .
COPY pyproject.toml poetry.lock README.md ./
COPY pyrovelocity pyrovelocity
COPY app app


RUN mamba init
RUN mamba env update -n base -f app/environment.yml
RUN pip install --no-deps -e .

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
