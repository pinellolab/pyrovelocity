# syntax=docker/dockerfile:1
FROM condaforge/mambaforge:23.1.0-1
LABEL maintainer="pyrovelocity team"


WORKDIR /pyrovelocity
COPY . .

RUN mamba init
RUN mamba env update -n base -f app/environment.yml
RUN pip install --no-deps -e .
RUN dvc stage list \
    --name-only reproducibility/figures/dvc.yaml | \
    grep -E "summarize" | \
    xargs -t -I {} dvc pull {}

CMD streamlit run app/app.py \
	--server.port=8080 \
	--server.enableCORS=false
