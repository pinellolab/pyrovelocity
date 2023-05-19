#!/usr/bin/env bash

#########################
# To test uncommitted changes and/or individual pipeline stages
# replace `dvc repro` with something analogous to:
#     dvc exp run \
#     -S config.yaml:model_training.pancreas_model1.training_parameters.max_epochs=421 \
#     train_pancreas_model1
# TODO: remove npm update ... after container rebuild
# DONE: replace npx ... with cml
# npx github:iterative/cml#b923f3e pr create --skip-ci .
# https://github.com/iterative/cml/issues/1344 resolved in
# https://github.com/iterative/cml/releases/tag/v0.18.21
# TODO: move matplotlib-venn to package manager if persistent
# TODO: move /usr/bin/time install to Docker
#########################

set -x

### Execute experiment run and submit PR ###
python --version
pip install --upgrade pip
pip --version
pip install -e .
pip install matplotlib-venn==0.11.9
sudo apt-get update && sudo apt-get install -y time && which time
cd reproducibility/figures || exit

dvc pull
# dvc repro
./parallel.sh
dvc push

npm update -g @dvcorg/cml
cml pr create --skip-ci .
#########################

generate_markdown() {
    local data_set="$1"
    local model="$2"

    printf "\n# %s\n\n## %s\n\n### Metrics\n" "$data_set" "$model"
    dvc metrics show --md "models/${data_set}_${model}/metrics.json"

    echo "### Training plots"
    echo '!'"[ELBO](./models/${data_set}_${model}/loss_plot.png)"

    echo "### Run information"
    printf "\n\`\`\`json\n"
    cat "models/${data_set}_${model}/run_info.json"
    printf "\n\`\`\`\n"
}

### Post experiment report comment ###
{
# pipeline
printf "# pipeline\n"
dvc dag --md

data_sets=(
    "pancreas"
    "pons"
    "pbmc68k"
    "pbmc10k"
    "larry"
    "larry_mono"
    "larry_neu"
    "larry_multilineage"
    "larry_tips"
)
models=("model2")

for data_set in "${data_sets[@]}"; do
    for model in "${models[@]}"; do
        generate_markdown "$data_set" "$model"
    done
done

# pipeline files
printf "# pipeline files\n"
dvc dag -o --md
} >> report.md

cml comment update report.md
#########################

set +x
