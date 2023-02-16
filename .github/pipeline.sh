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
#########################

set -x

### Execute experiment run and submit PR ###
pip install -e .
cd reproducibility/figures || exit

dvc pull
dvc repro
dvc push

npm update -g @dvcorg/cml
cml pr create --skip-ci .
#########################


### Post experiment report comment ###
{
# pipeline
printf "# pipeline\n"
dvc dag --md

# pancreas
## model 1
printf "\n# pancreas\n\n## model 1\n\n### Metrics\n"
dvc metrics show --md models/pancreas_model1/metrics.json

# pipeline
printf "# pipeline\n"
dvc dag --md

# pancreas
## model 1
printf "\n# pancreas\n\n## model 1\n\n### Metrics\n"
dvc metrics show --md models/pancreas_model1/metrics.json

echo "### Training plots"
echo '!'"[ELBO](./models/pancreas_model1/loss_plot.png)"

echo "### Run information"
printf "\n\`\`\`json\n"
cat models/pancreas_model1/run_info.json
printf "\n\`\`\`\n"

# pancreas
## model 2
printf "## model 2\n\n### Metrics\n"
dvc metrics show --md models/pancreas_model2/metrics.json

echo "### Training plots"
echo '!'"[ELBO](./models/pancreas_model2/loss_plot.png)"

echo "### Run information"
printf "\n\`\`\`json\n"
cat models/pancreas_model2/run_info.json
printf "\n\`\`\`\n"

# pbmc68k
## model 1
printf "# pbmc68k\n\n## model 1\n\n### Metrics\n"
dvc metrics show --md models/pbmc68k_model1/metrics.json

echo "### Training plots"
echo '!'"[ELBO](./models/pbmc68k_model1/loss_plot.png)"

echo "### Run information"
printf "\n\`\`\`json\n"
cat models/pbmc68k_model1/run_info.json
printf "\n\`\`\`\n"

# pbmc68k
## model 2
printf "## model 2\n\n### Metrics\n"
dvc metrics show --md models/pbmc68k_model2/metrics.json

echo "### Training plots"
echo '!'"[ELBO](./models/pbmc68k_model2/loss_plot.png)"

echo "### Run information"
printf "\n\`\`\`json\n"
cat models/pbmc68k_model2/run_info.json
printf "\n\`\`\`\n"
} >> report.md

cml comment update report.md
#########################

set +x
