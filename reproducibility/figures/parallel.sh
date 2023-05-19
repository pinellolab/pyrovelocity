#!/usr/bin/env bash

set -euxo pipefail


dvc stage list --name-only |\
    grep -E "data_download*" |\
    xargs -t -n 1 -P 4 bash -c 'sleep $((RANDOM % 6 + 3)); dvc repro "$@"' --

wait

dvc stage list --name-only |\
    grep -E "preprocess*" |\
    xargs -t -n 1 -P 4 bash -c 'sleep $((RANDOM % 6 + 3)); dvc repro "$@"' --

wait

# dvc stage list --name-only |\
#     grep -E "train*" |\
#     xargs -t -n 1 -P 4 bash -c 'sleep $((RANDOM % 6 + 3)); dvc repro "$@"' --
dvc repro train@pancreas_model2 &
sleep 2
dvc repro train@pbmc68k_model2 &
sleep 2
dvc repro train@pons_model2 &
sleep 2
dvc repro train@larry_model2 &
wait
dvc repro train@larry_tips_model2 &
sleep 2
dvc repro train@larry_mono_model2 &
sleep 2
dvc repro train@larry_neu_model2 &
sleep 2
dvc repro train@larry_multilineage_model2 &
wait
dvc repro train@pbmc10k_model2 &

wait

dvc stage list --name-only |\
    grep -E "postprocess*" |\
    xargs -t -n 1 -P 4 bash -c 'sleep $((RANDOM % 6 + 3)); dvc repro "$@"' --

wait

#-----------------#

# dvc stage list --name-only | grep -E "data_download*" | xargs -t -n 1 -P 4 dvc repro
# dvc repro data_download@simulate_medium &
# dvc repro data_download@pons & 
# dvc repro data_download@pancreas &
# dvc repro data_download@pbmc68k &
# dvc repro data_download@larry &
# wait
# dvc repro data_download@larry_tips &
# dvc repro data_download@larry_mono &
# dvc repro data_download@larry_neu &
# dvc repro data_download@larry_multilineage &
# dvc repro data_download@pbmc10k &
# wait
#
# dvc stage list --name-only | grep -E "preprocess*" | xargs -t -n 1 -P 4 dvc repro
# dvc repro preprocess@simulate_medium &
# dvc repro preprocess@pons &
# dvc repro preprocess@pancreas &
# dvc repro preprocess@pbmc68k &
# dvc repro preprocess@larry &
# wait
# dvc repro preprocess@larry_tips &
# dvc repro preprocess@larry_mono &
# dvc repro preprocess@larry_neu &
# dvc repro preprocess@larry_multilineage &
# dvc repro preprocess@pbmc10k &
# wait