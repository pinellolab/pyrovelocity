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

dvc stage list --name-only |\
    grep -E "summarize*" |\
    xargs -t -n 1 -P 4 bash -c 'sleep $((RANDOM % 6 + 3)); dvc repro "$@"' --

wait

dvc repro
