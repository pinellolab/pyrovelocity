#!/usr/bin/env /bin/sh

set -x

# This script generates a conda lock file for the environment 
# yaml files in this directory.
#
# It depends upon conda-lock
#
#   mamba install -y -c conda-forge conda-lock
# 
# you can recreate an environment from a conda lock file with:
#
#   conda create -n <env> --file <lockfile>
# 
# for example
# 
# conda create --name YOURENV --file conda-linux-64-cpu.lock

conda-lock --kind explicit -f environment-cpu.yml
mv conda-linux-64.lock conda-linux-64-cpu.lock

conda-lock --kind explicit -f environment-gpu.yml
mv conda-linux-64.lock conda-linux-64-gpu.lock