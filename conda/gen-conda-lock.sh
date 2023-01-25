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
#   conda-lock --conda mamba install -n YOURENV --file conda-lock.yml

# DEPRECATED
# conda-lock --kind explicit -f environment-cpu.yml
# mv conda-linux-64.lock conda-linux-64-cpu.lock
# conda-lock --kind explicit -f environment-gpu.yml
# mv conda-linux-64.lock conda-linux-64-gpu.lock

# UPDATED
# note that virtual packages are derived from ./virtual-packages.yml
# (re)move to disable
conda-lock --conda mamba --log-level DEBUG -f environment-cpu.yml --lockfile conda-lock-cpu.yml
conda-lock --conda mamba --log-level DEBUG -f environment-gpu.yml --lockfile conda-lock-gpu.yml
