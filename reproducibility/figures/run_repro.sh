#!/usr/bin/env bash

set -euxo pipefail

ulimit -n 4096
python config.py
dvc repro
