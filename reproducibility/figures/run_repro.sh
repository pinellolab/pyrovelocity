#!/bin/bash -ex

ulimit -n 4096

python config.py

dvc repro
