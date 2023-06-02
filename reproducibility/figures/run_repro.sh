#!/bin/bash -ex

python config.py

dvc repro
