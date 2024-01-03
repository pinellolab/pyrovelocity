#!/usr/bin/env bash

set -euxo pipefail

# Update the filter/* branches
# Afterward:
#  - gt rs -r
#  - git checkout --theirs dvc.lock && gt add dvc.lock && gt continue

git fetch upstream

for branch in `git branch -a | grep remotes | grep -v HEAD | grep -E 'upstream/filter/[a-z]+$'`; do
   echo "Checking out ${branch#remotes/upstream/}"
   git checkout ${branch#remotes/upstream/}
   git reset --hard upstream/${branch#remotes/upstream/}
   gt btr -p main
done
