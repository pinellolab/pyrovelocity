{
  poetry2nix,
  poetry2nixOverrides,
  python3,
  gitignoreSource,
}:
poetry2nix.mkPoetryApplication {
  python = python3;
  groups = [];
  checkGroups = ["test"];
  projectDir = gitignoreSource ../../.;
  src = gitignoreSource ../../src;
  extras = [];
  overrides = poetry2nixOverrides;
  preferWheels = true;
  __darwinAllowLocalNetworking = true;

  nativeCheckInputs = [];

  preCheck = ''
    set -euo pipefail

    mkdir -p $TMPDIR/numba_cache
    export NUMBA_CACHE_DIR=$TMPDIR/numba_cache
    echo "NUMBA_CACHE_DIR: $NUMBA_CACHE_DIR"
  '';

  checkPhase = ''
    set -euo pipefail

    runHook preCheck

    # TODO: fix train_model test on Darwin
    # > src/pyrovelocity/train.py line 1564: 50991 Trace/BPT trap: 5
    pytest \
    -rA \
    -k "not workflows and not git" \
    -m "not pyensembl" \
    --xdoc \
    --no-cov \
    --disable-warnings \
    --pyargs pyrovelocity

    runHook postCheck
  '';

  doCheck = true;

  pythonImportsCheck = ["pyrovelocity"];
}
