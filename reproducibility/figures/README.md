# Pyro-Velocity reproducibility

We provide python scripts as well as a [DVC pipeline](https://dvc.org/doc/user-guide/pipelines) (see [dvc.yaml](./dvc.yaml)) for reproducing our results. If you would like to use this, you would need to install [data version control](https://dvc.org/doc/install) with [google cloud storage support](https://dvc.org/doc/install/linux#install-with-pip) (i.e. `pipx install dvc[gs]`). Please refer to our [manuscript](https://www.biorxiv.org/content/10.1101/2022.09.12.507691v2) for interpretation of the resulting figures.

## Latest version

```bash
mamba env create -n pv2 -f conda/environment-gpu-upd.yml
conda activate pv2
pip install xdoctest or mamba install -y xdoctest 
```

## Debugging development dependencies in flux

As new dependencies are being tested, you may find the environmented created by the [reproducibility/environment/Makefile](../environment/Makefile) is transiently out of sync with some scripts in this directory. For example, recently added dependencies including [hydra](https://hydra.cc/docs/intro/#installation) ([hydra-core](https://anaconda.org/conda-forge/hydra-core) in conda), [mlflow](https://mlflow.org/docs/latest/python_api/mlflow.html), and [matplotlib-venn](https://github.com/konstantint/matplotlib-venn) may require some variation of:

```bash
mamba install --freeze-installed \
  importlib_metadata=5.0.0 \
  mlflow=1.30.0 \
  hydra-core=1.2.0 \
  matplotlib-venn=0.11.7
```

## DVC and git

Please refer to the [DVC documentation](https://dvc.org/doc/command-reference) for all `dvc` commands. The following workflow is intended to ensure that you update the dvc remote cache with data files consistent with those referred to by the folder and file hashes in the `dvc.lock` file prior to committing with git. This ensures that others can `dvc checkout` your work.

After completing any updates to the library run pre-commit in the poetry environment (no need to include `poetry run` if conda is deactivated and poetry is active)

```bash
poetry run nox -x -rs pre-commit
```

to lint all files prior to dvc hashing. Then run the following commands:

```bash
dvc status
dvc repro # take note this will run the stages listed by `dvc status`
dvc push
```

This should update the local `dvc.lock` file and the remote DVC cache accordingly. You can then run the following sanity checks to evaluate this:

```bash
dvc pull # might update if someone else pushed since you last pulled
dvc push # Expected output: Everything is up to date
dvc status # Expected output: Everything is up to date
```

Now the content of your `dvc.lock` and remote dvc cache should be equivalent to the hashes of ALL the code and data dependencies of the dvc pipeline in `dvc.yaml`. You should then be able to commit everything and have someone else `dvc checkout && dvc pull` later with no need to run `dvc repro`.

Finally, note that if pre-commit hooks update files after running `dvc repro`, you will have to rerun the pipeline, so it is best to run pre-commit before the pipeline as described above.
