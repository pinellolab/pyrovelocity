# Pyro-Velocity reproducibility

We provide python scripts as well as a [DVC pipeline](https://dvc.org/doc/user-guide/pipelines) (see [dvc.yaml](./dvc.yaml)) for reproducing our results. If you would like to use this, you would need to install [hydra](https://hydra.cc/docs/intro/#installation) ([hydra-core](https://anaconda.org/conda-forge/hydra-core) in conda) and [data version control](https://dvc.org/doc/install) with [google cloud storage support](https://dvc.org/doc/install/linux#install-with-pip) (i.e. `pipx install dvc[gs]`). Please refer to our [manuscript](https://www.biorxiv.org/content/10.1101/2022.09.12.507691v2) for interpretation of the resulting figures.

Some fix on the vm images created by the [Makefile](https://github.com/pinellolab/pyrovelocity/tree/master/reproducibility/environment):

```bash
mamba install importlib_metadata==5.0
mamba install --freeze-installed mlflow==1.30.0
mamba install --freeze-installed hydra-core
mamba install matplotlib-venn
```

# Correct order of running dvc and git

Finish all code updates.
```bash
dvc status
dvc repro (but be careful if, for example, we’ve integrated LARRY it will try to rerun)
dvc push
```

Sanity checks,
```bash
dvc pull =⇒ might update if someone else pushed since you last pulled
dvc push =⇒ Everything is up to date
dvc status =⇒ Everything is up to date
now the content of your dvc.lock should be equivalent to the hashes of ALL the code and data deps of your dvc pipeline in dvc.yaml 
and you should be able to commit everything and have someone else checkout later with no need to repro.
git add . and git commit
```
