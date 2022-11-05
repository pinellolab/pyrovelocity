# Conda environments

## minimal yaml configurations

We provide two example conda configuration files supporting [cpu](./environment-cpu.yml) and [gpu](./environment-gpu.yml) environments. Most examples will not run efficiently in the `cpu` environment, but this may still be useful for testing. These environments currently support `linux-64`. They may work on other operating systems and processor architectures.

```shell
mamba env create -n pyrovelocity -f environment-cpu.yaml
```

or

```shell
mamba env update -n pyrovelocity -f environment-cpu.yaml
```

## lockfile configurations

We use [conda-lock][conda-lock] to provide examples of explicit dependencies.
[conda-lock][conda-lock] can be installed with

```shell
mamba install -y conda-lock -c conda-forge
```

These cannot be guaranteed to work in your environment, but should generate an
equivalent environment to one in which `pyrovelocity` has previously been executed.

### generate lock files

The script [gen-conda-lock.sh](./gen-conda-lock.sh) depends upon [conda-lock] and can be used to generate `conda-linux-64-cpu.lock` and `conda/conda-linux-64-gpu.lock`. It may be necessary to edit [virtual-packages.yml](./virtual-packages.yml) to generate a configuration compatible with your system. See [managing virtual packages][conda-virt-packages] for reference. Either of the lock files can be used to reproduce a conda environment with:

```shell
conda create -n <env> --file <lockfile>
```

[conda-lock]: https://github.com/conda-incubator/conda-lock
[conda-virt-packages]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html#managing-virtual-packages
