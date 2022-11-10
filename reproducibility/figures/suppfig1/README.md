# Reproduce Supplementary Figure 1

The visualization of model graph structure needs pyro version >=1.8.1, please create an environment first:

```bash
conda create -n pyrovelocity_graph python=3.8.8
mamba env update -n pyrovelocity_graph -f environment.yml

cd ../../../ && python setup.py develop
```

Then, run the plot script:

```bash
cd && python suppfig1_models_graph.py
```

For explaination of model 1 and figure 2, please refer to our [manuscript](https://www.biorxiv.org/content/10.1101/2022.09.12.507691v2).
