# Reproduce Figure 2 with model 1

After installation of [pyrovelocity](https://github.com/pinellolab/pyrovelocity), run the three scripts in order:

1st script is used to generate intermediate model results for pancreas data,

```bash
python fig2_pancreas_data.py
```

2nd script is used to generate intermediate model results for PBMC data,

```bash
python fig2_pbmc_data.py
```

Last, run the plot script to assemble the results:

```bash
python fig2.py
```

For explaination of model 1 and figure 2, please refer to our [manuscript](https://www.biorxiv.org/content/10.1101/2022.09.12.507691v2).
