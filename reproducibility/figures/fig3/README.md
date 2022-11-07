Reproduce Figure 3 with model 2 for multi-fate cells and model 1 for uni-fate cells
=========================================================================================

After installation of [pyrovelocity](https://github.com/pinellolab/pyrovelocity), run the three scripts in order:

First, install cospar,

``` bash
pip install -r requirements.txt
```

Then, run the preparation scripts to generate intermediate pyrovelocity model results,

``` bash
# Model 1 for unipotent clones
python fig3_unipotent_mono_model1.py
python fig3_unipotent_neu_model1.py
# Model 2 for bifurcation and multi-fate clones
python fig3_uni_bifurcation_model2.py
python fig3_allcells_model2.py
```

Then, generate the benchmark standard using cospar and cytotrace,

``` bash
python fig3_allcells_cospar.py
python fig3_allcells_cytotrace.py
```

Last, run the visualization script,

``` bash
python fig3.py
```

For explaination of models and figure 3, please refer to our [manuscript](https://www.biorxiv.org/content/10.1101/2022.09.12.507691v2).
