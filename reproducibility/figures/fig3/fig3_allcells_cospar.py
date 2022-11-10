import os

import cospar as cs
import scvelo as scv

from pyrovelocity.cytotrace import cytotrace_sparse
from pyrovelocity.data import load_larry


"""Analyzes LARRY data with cospar for model comparison.

Inputs:
  "data/larry.h5ad" via load_larry()

Outputs:
  data:
    "LARRY_data/LARRY_MultiTimeClone_Later_FullSpace0_t*2.0*4.0*6_adata_with_transition_map.h5ad"
    "LARRY_data/LARRY_Similarity_matrix_with_all_cell_states_kNN20_Truncate0001_SM5.npz"
    "LARRY_data/LARRY_Similarity_matrix_with_all_cell_states_kNN20_Truncate0001_SM10.npz"
    "LARRY_data/LARRY_Similarity_matrix_with_all_cell_states_kNN20_Truncate0001_SM15.npz"
    "LARRY_data/LARRY_Similarity_matrix_with_all_cell_states_kNN20_Truncate0001_SM20.npz"
"""


###########
# load data
###########

cs.logging.print_version()
cs.settings.verbosity = 2
cs.settings.data_path = "LARRY_data"
cs.settings.figure_path = "LARRY_figure"
cs.settings.set_figure_params(
    format="png", figsize=[4, 3.5], dpi=75, fontsize=14, pointsize=2
)

larry_invitro_adata_sub = load_larry()
scv.pp.filter_and_normalize(
    larry_invitro_adata_sub, min_shared_counts=30, n_top_genes=2000
)
scv.pp.moments(larry_invitro_adata_sub, n_pcs=30, n_neighbors=30)

os.mkdir("LARRY_data")
os.mkdir("LARRY_figure")


#############
# train model
#############

larry_invitro_adata_sub = cs.tmap.infer_Tmap_from_multitime_clones(
    larry_invitro_adata_sub,
    compute_new=True,
    clonal_time_points=[2, 4, 6],
    later_time_point=6,
    smooth_array=[20, 15, 10],
    sparsity_threshold=0.2,
    max_iter_N=3,
)


##################
# save checkpoints
##################

# note that the file name will have the form:
#   file_name = f"{data_path}/{data_des}_adata_with_transition_map.h5ad"
cs.hf.save_map(larry_invitro_adata_sub)
