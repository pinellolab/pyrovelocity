from pyrovelocity.cytotrace import cytotrace_sparse
import os
import cospar as cs
import scvelo as scv
from pyrovelocity.data import load_larry

cs.logging.print_version()
cs.settings.verbosity=2
cs.settings.data_path='LARRY_data'
cs.settings.figure_path='LARRY_figure'
cs.settings.set_figure_params(format='png',figsize=[4,3.5],dpi=75,fontsize=14,pointsize=2)

larry_invitro_adata_sub = load_larry()
scv.pp.filter_and_normalize(larry_invitro_adata_sub, min_shared_counts=30, n_top_genes=2000)
scv.pp.moments(larry_invitro_adata_sub, n_pcs=30, n_neighbors=30)

os.mkdir("LARRY_data")
os.mkdir("LARRY_figure")

larry_invitro_adata_sub = cs.tmap.infer_Tmap_from_multitime_clones(larry_invitro_adata_sub,
                                                                   compute_new=True,
                                                                   clonal_time_points=[2,4,6], later_time_point=6,
                                                                   smooth_array=[20,15,10], sparsity_threshold=0.2, max_iter_N=3)

cs.hf.save_map(larry_invitro_adata_sub)
