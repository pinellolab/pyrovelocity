from pyrovelocity.cytotrace import cytotrace_sparse
from pyrovelocity.data import load_larry


"""Analyzes LARRY data with cytotrace for model comparison.

Inputs:
  "data/larry.h5ad" via load_larry()

Outputs:
  data:
    "larry_invitro_adata_sub_raw_withcytotrace.h5ad"
"""


###########
# load data
###########

larry_invitro_adata_sub_raw = load_larry()


#############
# train model
#############

cytotrace_sparse(larry_invitro_adata_sub_raw, layer="spliced", skip_regress=False)


##################
# save checkpoints
##################

larry_invitro_adata_sub_raw.write("larry_invitro_adata_sub_raw_withcytotrace.h5ad")
