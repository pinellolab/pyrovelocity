from pyrovelocity.cytotrace import cytotrace_sparse
from pyrovelocity.data import load_larry

larry_invitro_adata_sub_raw = load_larry()
cytotrace_sparse(larry_invitro_adata_sub_raw, layer='spliced', skip_regress=False)
larry_invitro_adata_sub_raw.write("larry_invitro_adata_sub_raw_withcytotrace.h5ad")
