import pyro
import scvelo as scv
import torch
from pyrovelocity.api import train_model

adata = scv.datasets.pancreas()
adata.layers["raw_unspliced"] = adata.layers["unspliced"]
adata.layers["raw_spliced"] = adata.layers["spliced"]
adata.obs["u_lib_size_raw"] = adata.layers["raw_unspliced"].toarray().sum(-1)
adata.obs["s_lib_size_raw"] = adata.layers["raw_spliced"].toarray().sum(-1)
adata = adata[:100]

adata_model_pos = train_model(
    adata,
    max_epochs=1,
    svi_train=False,
    log_every=1000,
    patient_init=45,
    batch_size=-1,
    use_gpu=0,
    include_prior=True,
    offset=True,
    library_size=True,
    patient_improve=1e-4,
    guide_type="auto",
    train_size=1.0,
)
# Model 2
demo_data = torch.ones(adata.shape[0], adata.shape[1]).to("cuda:0")
pyrovelocity_graph = pyro.render_model(
    adata_model_pos[0].module._model,
    model_args=(
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
    ),
    render_params=True,
    render_distributions=True,
    filename="suppfig1_graph_model2.dot",
)
# pyrovelocity_graph.unflatten(stagger=2)

# Model 1
adata_model_pos = train_model(
    adata,
    max_epochs=1,
    svi_train=False,
    log_every=1000,
    patient_init=45,
    batch_size=-1,
    use_gpu=0,
    include_prior=True,
    offset=False,
    library_size=True,
    patient_improve=1e-4,
    guide_type="auto_t0_constraint",
    train_size=1.0,
)

pyrovelocity_graph = pyro.render_model(
    adata_model_pos[0].module._model,
    model_args=(
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
    ),
    render_params=True,
    render_distributions=True,
    filename="suppfig1_graph_model1.dot",
)
# pyrovelocity_graph.unflatten(stagger=2)
