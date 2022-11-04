import pyro
import torch

from pyrovelocity.api import train_model
from pyrovelocity.data import load_data


adata = load_data(top_n=2000, min_shared_counts=30)
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
    filename="suppfig1_graph_model2.pdf",
)
# pyrovelocity_graph.unflatten(stagger=2)


adata = load_data(top_n=2000, min_shared_counts=30)
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
    filename="suppfig1_graph_model1.pdf",
)
# pyrovelocity_graph.unflatten(stagger=2)
