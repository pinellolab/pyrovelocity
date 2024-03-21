import contextlib
import importlib
import inspect
import io
import os
import sys
from inspect import getmembers
from numbers import Integral
from numbers import Real
from pathlib import Path
from pprint import pprint
from types import FunctionType
from types import ModuleType
from typing import Callable
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich.syntax
import rich.tree
import scvelo as scv
import seaborn as sns
import yaml
from anndata._core.anndata import AnnData
from beartype import beartype
from scvi.data import synthetic_iid

from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.logging import configure_logging


# import torch
# from scipy.sparse import issparse
# from sklearn.decomposition import PCA
# from torch.nn.functional import relu
# from pyrovelocity.models._transcription_dynamics import inv
# from pyrovelocity.models import mrna_dynamics

__all__ = [
    "anndata_counts_to_df",
    "attributes",
    "ensure_numpy_array",
    "filter_startswith_dict",
    "generate_public_api",
    "generate_sample_data",
    "internal_help",
    "mae",
    "mae_evaluate",
    "pretty_log_dict",
    "pretty_print_dict",
    "print_anndata",
    "print_attributes",
    "print_config_tree",
    "save_anndata_counts_to_dataframe",
    "str_to_bool",
]

logger = configure_logging(__name__)


def mae(pred_counts, true_counts):
    """
    Computes the mean average error between predicted counts and true counts.

    Args:
        pred_counts (np.ndarray): Predicted counts.
        true_counts (np.ndarray): True counts.

    Returns:
        float: Mean average error.

    Examples:
        >>> import numpy as np
        >>> pred_counts = np.array([[1, 2], [3, 4]])
        >>> true_counts = np.array([[2, 3], [4, 5]])
        >>> mae(pred_counts, true_counts)
        1.0
    """
    error = np.abs(true_counts - pred_counts).sum()
    total = pred_counts.shape[0] * pred_counts.shape[1]
    return (error / total).mean().item()


def ensure_numpy_array(obj):
    return obj.toarray() if hasattr(obj, "toarray") else obj


def mae_evaluate(posterior_samples, adata):
    maes_list = []
    labels = []
    if isinstance(posterior_samples, tuple):
        for model_label, model_obj, split_index in zip(
            ["Poisson train", "Poisson valid"],
            posterior_samples[:2],
            posterior_samples[2:],
        ):
            for sample in range(model_obj["u"].shape[0]):
                maes_list.append(
                    mae(
                        np.hstack(
                            [model_obj["u"][sample], model_obj["s"][sample]]
                        ),
                        np.hstack(
                            [
                                adata.layers["raw_unspliced"].toarray()[
                                    split_index
                                ],
                                adata.layers["raw_spliced"].toarray()[
                                    split_index
                                ],
                            ]
                        ),
                    )
                )
                labels.append(model_label)
    else:
        for model_label, model_obj in zip(
            ["Poisson all cells"], [posterior_samples]
        ):
            for sample in range(model_obj["u"].shape[0]):
                maes_list.append(
                    mae(
                        np.hstack(
                            [model_obj["u"][sample], model_obj["s"][sample]]
                        ),
                        np.hstack(
                            [
                                ensure_numpy_array(
                                    adata.layers["raw_unspliced"]
                                ),
                                ensure_numpy_array(adata.layers["raw_spliced"]),
                            ]
                        ),
                    )
                )
                labels.append(model_label)

    import pandas as pd

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 4)
    df = pd.DataFrame({"MAE": maes_list, "label": labels})
    sns.boxplot(x="label", y="MAE", data=df, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    # print(df.groupby("label").mean())
    return df


def attributes(obj):
    """
    get object attributes
    """
    disallowed_names = {
        name
        for name, value in getmembers(type(obj))
        if isinstance(value, FunctionType)
    }
    return {
        name: getattr(obj, name)
        for name in dir(obj)
        if name[0] != "_"
        and name not in disallowed_names
        and hasattr(obj, name)
    }


def print_attributes(obj):
    """
    print object attributes
    """
    pprint(attributes(obj))


@beartype
def pretty_log_dict(d: dict) -> str:
    dict_as_string = "\n"
    for key, value in d.items():
        # key_colored = colored(key, "green")
        key_colored = key
        value_lines = str(value).split("\n")
        value_colored = "\n".join(
            # colored(line, "white") for line in value_lines
            line
            for line in value_lines
        )
        dict_as_string += f"{key_colored}:\n{value_colored}\n"
    return dict_as_string


@beartype
def pretty_print_dict(d: dict):
    logger.info(pretty_log_dict(d))


@beartype
def print_config_tree(
    config: dict,
    name: str = "",
    theme: str = "nord-darker",
    max_width: int = 148,
):
    config_yaml = yaml.dump(config, default_flow_style=False)
    tree = rich.tree.Tree(name, style="dim", guide_style="dim")
    tree.add(rich.syntax.Syntax(config_yaml, "yaml", theme=theme))

    console = rich.console.Console(width=max_width)
    console.print(tree)


@beartype
def print_docstring(
    obj: Callable | ModuleType,
    theme: str = "nord-darker",
):
    docstring = inspect.getdoc(obj)
    if docstring is None:
        rich.print("[bold red]No docstring found for this object.[/bold red]")
        return
    syntax_highlighted_docstring = rich.syntax.Syntax(
        docstring, "python", theme=theme, line_numbers=False
    )
    rich.print(syntax_highlighted_docstring)


@beartype
def str_to_bool(value: str | bool, default: bool = False) -> bool:
    """
    Convert strings that could be interpreted as booleans to a boolean value,
    with a default fallback.

    Args:
        value (str | bool): input string or boolean value.
        default (bool, optional): Defaults to False.

    Returns:
        bool: boolean interpretation of the input string
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "1", "yes", "y"):
        return True
    elif value.lower() in ("false", "f", "0", "no", "n"):
        return False
    else:
        return default


def filter_startswith_dict(dictionary_with_underscore_keys):
    """Remove entries from a dictionary whose keys start with an underscore.

    Args:
        dictionary_with_underscore_keys (dict): Dictionary to be filtered.

    Returns:
        dict: Filtered dictionary.
    """
    return {
        k: v
        for k, v in dictionary_with_underscore_keys.items()
        if not k.startswith("_")
    }


def print_anndata(anndata_obj):
    """
    Print a formatted representation of an AnnData object.

    This function produces a custom output for the AnnData object with each
    element of obs, var, uns, obsm, varm, layers, obsp, varp indented and
    displayed on a new line.

    Args:
        anndata_obj (AnnData): The AnnData object to be printed.

    Raises:
        AssertionError: If the input object is not an instance of AnnData.

    Examples:
        >>> import anndata
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(42)
        >>> X = np.random.randn(10, 5)
        >>> obs = pd.DataFrame({"clusters_coarse": np.random.randint(0, 2, 10),
        ...                     "clusters": np.random.randint(0, 2, 10),
        ...                     "S_score": np.random.rand(10),
        ...                     "G2M_score": np.random.rand(10)})
        >>> var = pd.DataFrame({"gene_name": [f"gene_{i}" for i in range(5)]})
        >>> adata = AnnData(X, obs=obs, var=var)
        >>> print_anndata(adata)  # doctest: +NORMALIZE_WHITESPACE
    """
    assert isinstance(
        anndata_obj, AnnData
    ), "Input object must be of type AnnData."

    def format_elements(elements):
        formatted = "\n".join([f"        {elem}," for elem in elements])
        return formatted

    anndata_string = [
        f"\nAnnData object with n_obs × n_vars = {anndata_obj.n_obs} × {anndata_obj.n_vars}"
    ]

    properties = {
        "obs": anndata_obj.obs.columns,
        "var": anndata_obj.var.columns,
        "uns": anndata_obj.uns.keys(),
        "obsm": anndata_obj.obsm.keys(),
        "varm": anndata_obj.varm.keys(),
        "layers": anndata_obj.layers.keys(),
        "obsp": anndata_obj.obsp.keys(),
        "varp": anndata_obj.varp.keys(),
    }

    for prop_name, elements in properties.items():
        if len(elements) > 0:
            anndata_string.append(
                f"    {prop_name}:\n{format_elements(elements)}"
            )

    logger.info("\n".join(anndata_string))


def generate_sample_data(
    n_obs: int = 100,
    n_vars: int = 12,
    alpha: float = 5,
    beta: float = 0.5,
    gamma: float = 0.3,
    alpha_: float = 0,
    noise_model: str = "gillespie",
    random_seed: int = 0,
) -> AnnData:
    """
    Generate synthetic single-cell RNA sequencing data with spliced and unspliced layers.
    If using the "iid" noise model, the data will be generated with scvi.data.synthetic_iid.
    If using the "normal" or "gillespie" noise model, the data will be generated with
    scvelo.datasets.simulation accounting for the given expression dynamics parameters.

    Args:
        n_obs (int, optional): Number of observations (cells). Default is 100.
        n_vars (int, optional): Number of variables (genes). Default is 12.
        alpha (float, optional): Transcription rate. Default is 5.
        beta (float, optional): Splicing rate. Default is 0.5.
        gamma (float, optional): Degradation rate. Default is 0.3.
        alpha_ (float, optional): Additional transcription rate. Default is 0.
        noise_model (str, optional): Noise model to be used. Must be one of 'iid', 'gillespie', or 'normal'. Default is 'gillespie'.
        random_seed (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        AnnData: An AnnData object containing the generated synthetic data.

    Raises:
        ValueError: If noise_model is not one of 'iid', 'gillespie', or 'normal'.

    Examples:
        >>> from pyrovelocity.utils import generate_sample_data, print_anndata
        >>> adata = generate_sample_data(random_seed=99)
        >>> print_anndata(adata)
        >>> adata = generate_sample_data(n_obs=50, n_vars=10, noise_model="normal", random_seed=99)
        >>> print_anndata(adata)
        >>> adata = generate_sample_data(n_obs=50, n_vars=10, noise_model="iid", random_seed=99)
        >>> print_anndata(adata)
        >>> adata = generate_sample_data(noise_model="wishful thinking")
        Traceback (most recent call last):
            ...
        ValueError: noise_model must be one of 'iid', 'gillespie', 'normal'
    """
    if noise_model == "iid":
        adata = synthetic_iid(
            batch_size=n_obs,
            n_genes=n_vars,
            n_batches=1,
            n_labels=1,
        )
        adata.layers["spliced"] = adata.X.copy()
        adata.layers["unspliced"] = adata.X.copy()
    elif noise_model in {"gillespie", "normal"}:
        adata = scv.datasets.simulation(
            random_seed=random_seed,
            n_obs=n_obs,
            n_vars=n_vars,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            alpha_=alpha_,
            noise_model=noise_model,
        )
    else:
        raise ValueError(
            "noise_model must be one of 'iid', 'gillespie', 'normal'"
        )
    return adata


@beartype
def anndata_counts_to_df(
    adata: AnnData,
) -> Tuple[pd.DataFrame, Integral, Integral, Real, Real]:
    spliced_df = pd.DataFrame(
        ensure_numpy_array(adata.layers["raw_spliced"]),
        index=list(adata.obs_names),
        columns=list(adata.var_names),
    )
    unspliced_df = pd.DataFrame(
        ensure_numpy_array(adata.layers["raw_unspliced"]),
        index=list(adata.obs_names),
        columns=list(adata.var_names),
    )

    spliced_melted = spliced_df.reset_index().melt(
        id_vars="index", var_name="var_name", value_name="spliced"
    )
    unspliced_melted = unspliced_df.reset_index().melt(
        id_vars="index", var_name="var_name", value_name="unspliced"
    )

    df = spliced_melted.merge(unspliced_melted, on=["index", "var_name"])

    df = df.rename(columns={"index": "obs_name"})

    total_obs = adata.n_obs
    total_var = adata.n_vars

    max_spliced = adata.layers["raw_spliced"].max()
    max_unspliced = adata.layers["raw_unspliced"].max()

    logger.warning(
        f"Type of df is {type(df)}\n"
        f"Type of total_obs is {type(total_obs)}\n"
        f"Type of total_var is {type(total_var)}\n"
        f"Type of max_spliced is {type(max_spliced)}\n"
        f"Type of max_unspliced is {type(max_unspliced)}\n"
    )

    return (
        df,
        total_obs,
        total_var,
        max_spliced,
        max_unspliced,
    )


# TODO: migrate to parquet
@beartype
def save_anndata_counts_to_dataframe(
    adata: AnnData,
    dataframe_path: os.PathLike | str,
) -> Path:
    logger.info(f"Saving AnnData object to dataframe: {dataframe_path}")
    df_tuple = anndata_counts_to_df(adata)

    CompressedPickle.save(dataframe_path, df_tuple)

    if not os.path.isfile(dataframe_path) or not os.access(
        dataframe_path, os.R_OK
    ):
        raise FileNotFoundError(f"Failed to create readable {dataframe_path}")

    return Path(dataframe_path)


@beartype
def generate_public_api(module_name: str) -> List[str]:
    """
    Generates a list of names of functions and classes defined in a specified
    module. This may be used, for example, to generate a candidate for the
    public interface for a module or package.

    Args:
        module_name (str): The name of the module.

    Returns:
        List[str]: A list of names of functions and classes defined in the module.
    """
    module = importlib.import_module(module_name)
    public_api: List[str] = [
        name
        for name, obj in inspect.getmembers(
            module,
            lambda obj: (inspect.isfunction(obj) or inspect.isclass(obj))
            and obj.__module__ == module.__name__,
        )
    ]
    return public_api


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def internal_help(obj: Callable | ModuleType):
    """Generate help text for a callable or module.

    Args:
        obj (Callable | ModuleType): The object to generate help text for.
    """
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        help(obj)

    help_text = captured_output.getvalue()
    lines = help_text.split("\n")

    processed_lines = []
    for line in lines:
        if line.startswith("FILE"):
            break
        processed_lines.append(line)

    print("\n".join(processed_lines))


# TODO: remove unused functions
# def log(x):
#     """
#     Computes the element-wise natural logarithm of a tensor, while clipping
#     its values to avoid numerical instability.

#     Args:
#         x (torch.Tensor): Input tensor.

#     Returns:
#         torch.Tensor: Tensor with element-wise natural logarithm of x.

#     Examples:
#         >>> import torch
#         >>> x = torch.tensor([0.0001, 0.5, 0.9999])
#         >>> log(x)
#         tensor([-9.2103e+00, -6.9315e-01, -1.0002e-04])
#     """
#     eps = torch.finfo(x.dtype).eps
#     return torch.log(x.clamp(eps, 1 - eps))


# TODO: remove unused functions
# def tau_inv(
#     u=None, s=None, u0=None, s0=None, alpha=None, beta=None, gamma=None
# ):
#     """
#     Computes the inverse tau given the parameters and initial conditions.

#     Args:
#         u (torch.Tensor): Value of u.
#         s (torch.Tensor): Value of s.
#         u0 (torch.Tensor): Initial value of u.
#         s0 (torch.Tensor): Initial value of s.
#         alpha (torch.Tensor): Alpha parameter.
#         beta (torch.Tensor): Beta parameter.
#         gamma (torch.Tensor): Gamma parameter.

#     Returns:
#         torch.Tensor: Inverse tau.

#     Examples:
#         >>> import torch
#         >>> u = torch.tensor(0.6703)
#         >>> s = torch.tensor(0.4596)
#         >>> u0 = torch.tensor(1.0)
#         >>> s0 = torch.tensor(0.5)
#         >>> alpha = torch.tensor(0.5)
#         >>> beta = torch.tensor(0.4)
#         >>> gamma = torch.tensor(0.3)
#         >>> tau_inv(u, s, u0, s0, alpha, beta, gamma)
#         tensor(3.9736e-07)
#     """
#     beta_ = beta * inv(gamma - beta)
#     xinf = alpha / gamma - beta_ * (alpha / beta)
#     tau1 = (
#         -1.0 / gamma * log((s - beta_ * u - xinf) * inv(s0 - beta_ * u0 - xinf))
#     )

#     uinf = alpha / beta
#     tau2 = -1 / beta * log((u - uinf) * inv(u0 - uinf))
#     tau = torch.where(beta > gamma, tau1, tau2)
#     return relu(tau)


# TODO: remove unused functions
# def init_with_all_cells(
#     adata,
#     input_type="knn",
#     shared_time=True,
#     latent_factor="linear",
#     latent_factor_size=10,
#     plate_size=2,
#     num_aux_cells=200,
#     init_smooth=True,
# ):
#     ## hard to use unsmoothed data for initialization
#     ## always initialize the model with smoothed data
#     if "Mu" in adata.layers and "Ms" in adata.layers and init_smooth:
#         u_obs = torch.tensor(adata.layers["Mu"], dtype=torch.float32)
#         s_obs = torch.tensor(adata.layers["Ms"], dtype=torch.float32)
#     elif "spliced" in adata.layers:
#         u_obs = torch.tensor(
#             adata.layers["unspliced"].toarray()
#             if issparse(adata.layers["unspliced"])
#             else adata.layers["unspliced"],
#             dtype=torch.float32,
#         )
#         s_obs = torch.tensor(
#             adata.layers["spliced"].toarray()
#             if issparse(adata.layers["spliced"])
#             else adata.layers["spliced"],
#             dtype=torch.float32,
#         )
#     else:
#         raise

#     ub_u = torch.stack(
#         [torch.quantile(u[u > 0], 0.99) for u in torch.unbind(u_obs, axis=1)]
#     )
#     ub_s = torch.stack(
#         [torch.quantile(s[s > 0], 0.99) for s in torch.unbind(s_obs, axis=1)]
#     )
#     s_mask = (s_obs > 0) & (s_obs <= ub_s)
#     u_mask = (u_obs > 0) & (u_obs <= ub_u)
#     # include zeros
#     training_mask = s_mask & u_mask

#     u_scale = torch.stack(
#         [u[u > 0].std() for u in torch.unbind(u_obs * training_mask, axis=1)]
#     )
#     s_scale = torch.stack(
#         [s[s > 0].std() for s in torch.unbind(s_obs * training_mask, axis=1)]
#     )
#     scale = u_scale / s_scale

#     lb_steady_u = torch.stack(
#         [
#             torch.quantile(u[u > 0], 0.98)
#             for u in torch.unbind(u_obs * training_mask, axis=1)
#         ]
#     )
#     lb_steady_s = torch.stack(
#         [
#             torch.quantile(s[s > 0], 0.98)
#             for s in torch.unbind(s_obs * training_mask, axis=1)
#         ]
#     )

#     steady_u_mask = training_mask & (u_obs >= lb_steady_u)
#     steady_s_mask = training_mask & (s_obs >= lb_steady_s)

#     u_obs = u_obs / scale
#     u_inf = (u_obs * (steady_u_mask | steady_s_mask)).sum(dim=0) / (
#         steady_u_mask | steady_s_mask
#     ).sum(dim=0)
#     s_inf = (s_obs * steady_s_mask).sum(dim=0) / steady_s_mask.sum(dim=0)

#     gamma = (u_obs * steady_s_mask * s_obs).sum(axis=0) / (
#         (steady_s_mask * s_obs) ** 2
#     ).sum(axis=0) + 1e-6
#     gamma = torch.where(gamma < 0.05 / scale, gamma * 1.2, gamma)
#     gamma = torch.where(gamma > 1.5 / scale, gamma / 1.2, gamma)
#     alpha = gamma * s_inf
#     beta = alpha / u_inf

#     switching = tau_inv(u_inf, s_inf, 0.0, 0.0, alpha, beta, gamma)
#     tau = tau_inv(u_obs, s_obs, 0.0, 0.0, alpha, beta, gamma)
#     tau = torch.where(tau >= switching, switching, tau)
#     tau_ = tau_inv(u_obs, s_obs, u_inf, s_inf, 0.0, beta, gamma)
#     tau_ = torch.where(
#         tau_ >= tau_[s_obs > 0].max(dim=0)[0],
#         tau_[s_obs > 0].max(dim=0)[0],
#         tau_,
#     )
#     ut, st = mrna_dynamics(tau, 0.0, 0.0, alpha, beta, gamma)
#     ut_, st_ = mrna_dynamics(tau_, u_inf, s_inf, 0.0, beta, gamma)

#     u_scale_ = u_scale / scale
#     state_on = ((ut - u_obs) / u_scale_) ** 2 + ((st - s_obs) / s_scale) ** 2
#     state_off = ((ut_ - u_obs) / u_scale_) ** 2 + ((st_ - s_obs) / s_scale) ** 2
#     cell_gene_state_logits = state_on - state_off
#     cell_gene_state = cell_gene_state_logits < 0
#     t = torch.where(cell_gene_state_logits < 0, tau, tau_ + switching)
#     init_values = {
#         "alpha": alpha,
#         "beta": beta,
#         "gamma": gamma,
#         "switching": switching,
#         "latent_time": t,
#         "u_scale": u_scale,
#         "s_scale": s_scale,
#         "u_inf": u_inf,
#         "s_inf": s_inf,
#     }
#     if input_type == "knn":
#         init_values["mask"] = training_mask

#     init_values["cell_gene_state"] = cell_gene_state.int()
#     if latent_factor == "linear":
#         if "spliced" in adata.layers:
#             u_obs = torch.tensor(
#                 adata.layers["unspliced"].toarray()
#                 if issparse(adata.layers["unspliced"])
#                 else adata.layers["unspliced"],
#                 dtype=torch.float32,
#             )
#             s_obs = torch.tensor(
#                 adata.layers["spliced"].toarray()
#                 if issparse(adata.layers["spliced"])
#                 else adata.layers["spliced"],
#                 dtype=torch.float32,
#             )
#         test = np.hstack([u_obs, s_obs])
#         pca = PCA(n_components=latent_factor_size)
#         pca.fit(test)
#         X_train_pca = pca.transform(test)
#         init_values["cell_codebook"] = torch.tensor(pca.components_)
#         init_values["u_pcs_mean"] = torch.tensor(pca.mean_[: adata.shape[1]])
#         init_values["s_pcs_mean"] = torch.tensor(pca.mean_[adata.shape[1] :])
#         if plate_size == 2:
#             init_values["cell_code"] = (
#                 torch.tensor(X_train_pca).unsqueeze(-1).transpose(-1, -2)
#             )
#         else:
#             init_values["cell_code"] = torch.tensor(X_train_pca)

#     if num_aux_cells > 0:
#         np.random.seed(99)
#         if "cytotrace" in adata.obs.columns:
#             order_aux = np.array_split(
#                 np.sort(adata.obs["cytotrace"].values), 50
#             )
#             order_aux_list = []
#             for i in order_aux:
#                 order_aux_list.append(
#                     np.random.choice(
#                         np.where(
#                             (adata.obs["cytotrace"].values > i[0])
#                             & (adata.obs["cytotrace"].values < i[-1])
#                         )[0]
#                     )
#                 )
#             order_aux = np.array(order_aux_list)
#         else:
#             order_aux = np.argsort(
#                 (adata.layers["spliced"].toarray() > 0).sum(axis=1)
#             )[::-1]

#         init_values["order_aux"] = torch.from_numpy(
#             order_aux[:num_aux_cells].copy()
#         )

#         if input_type == "raw":
#             u_obs = adata.layers["raw_unspliced"].toarray()
#             s_obs = adata.layers["raw_spliced"].toarray()
#         init_values["aux_u_obs"] = torch.tensor(
#             u_obs[order_aux][:num_aux_cells]
#         )
#         init_values["aux_s_obs"] = torch.tensor(
#             s_obs[order_aux][:num_aux_cells]
#         )
#         init_values["cell_gene_state_aux"] = cell_gene_state[:num_aux_cells]
#         init_values["latent_time_aux"] = t[:num_aux_cells]

#         if latent_factor == "linear":
#             init_values["cell_code_aux"] = (
#                 torch.tensor(X_train_pca)
#                 .unsqueeze(-1)
#                 .transpose(-1, -2)[:num_aux_cells]
#             )

#     if shared_time:
#         cell_time = t.mean(dim=-1, keepdims=True)
#         init_values["cell_time"] = cell_time
#         init_values["latent_time"] = init_values["latent_time"] - cell_time

#         if num_aux_cells > 0:
#             init_values["cell_time_aux"] = cell_time[:num_aux_cells]
#             init_values["latent_time_aux"] = (
#                 init_values["latent_time_aux"] - init_values["cell_time_aux"]
#             )

#     for key in init_values:
#         print(key, init_values[key].shape)
#         print(init_values[key].isnan().sum())
#         assert init_values[key].isnan().sum() == 0
#     return init_values


# TODO: remove unused functions
# def trace(func):
#     def wrapper(*args, **kwargs):
#         params = list(args) + [f"{k}={v}" for k, v in kwargs.items()]
#         params_str = ",\n    ".join(map(str, params))
#         print(f"{func.__name__}(\n    {params_str},\n)")
#         return func(*args, **kwargs)

#     return wrapper


# def velocity_dus_dt(alpha, beta, gamma, tau, x):
#     """
#     Computes the velocity du/dt and ds/dt.

#     Args:
#         alpha (torch.Tensor): Alpha parameter.
#         beta (torch.Tensor): Beta parameter.
#         gamma (torch.Tensor): Gamma parameter.
#         tau (torch.Tensor): Time points.
#         x (Tuple[torch.Tensor, torch.Tensor]): Tuple containing u and s.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: Tuple containing du/dt and ds/dt.

#     Examples:
#         >>> import torch
#         >>> alpha = torch.tensor(0.5)
#         >>> beta = torch.tensor(0.4)
#         >>> gamma = torch.tensor(0.3)
#         >>> tau = torch.tensor(2.0)
#         >>> x = (torch.tensor(1.0), torch.tensor(0.5))
#         >>> velocity_dus_dt(alpha, beta, gamma, tau, x)
#         (tensor(0.1000), tensor(0.2500))
#     """
#     u, s = x
#     du_dt = alpha - beta * u
#     ds_dt = beta * u - gamma * s
#     return du_dt, ds_dt


# def rescale_time(dx_dt, t_start, t_end):
#     """
#     Converts an ODE to be solved on a batch of different time intervals into an
#     equivalent system of ODEs to be solved on [0, 1].

#     Args:
#         dx_dt (Callable): Function representing the ODE system.
#         t_start (torch.Tensor): Start time of the time interval.
#         t_end (torch.Tensor): End time of the time interval.

#     Returns:
#         Callable: Function representing the rescaled ODE system.

#     Examples:
#         >>> import torch
#         >>> def dx_dt(t, x):
#         ...     return -x
#         >>> t_start = torch.tensor(0.0)
#         >>> t_end = torch.tensor(1.0)
#         >>> rescaled_dx_dt = rescale_time(dx_dt, t_start, t_end)
#         >>> rescaled_dx_dt(torch.tensor(0.5), torch.tensor(2.0))
#         tensor(-2.)
#     """
#     dt = t_end - t_start

#     @functools.wraps(dx_dt)
#     def dx_ds(s, *args):
#         # move any batch dimensions in s to the left of all batch dimensions in dt
#         # s_shape = (1,) if len(s.shape) == 0 else s.shape  # XXX unnecessary?
#         s = s.reshape(s.shape + (1,) * len(dt.shape))
#         t = s * dt + t_start
#         xts = dx_dt(t, *args)
#         if isinstance(xts, tuple):
#             xss = tuple(xt * dt for xt in xts)
#         else:
#             xss = xts * dt
#         return xss

#     return dx_ds


# def ode_mrna_dynamics(tau, u0, s0, alpha, beta, gamma):
#     """
#     Solves the ODE system for mRNA dynamics.

#     Args:
#         tau (torch.Tensor): Time points.
#         u0 (torch.Tensor): Initial value of u.
#         s0 (torch.Tensor): Initial value of s.
#         alpha (torch.Tensor): Alpha parameter.
#         beta (torch.Tensor): Beta parameter.
#         gamma (torch.Tensor): Gamma parameter.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: Tuple containing the final values of u and s.
#     """
#     """
#     Examples:
#         >>> import torch
#         >>> tau = torch.tensor(2.0)
#         >>> u0 = torch.tensor(1.0)
#         >>> s0 = torch.tensor(0.5)
#         >>> alpha = torch.tensor(0.5)
#         >>> beta = torch.tensor(0.4)
#         >>> gamma = torch.tensor(0.3)
#         >>> ode_mrna_dynamics(tau, u0, s0, alpha, beta, gamma)
#         (tensor(0.6703), tensor(0.4596))
#     """
#     dx_dt = functools.partial(velocity_dus_dt, alpha, beta, gamma)
#     dx_dt = rescale_time(
#         dx_dt, torch.tensor(0.0, dtype=tau.dtype, device=tau.device), tau
#     )
#     grid = torch.linspace(0.0, 1.0, 100, dtype=tau.dtype, device=tau.device)
#     x0 = (u0.expand_as(tau), s0.expand_as(tau))
#     # xts = odeint_adjoint(dx_dt, x0, grid, adjoint_params=(alpha, beta, gamma, tau))
#     xts = odeint(dx_dt, x0, grid)
#     uts, sts = xts
#     ut, st = uts[-1], sts[-1]
#     return ut, st


# def get_velocity_samples(posterior_samples, model):
#     """
#     Computes the velocity samples from the posterior samples.

#     Args:
#         posterior_samples (dict): Dictionary containing posterior samples.
#         model: Model used for predictions.

#     Returns:
#         torch.Tensor: Velocity samples.

#     Examples:
#         >>> import torch
#         >>> posterior_samples = {
#         ...     "beta": torch.tensor([[0.4]]),
#         ...     "gamma": torch.tensor([[0.3]]),
#         ...     "u_scale": torch.tensor([[[1.0, 2.0]]]),
#         ...     "s_scale": torch.tensor([[[2.0, 4.0]]]),
#         ...     "u": torch.tensor([[1.0, 2.0]]),
#         ...     "s": torch.tensor([[0.5, 1.0]])
#         ... }
#         >>> model = None  # Model is not used in the function
#         >>> get_velocity_samples(posterior_samples, model)
#         tensor([[0.6500, 1.3000]])
#     """
#     beta = posterior_samples["beta"].mean(0)[0]
#     gamma = posterior_samples["gamma"].mean(0)[0]
#     scale = (
#         posterior_samples["u_scale"][:, 0, :]
#         / posterior_samples["s_scale"][:, 0, :]
#     ).mean(0)
#     ut = posterior_samples["u"] / scale
#     st = posterior_samples["s"]
#     v = beta * ut - gamma * st
#     return v

# def debug(x):
#     if torch.any(torch.isnan(x)):
#         print("nan number: ", torch.isnan(x).sum())
#         pdb.set_trace()


# def site_is_discrete(site: dict) -> bool:
#     return (
#         site["type"] == "sample"
#         and not site["is_observed"]
#         and getattr(site["fn"], "has_enumerate_support", False)
#     )


# def get_pylogger(name=__name__, log_level="DEBUG") -> logging.Logger:
#     """Initializes multi-GPU-friendly python command line logger."""

#     formatter = colorlog.ColoredFormatter(
#         "%(log_color)s%(levelname)s:%(name)s: %(message)s"
#         # "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
#         # datefmt=None,
#         # reset=True,
#         # log_colors={
#         #     "debug": "cyan",
#         #     "info": "green",
#         #     "warning": "yellow",
#         #     "error": "red",
#         #     "exception": "red",
#         #     "fatal": "red",
#         #     "critical": "red",
#         #     },
#     )

#     handler = colorlog.StreamHandler()
#     handler.setFormatter(formatter)
#     logger = colorlog.getLogger(name)
#     logger.setLevel(log_level)
#     logger.addHandler(handler)
#     logger.propagate = False

#     # this ensures all logging levels get marked with the rank zero decorator
#     # otherwise logs would get multiplied for each GPU process in multi-GPU setup
#     logging_levels = (
#         "debug",
#         "info",
#         "warning",
#         "error",
#         "exception",
#         "fatal",
#         "critical",
#     )
#     for level in logging_levels:
#         setattr(logger, level, rank_zero_only(getattr(logger, level)))

#     return logger
#     for level in logging_levels:
#         setattr(logger, level, rank_zero_only(getattr(logger, level)))

#     return logger
