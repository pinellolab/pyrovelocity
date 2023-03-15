import logging
import os
import pickle
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import pyro
import scvi
import torch
from anndata import AnnData
from numpy import ndarray
from scvi.data import transfer_anndata_setup
from scvi.model._utils import parse_use_gpu_arg
from scvi.model.base import BaseModelClass
from scvi.model.base._utils import _initialize_model
from scvi.model.base._utils import _validate_var_names
from scvi.module.base import PyroBaseModuleClass

from pyrovelocity.data import setup_anndata_multilayers

from ._trainer import VelocityTrainingMixin
from ._velocity_module import VelocityModule
from .utils import init_with_all_cells


logger = logging.getLogger(__name__)


class PyroVelocity(VelocityTrainingMixin, BaseModelClass):
    def __init__(
        self,
        adata: AnnData,
        input_type: str = "raw",
        shared_time: bool = True,
        model_type: str = "auto",
        guide_type: str = "auto",
        likelihood: str = "Poisson",
        t_scale_on: bool = False,
        plate_size: int = 2,
        latent_factor: str = "none",
        latent_factor_operation: str = "selection",
        inducing_point_size: int = 0,
        latent_factor_size: int = 0,
        include_prior: bool = False,
        use_gpu: int = 0,
        init: bool = False,
        num_aux_cells: int = 0,
        only_cell_times: bool = True,
        decoder_on: bool = False,
        add_offset: bool = False,
        correct_library_size: Union[bool, str] = True,
        cell_specific_kinetics: Optional[str] = None,
        kinetics_num: Optional[int] = None,
    ) -> None:
        self.use_gpu = use_gpu
        self.cell_specific_kinetics = cell_specific_kinetics
        self.k = kinetics_num
        if input_type == "knn":
            layers = ["Mu", "Ms"]
            assert likelihood in ["Normal", "LogNormal"]
            assert "Mu" in adata.layers
        elif input_type == "raw_cpm":
            layers = ["unspliced", "spliced"]
            assert likelihood in ["Normal", "LogNormal"]
        else:
            layers = ["raw_unspliced", "raw_spliced"]
            assert likelihood != "Normal"

        self.layers = layers
        self.input_type = input_type

        adata = setup_anndata_multilayers(
            adata,
            layer=self.layers,
            copy=True,
            batch_key=None,
            input_type=input_type,
            cluster=self.cell_specific_kinetics,
        )
        scvi.data.view_anndata_setup(adata)
        super().__init__(adata)
        if init:
            initial_values = init_with_all_cells(
                self.adata,
                input_type,
                shared_time,
                latent_factor,
                latent_factor_size,
                plate_size,
                num_aux_cells=num_aux_cells,
            )
        else:
            initial_values = {}
        logger.info(self.summary_stats)
        self.module = VelocityModule(
            self.summary_stats["n_cells"],
            self.summary_stats["n_vars"],
            model_type=model_type,
            guide_type=guide_type,
            likelihood=likelihood,
            shared_time=shared_time,
            t_scale_on=t_scale_on,
            plate_size=plate_size,
            latent_factor=latent_factor,
            latent_factor_operation=latent_factor_operation,
            latent_factor_size=latent_factor_size,
            inducing_point_size=inducing_point_size,
            include_prior=include_prior,
            use_gpu=use_gpu,
            num_aux_cells=num_aux_cells,
            only_cell_times=only_cell_times,
            decoder_on=decoder_on,
            add_offset=add_offset,
            correct_library_size=correct_library_size,
            cell_specific_kinetics=cell_specific_kinetics,
            kinetics_num=self.k,
            **initial_values,
        )
        self.num_cells = self.module.num_cells
        self._model_summary_string = f"""
        RNA velocity Pyro model with following parameters:
        """
        self.init_params_ = self._get_init_params(locals())
        logger.info("The model has been initialized")

    def train(self, **kwargs):
        """
        Trains the PyroVelocity model using the provided data and configuration.

        The method leverages the Pyro library to train the model using the underlying
        data. It relies on the `VelocityTrainingMixin` to define the training logic.

        Args:

            **kwargs : dict, optional
                Additional keyword arguments to be passed to the underlying train method
                provided by the `VelocityTrainingMixin`.
        """
        pyro.enable_validation(True)
        super().train(**kwargs)

    def predict_new_samples(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        num_samples: Optional[int] = 100,
    ) -> Dict[str, ndarray]:
        adata = setup_anndata_multilayers(
            adata,
            layer=self.layers,
            copy=True,
            batch_key=None,
            input_type=self.input_type,
        )
        return self.posterior_samples(adata, indices, batch_size, num_samples)

    def enum_parallel_predict(self):
        """work for parallel enumeration"""
        return

    def posterior_samples(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        num_samples: int = 100,
    ) -> Dict[str, ndarray]:
        """
        If the guide uses sequential enumeration, computes the posterior samples for
        the given data using the trained PyroVelocity model.

        The method generates posterior samples by running the trained model on the
        provided data and returns a dictionary containing samples for each parameter.

        Args:
            adata (AnnData, optional): Anndata object containing the data for which posterior samples
                are to be computed. If not provided, the anndata used to initialize the model will be used.
            indices (Sequence[int], optional): Indices of cells in `adata` for which the posterior
                samples are to be computed.
            batch_size (int, optional): The size of the mini-batches used during computation.
                If not provided, the entire dataset will be used.
            num_samples (int, default: 100): The number of posterior samples to compute for each parameter.

        Returns:
            Dict[str, ndarray]: A dictionary containing the posterior samples for each parameter.
        """
        self.module.eval()
        predictive = self.module.create_predictive(
            model=pyro.poutine.uncondition(
                self.module.model
            ),  # do not input u_obs, and s_obs
            num_samples=num_samples,
        )

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        with torch.no_grad(), pyro.poutine.mask(mask=False):
            posterior_samples = []
            for tensor in scdl:
                args, kwargs = self.module._get_fn_args_from_batch(tensor)
                posterior_sample = {
                    k: v.cpu().numpy() for k, v in predictive(*args, **kwargs).items()
                }
                posterior_samples.append(posterior_sample)
            samples = {}
            for k in posterior_samples[0].keys():
                if "aux" in k:
                    samples[k] = posterior_samples[0][k]  # alpha, beta, gamma...
                elif posterior_samples[0][k].shape[-2] == 1:
                    samples[k] = posterior_samples[0][k]
                    if k == "kinetics_prob":
                        samples[k] = np.concatenate(
                            [
                                # cat mini-batches
                                posterior_samples[j][k]
                                for j in range(len(posterior_samples))
                            ],
                            axis=-3,
                        )
                else:
                    samples[k] = np.concatenate(
                        [
                            # cat mini-batches
                            posterior_samples[j][k]
                            for j in range(len(posterior_samples))
                        ],
                        axis=-2,
                    )
        return samples

    def save(
        self,
        dir_path: str,
        overwrite: bool = True,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ) -> None:
        super().save(dir_path, overwrite, save_anndata, **anndata_write_kwargs)
        pyro.get_param_store().save(os.path.join(dir_path, "param_store_test.pt"))

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
    ):
        load_adata = adata is None
        _, device = parse_use_gpu_arg(use_gpu)

        (
            scvi_setup_dict,
            attr_dict,
            var_names,
            model_state_dict,
            new_adata,
        ) = _load_saved_files(dir_path, load_adata, map_location=device)
        adata = new_adata if new_adata is not None else adata

        _validate_var_names(adata, var_names)
        transfer_anndata_setup(scvi_setup_dict, adata)
        model = _initialize_model(cls, adata, attr_dict)

        # set saved attrs for loaded model
        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        # some Pyro modules with AutoGuides may need one training step
        old_history = model.history_
        try:
            model.module.load_state_dict(model_state_dict)
        except RuntimeError as err:
            if not isinstance(model.module, PyroBaseModuleClass):
                raise err
            # old_history = model.history_
            logger.info("Preparing underlying module for load")
            try:
                model.train(max_steps=1, use_gpu=use_gpu)
            except Exception:
                model.train(max_steps=1, use_gpu=use_gpu, batch_size=adata.shape[0])
            model.history_ = old_history
            pyro.clear_param_store()
            model.module.load_state_dict(model_state_dict)
        model.to_device(device)

        model.module.eval()

        model._validate_anndata(adata)

        # load pyro pyaram stores
        pyro.get_param_store().load(
            os.path.join(dir_path, "param_store_test.pt"), map_location=device
        )
        return model


def _load_saved_files(
    dir_path: str,
    load_adata: bool,
    map_location: Optional[Literal["cpu", "cuda"]] = None,
):
    """Helper to load saved files, adapt from scvi-tools removing check in _CONSTANTS keys"""
    setup_dict_path = os.path.join(dir_path, "attr.pkl")
    adata_path = os.path.join(dir_path, "adata.h5ad")
    varnames_path = os.path.join(dir_path, "var_names.csv")
    model_path = os.path.join(dir_path, "model_params.pt")

    if os.path.exists(adata_path) and load_adata:
        adata = read(adata_path)
    elif not os.path.exists(adata_path) and load_adata:
        raise ValueError("Save path contains no saved anndata and no adata was passed.")
    else:
        adata = None

    var_names = np.genfromtxt(varnames_path, delimiter=",", dtype=str)

    with open(setup_dict_path, "rb") as handle:
        attr_dict = pickle.load(handle)
    scvi_setup_dict = attr_dict.pop("scvi_setup_dict_")
    print(map_location)
    model_state_dict = torch.load(model_path, map_location=map_location)
    return scvi_setup_dict, attr_dict, var_names, model_state_dict, adata
