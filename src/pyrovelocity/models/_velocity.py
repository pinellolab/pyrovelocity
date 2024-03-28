import os
import pickle
import sys
from statistics import harmonic_mean
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

import mlflow
import numpy as np
import pyro
import torch
from anndata import AnnData
from beartype import beartype
from numpy import ndarray
from scvi.data import AnnDataManager
from scvi.data._constants import _SETUP_ARGS_KEY
from scvi.data._constants import _SETUP_METHOD_NAME
from scvi.data.fields import LayerField
from scvi.data.fields import NumericalObsField
from scvi.model._utils import parse_device_args
from scvi.model.base import BaseModelClass
from scvi.model.base._utils import _initialize_model
from scvi.model.base._utils import _load_saved_files
from scvi.model.base._utils import _validate_var_names
from scvi.module.base import PyroBaseModuleClass

from pyrovelocity.analysis.analyze import compute_mean_vector_field
from pyrovelocity.analysis.analyze import compute_volcano_data
from pyrovelocity.analysis.analyze import vector_field_uncertainty
from pyrovelocity.logging import configure_logging
from pyrovelocity.models._trainer import VelocityTrainingMixin
from pyrovelocity.models._velocity_module import VelocityModule


__all__ = ["PyroVelocity"]


logger = configure_logging(__name__)


class PyroVelocity(VelocityTrainingMixin, BaseModelClass):
    """
    PyroVelocity is a class for constructing and training a Pyro model for
    probabilistic RNA velocity estimation. This model leverages the
    probabilistic programming language Pyro to estimate the parameters of models
    for the dynamics of RNA transcription, splicing, and degradation, providing
    the opportunity for insight into cellular states and associated state
    transitions. It makes use of AnnData, scvi-tools, and other scverse
    ecosystem libraries.

    Public methods include training the model with various configurations,
    generating posterior samples for further analysis, and saving/loading the
    model for reproducibility and further analysis.

    Attributes:
        use_gpu (str): Whether and which GPU to use.
        cell_specific_kinetics (Optional[str]): Type of cell-specific kinetics.
        k (Optional[int]): Number of kinetics.
        layers (List[str]): List of layers in the dataset.
        input_type (str): Type of input data.
        module (VelocityModule):
            The Pyro module used for the velocity estimation model.
        num_cells (int): Number of cells in the dataset.
        num_samples (int): Number of posterior samples to generate.
        _model_summary_string (str): Summary string for the model.
        init_params_ (Dict[str, Any]): Initial parameters for the model.

    For usage examples, including training the model and generating posterior
    samples, refer to the individual method docstrings.
    """

    """
    The `Methods` section is not supported by all documentation generators but
    is provided detached from the class docstring for reference. Please
    see the docstrings for each method for more details. This list may ignore
    unused or private methods.

    Methods:
        train:
            Trains the PyroVelocity model using the provided data and configuration.
        setup_anndata: 
            Set up AnnData object for compatibility with the scvi-tools
            model training interface.
        generate_posterior_samples:
            Generates posterior samples for the given data using the trained
            PyroVelocity model.
        compute_statistics_from_posterior_samples:
            Estimate statistics from posterior samples and add them to the
            `posterior_samples` dictionary.
        save_pyrovelocity_data:
            Saves the PyroVelocity data to a pickle file.
        save_model:
            Saves the Pyro-Velocity model to a directory.
        load_model:
            Load the model from a directory with the same structure as that produced
            by the save method.
    """

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
        use_gpu: str = "auto",
        init: bool = False,
        num_aux_cells: int = 0,
        only_cell_times: bool = True,
        decoder_on: bool = False,
        add_offset: bool = False,
        correct_library_size: Union[bool, str] = True,
        cell_specific_kinetics: Optional[str] = None,
        kinetics_num: Optional[int] = None,
    ) -> None:
        """
        PyroVelocity class for estimating RNA velocity and related tasks.

        Args:
            adata (AnnData): An AnnData object containing the gene expression data.
            input_type (str, optional): Type of input data. Can be "raw", "knn", or "raw_cpm". Defaults to "raw".
            shared_time (bool, optional): Whether to use shared time. Defaults to True.
            model_type (str, optional): Type of model to use. Defaults to "auto".
            guide_type (str, optional): Type of guide to use. Defaults to "auto".
            likelihood (str, optional): Type of likelihood to use. Defaults to "Poisson".
            t_scale_on (bool, optional): Whether to use t_scale. Defaults to False.
            plate_size (int, optional): Size of the plate. Defaults to 2.
            latent_factor (str, optional): Type of latent factor. Defaults to "none".
            latent_factor_operation (str, optional): Operation to perform on the latent factor. Defaults to "selection".
            inducing_point_size (int, optional): Size of inducing points. Defaults to 0.
            latent_factor_size (int, optional): Size of latent factors. Defaults to 0.
            include_prior (bool, optional): Whether to include prior information. Defaults to False.
            use_gpu (Union[bool, int], optional): Whether and which GPU to use. Defaults to 0. Can be False.
            init (bool, optional): Whether to initialize the model. Defaults to False.
            num_aux_cells (int, optional): Number of auxiliary cells. Defaults to 0.
            only_cell_times (bool, optional): Whether to use only cell times. Defaults to True.
            decoder_on (bool, optional): Whether to use decoder. Defaults to False.
            add_offset (bool, optional): Whether to add offset. Defaults to False.
            correct_library_size (Union[bool, str], optional): Whether to correct library size or method to correct. Defaults to True.
            cell_specific_kinetics (Optional[str], optional): Type of cell-specific kinetics. Defaults to None.
            kinetics_num (Optional[int], optional): Number of kinetics. Defaults to None.

        Examples:
            >>> # import necessary libraries
            >>> import numpy as np
            >>> import anndata
            >>> from pyrovelocity.utils import pretty_log_dict, print_anndata, generate_sample_data
            >>> from pyrovelocity.tasks.preprocess import copy_raw_counts
            >>> from pyrovelocity.models._velocity import PyroVelocity
            ...
            >>> # define fixtures
            >>> try:
            >>>     tmp = getfixture("tmp_path")
            >>> except NameError:
            >>>     import tempfile
            >>>     tmp = tempfile.TemporaryDirectory().name
            >>> doctest_model_path = str(tmp) + "/save_pyrovelocity_doctest_model"
            >>> print(doctest_model_path)
            ...
            >>> # setup sample data
            >>> n_obs = 10
            >>> n_vars = 5
            >>> adata = generate_sample_data(n_obs=n_obs, n_vars=n_vars)
            >>> copy_raw_counts(adata)
            >>> print_anndata(adata)
            >>> print(adata.X)
            >>> print(adata.layers['spliced'])
            >>> print(adata.layers['unspliced'])
            >>> print(adata.obs['u_lib_size_raw'])
            >>> print(adata.obs['s_lib_size_raw'])
            >>> PyroVelocity.setup_anndata(adata)
            ...
            >>> # train model with macroscopic validation set
            >>> model = PyroVelocity(adata)
            >>> model.train(max_epochs=5, train_size=0.8, valid_size=0.2, use_gpu="auto")
            >>> posterior_samples = model.generate_posterior_samples(model.adata, num_samples=30)
            >>> print(posterior_samples.keys())
            >>> assert isinstance(posterior_samples, dict), f"Expected a dictionary, got {type(posterior_samples)}"
            >>> posterior_samples_log = pretty_log_dict(posterior_samples)
            >>> model.save_model(doctest_model_path, overwrite=True)
            >>> model = PyroVelocity.load_model(doctest_model_path, adata, use_gpu="auto")
            ...
            >>> # train model with default parameters
            >>> model = PyroVelocity(adata)
            >>> model.train_faster(max_epochs=5, use_gpu="auto")
            >>> model.save_model(doctest_model_path, overwrite=True)
            >>> model = PyroVelocity.load_model(doctest_model_path, adata, use_gpu="auto")
            >>> posterior_samples = model.generate_posterior_samples(model.adata, num_samples=30)
            >>> posterior_samples_log = pretty_log_dict(posterior_samples)
            >>> print(posterior_samples.keys())
            ...
            >>> # train model with specified batch size
            >>> model = PyroVelocity(adata)
            >>> model.train_faster_with_batch(batch_size=24, max_epochs=5, use_gpu="auto")
            >>> model.save_model(doctest_model_path, overwrite=True)
            >>> model = PyroVelocity.load_model(doctest_model_path, adata, use_gpu="auto")
            >>> posterior_samples = model.generate_posterior_samples(model.adata, num_samples=30)
            >>> posterior_samples_log = pretty_log_dict(posterior_samples)
            >>> print(posterior_samples.keys())
            ...
            >>> # If running from an interactive session, the temporary directory
            >>> # can be inspected to review the saved model files. When run as a
            >>> # doctest it is automatically cleaned up after the test completes.
            >>> print(f"Output located in {doctest_model_path}")
        """
        self.use_gpu = use_gpu
        self.cell_specific_kinetics = cell_specific_kinetics
        self.k = kinetics_num
        if input_type == "knn":
            layers = ["Mu", "Ms"]
            assert likelihood in {"Normal", "LogNormal"}
            assert "Mu" in adata.layers
        elif input_type == "raw_cpm":
            layers = ["unspliced", "spliced"]
            assert likelihood in {"Normal", "LogNormal"}
        else:
            layers = ["raw_unspliced", "raw_spliced"]
            assert likelihood != "Normal"

        self.layers = layers
        self.input_type = input_type

        super().__init__(adata)
        # TODO: remove unused code
        # from pyrovelocity.utils import init_with_all_cells
        # if init:
        #     initial_values = init_with_all_cells(
        #         self.adata,
        #         input_type,
        #         shared_time,
        #         latent_factor,
        #         latent_factor_size,
        #         plate_size,
        #         num_aux_cells=num_aux_cells,
        #     )
        # else:
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
        self._model_summary_string = """
        RNA velocity Pyro model with parameters:
        """
        self.init_params_ = self._get_init_params(locals())
        logger.info("Model initialized")

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

    def enum_parallel_predict(self):
        """work for parallel enumeration"""
        return

    @classmethod
    def setup_anndata(cls, adata: AnnData, *args, **kwargs):
        """
        Set up AnnData object for compatibility with the scvi-tools
        model training interface.

        Args:
            adata (AnnData): Anndata object to be used in model training.
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        adata.obs["u_lib_size"] = np.log(
            adata.obs["u_lib_size_raw"].astype(float) + 1e-6
        )
        adata.obs["s_lib_size"] = np.log(
            adata.obs["s_lib_size_raw"].astype(float) + 1e-6
        )

        adata.obs["u_lib_size_mean"] = adata.obs["u_lib_size"].mean()
        adata.obs["s_lib_size_mean"] = adata.obs["s_lib_size"].mean()
        adata.obs["u_lib_size_scale"] = adata.obs["u_lib_size"].std()
        adata.obs["s_lib_size_scale"] = adata.obs["s_lib_size"].std()
        adata.obs["ind_x"] = np.arange(adata.n_obs).astype("int64")

        anndata_fields = [
            LayerField("U", "raw_unspliced", is_count_data=True),
            LayerField("X", "raw_spliced", is_count_data=True),
            NumericalObsField("u_lib_size", "u_lib_size"),
            NumericalObsField("s_lib_size", "s_lib_size"),
            NumericalObsField("u_lib_size_mean", "u_lib_size_mean"),
            NumericalObsField("s_lib_size_mean", "s_lib_size_mean"),
            NumericalObsField("u_lib_size_scale", "u_lib_size_scale"),
            NumericalObsField("s_lib_size_scale", "s_lib_size_scale"),
            NumericalObsField("ind_x", "ind_x"),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def generate_posterior_samples(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        num_samples: int = 100,
    ) -> Dict[str, ndarray]:
        """
        Generates posterior samples for the given data using the trained
        PyroVelocity model.

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
            model=pyro.poutine.uncondition(self.module.model),
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
                    k: v.cpu().numpy()
                    for k, v in predictive(*args, **kwargs).items()
                }
                posterior_samples.append(posterior_sample)
            samples = {}
            for k in posterior_samples[0].keys():
                if k in [
                    "ut_norm",
                    "st_norm",
                    "time_constraint",
                ]:
                    continue

                if "aux" in k:
                    samples[k] = posterior_samples[0][k]
                elif posterior_samples[0][k].shape[-2] == 1:
                    samples[k] = posterior_samples[0][k]
                    if k == "kinetics_prob":
                        samples[k] = np.concatenate(
                            [
                                posterior_samples[j][k]
                                for j in range(len(posterior_samples))
                            ],
                            axis=-3,
                        )
                else:
                    samples[k] = np.concatenate(
                        [
                            posterior_samples[j][k]
                            for j in range(len(posterior_samples))
                        ],
                        axis=-2,
                    )

                logger.debug(k, "before", sys.getsizeof(samples[k]))
        self.num_samples = num_samples
        return samples

    def get_mlflow_logs(self):
        return

    def compute_statistics_from_posterior_samples(
        self,
        adata: AnnData,
        posterior_samples: Dict[str, ndarray],
        vector_field_basis: str = "umap",
        ncpus_use: int = 1,
    ) -> Dict[str, ndarray]:
        """
        Estimate statistics from posterior samples and add them to the
        `posterior_samples` dictionary. The names of the statistics incorporated into
        the dictionary are:

        - `gene_ranking`
        - `original_spaces_embeds_magnitude`
        - `genes`
        - `vector_field_posterior_samples`
        - `vector_field_posterior_mean`
        - `fdri`
        - `embeds_magnitude`
        - `embeds_angle`
        - `ut_mean`
        - `st_mean`
        - `pca_vector_field_posterior_samples`
        - `pca_embeds_angle`
        - `pca_fdri`

        The following data are removed from the `posterior_samples` dictionary:

        - `u`
        - `s`
        - `ut`
        - `st`

        Each of these sets requires further documentation.

        Args:
            adata (AnnData): Anndata object containing the data for which posterior samples
                were computed.
            posterior_samples (Dict[str, ndarray]): Dictionary containing the posterior samples
                for each parameter.
            vector_field_basis (str, optional): Basis for the vector field. Defaults to "umap".
            ncpus_use (int, optional): Number of CPUs to use for computation. Defaults to 1.

        Returns:
            Dict[str, ndarray]: Dictionary containing the posterior samples with added statistics.
        """
        if ("u_scale" in posterior_samples) and (
            "s_scale" in posterior_samples
        ):
            scale = posterior_samples["u_scale"] / posterior_samples["s_scale"]
        elif ("u_scale" in posterior_samples) and not (
            "s_scale" in posterior_samples
        ):
            scale = posterior_samples["u_scale"]
        else:
            scale = 1
        original_spaces_velocity_samples = (
            posterior_samples["beta"] * posterior_samples["ut"] / scale
            - posterior_samples["gamma"] * posterior_samples["st"]
        )
        original_spaces_embeds_magnitude = np.sqrt(
            (original_spaces_velocity_samples**2).sum(axis=-1)
        )

        (
            vector_field_posterior_samples,
            embeds_radian,
            fdri,
        ) = vector_field_uncertainty(
            adata,
            posterior_samples,
            basis=vector_field_basis,
            n_jobs=ncpus_use,
        )
        embeds_magnitude = np.sqrt(
            (vector_field_posterior_samples**2).sum(axis=-1)
        )

        mlflow.log_metric(
            "FDR_sig_frac", round((fdri < 0.05).sum() / fdri.shape[0], 3)
        )
        mlflow.log_metric("FDR_HMP", harmonic_mean(fdri))

        compute_mean_vector_field(
            posterior_samples=posterior_samples,
            adata=adata,
            basis=vector_field_basis,
            n_jobs=ncpus_use,
        )

        vector_field_posterior_mean = adata.obsm[
            f"velocity_pyro_{vector_field_basis}"
        ]

        gene_ranking, genes = compute_volcano_data(
            [posterior_samples], [adata], time_correlation_with="st"
        )
        gene_ranking = (
            gene_ranking.sort_values("mean_mae", ascending=False)
            .head(300)
            .sort_values("time_correlation", ascending=False)
        )
        posterior_samples["gene_ranking"] = gene_ranking
        posterior_samples[
            "original_spaces_embeds_magnitude"
        ] = original_spaces_embeds_magnitude
        posterior_samples["genes"] = genes
        posterior_samples[
            "vector_field_posterior_samples"
        ] = vector_field_posterior_samples
        posterior_samples[
            "vector_field_posterior_mean"
        ] = vector_field_posterior_mean
        posterior_samples["fdri"] = fdri
        posterior_samples["embeds_magnitude"] = embeds_magnitude
        posterior_samples["embeds_angle"] = embeds_radian
        posterior_samples["ut_mean"] = posterior_samples["ut"].mean(0).squeeze()
        posterior_samples["st_mean"] = posterior_samples["st"].mean(0).squeeze()

        (
            pca_vector_field_posterior_samples,
            pca_embeds_radian,
            pca_fdri,
        ) = vector_field_uncertainty(
            adata,
            posterior_samples,
            basis="pca",
            n_jobs=ncpus_use,
        )
        posterior_samples[
            "pca_vector_field_posterior_samples"
        ] = pca_vector_field_posterior_samples
        posterior_samples["pca_embeds_angle"] = pca_embeds_radian
        posterior_samples["pca_fdri"] = pca_fdri

        del posterior_samples["u"]
        del posterior_samples["s"]
        del posterior_samples["ut"]
        del posterior_samples["st"]
        return posterior_samples

    @beartype
    def save_pyrovelocity_data(
        self,
        posterior_samples: Dict[str, ndarray],
        pyrovelocity_data_path: os.PathLike | str,
    ):
        """
        Save the PyroVelocity data to a pickle file.

        Args:
            posterior_samples (Dict[str, ndarray]): Dictionary containing the posterior samples
            pyrovelocity_data_path (os.PathLike | str): Path to save the PyroVelocity data.
        """
        with open(pyrovelocity_data_path, "wb") as f:
            pickle.dump(posterior_samples, f)
        for k in posterior_samples:
            logger.debug(k, "after", sys.getsizeof(posterior_samples[k]))

    def save_model(
        self,
        dir_path: str,
        prefix: Optional[str] = None,
        overwrite: bool = True,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ) -> None:
        """
        Save the Pyro-Velocity model to a directory.

        Dispatches to the `save` method of the inherited `BaseModelClass` which
        calls `torch.save` on a model state dictionary, variable names, and user
        attributes.

        Args:
            dir_path (str): Path to the directory where the model will be saved.
            prefix (Optional[str], optional): Prefix to add to the saved files. Defaults to None.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to True.
            save_anndata (bool, optional): Whether to save the AnnData object. Defaults to False.
        """
        super().save(
            dir_path, prefix, overwrite, save_anndata, **anndata_write_kwargs
        )
        pyro.get_param_store().save(
            os.path.join(dir_path, "param_store_test.pt")
        )

    @classmethod
    def load_model(
        cls,
        dir_path: str,
        adata: Optional[AnnData] = None,
        use_gpu: str = "auto",
        prefix: Optional[str] = None,
        backup_url: Optional[str] = None,
    ) -> BaseModelClass:
        """
        Load the model from a directory with the same structure as that produced
        by the save method.

        Args:
            dir_path (str): Path to the directory where the model is saved.
            adata (Optional[AnnData], optional): Anndata object to load into the model. Defaults to None.
            use_gpu (str, optional): Whether and which GPU to use. Defaults to "auto".
            prefix (Optional[str], optional): Prefix to add to the saved files. Defaults to None.
            backup_url (Optional[str], optional): URL to download the model from. Defaults to None.

        Raises:
            RuntimeError: If the model is not an instance of PyroBaseModuleClass.

        Returns:
            PyroVelocity: The loaded PyroVelocity model.
        """
        load_adata = adata is None
        _accelerator, _devices, device = parse_device_args(
            accelerator=use_gpu, return_device="torch"
        )
        logger.info(
            f"\nLoading model with:\n"
            f"\taccelerator: {_accelerator}\n"
            f"\tdevices: {_devices}\n"
            f"\tdevice: {device}\n"
        )

        (
            attr_dict,
            var_names,
            model_state_dict,
            new_adata,
        ) = _load_saved_files(
            dir_path,
            load_adata,
            map_location=device,
            prefix=prefix,
            backup_url=backup_url,
        )

        adata = new_adata if new_adata is not None else adata

        _validate_var_names(adata, var_names)

        registry = attr_dict.pop("registry_")
        method_name = registry.get(_SETUP_METHOD_NAME, "setup_anndata")
        getattr(cls, method_name)(
            adata, source_registry=registry, **registry[_SETUP_ARGS_KEY]
        )

        model = _initialize_model(cls, adata, attr_dict)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        pyro.clear_param_store()
        old_history = model.history_
        try:
            model.module.load_state_dict(model_state_dict)
        except RuntimeError as err:
            if not isinstance(model.module, PyroBaseModuleClass):
                raise err
            logger.info(
                "Preparing underlying `PyroBaseModuleClass` module for load"
            )
            try:
                model.train(max_epochs=1, max_steps=1)
            except Exception:
                model.train(
                    max_epochs=1,
                    max_steps=1,
                    batch_size=adata.shape[0],
                    train_size=0.8,
                    valid_size=0.2,
                )
            model.module.load_state_dict(model_state_dict)

        model.history_ = old_history
        model.to_device(device)
        model.module.eval()
        model._validate_anndata(adata)
        pyro.get_param_store().load(
            os.path.join(dir_path, "param_store_test.pt"),
            map_location=device,
        )
        return model
