from typing import Optional
from typing import Union

from pyro import poutine
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.guides import AutoGuideList
from scvi.module.base import PyroBaseModuleClass

from pyrovelocity._velocity_model import VelocityModelAuto


class VelocityModule(PyroBaseModuleClass):
    """
    VelocityModule is an scvi-tools pyro module that combines the VelocityModelAuto and pyro AutoGuideList classes.

    Args:
        num_cells (int): Number of cells.
        num_genes (int): Number of genes.
        model_type (str, optional): Model type. Default is "auto".
        guide_type (str, optional): Guide type. Default is "velocity_auto".
        likelihood (str, optional): Likelihood type. Default is "Poisson".
        shared_time (bool, optional): If True, a shared time parameter will be used. Default is True.
        t_scale_on (bool, optional): If True, scale time parameter. Default is False.
        plate_size (int, optional): Size of the plate set. Default is 2.
        latent_factor (str, optional): Latent factor. Default is "none".
        latent_factor_operation (str, optional): Latent factor operation mode. Default is "selection".
        latent_factor_size (int, optional): Size of the latent factor. Default is 10.
        inducing_point_size (int, optional): Inducing point size. Default is 0.
        include_prior (bool, optional): If True, include prior in the model. Default is False.
        use_gpu (int, optional): GPU device index. Default is 0.
        num_aux_cells (int, optional): Number of auxiliary cells. Default is 0.
        only_cell_times (bool, optional): If True, only model cell times. Default is True.
        decoder_on (bool, optional): If True, use the decoder. Default is False.
        add_offset (bool, optional): If True, add offset to the model. Default is True.
        correct_library_size (Union[bool, str], optional): Library size correction method. Default is True.
        cell_specific_kinetics (Optional[str], optional): Cell-specific kinetics method. Default is None.
        kinetics_num (Optional[int], optional): Number of kinetics. Default is None.
        **initial_values: Initial values for the model parameters.

    Examples:
        >>> from scvi.module.base import PyroBaseModuleClass
        >>> from pyrovelocity._velocity_module import VelocityModule
        >>> num_cells = 10
        >>> num_genes = 20
        >>> velocity_module1 = VelocityModule(
        ...     num_cells, num_genes, model_type="auto",
        ...     guide_type="auto_t0_constraint", add_offset=False
        ... )
        -----------
        auto
        auto_t0_constraint
        >>> type(velocity_module1.model)
        <class 'pyrovelocity._velocity_model.VelocityModelAuto'>
        >>> type(velocity_module1.guide)
        <class 'pyro.infer.autoguide.guides.AutoGuideList'>
        >>> velocity_module2 = VelocityModule(
        ...     num_cells, num_genes, model_type="auto",
        ...     guide_type="auto", add_offset=True
        ... )
        -----------
        auto
        auto
        >>> type(velocity_module2.model)
        <class 'pyrovelocity._velocity_model.VelocityModelAuto'>
        >>> type(velocity_module2.guide)
        <class 'pyro.infer.autoguide.guides.AutoGuideList'>
    """

    def __init__(
        self,
        num_cells: int,
        num_genes: int,
        model_type: str = "auto",
        guide_type: str = "velocity_auto",
        likelihood: str = "Poisson",
        shared_time: bool = True,
        t_scale_on: bool = False,
        plate_size: int = 2,
        latent_factor: str = "none",
        latent_factor_operation: str = "selection",
        latent_factor_size: int = 10,
        inducing_point_size: int = 0,
        include_prior: bool = False,
        use_gpu: int = 0,
        num_aux_cells: int = 0,
        only_cell_times: bool = True,
        decoder_on: bool = False,
        add_offset: bool = True,
        correct_library_size: Union[bool, str] = True,
        cell_specific_kinetics: Optional[str] = None,
        kinetics_num: Optional[int] = None,
        **initial_values
    ) -> None:
        super().__init__()
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.model_type = model_type
        self.guide_type = guide_type
        self._model = None
        self.plate_size = plate_size
        self.num_aux_cells = num_aux_cells
        self.only_cell_times = only_cell_times
        print("-----------")
        print(self.model_type)
        print(self.guide_type)

        self.cell_specific_kinetics = cell_specific_kinetics

        self._model = VelocityModelAuto(
            self.num_cells,
            self.num_genes,
            likelihood,
            shared_time,
            t_scale_on,
            self.plate_size,
            latent_factor,
            latent_factor_operation=latent_factor_operation,
            latent_factor_size=latent_factor_size,
            include_prior=include_prior,
            num_aux_cells=num_aux_cells,
            only_cell_times=self.only_cell_times,
            decoder_on=decoder_on,
            add_offset=add_offset,
            correct_library_size=correct_library_size,
            guide_type=self.guide_type,
            cell_specific_kinetics=self.cell_specific_kinetics,
            **initial_values
        )

        guide = AutoGuideList(self._model, create_plates=self._model.create_plates)
        guide.append(
            AutoNormal(
                poutine.block(
                    self._model,
                    expose=[
                        "cell_time",
                        "u_read_depth",
                        "s_read_depth",
                        "kinetics_prob",
                        "kinetics_weights",
                    ],
                ),
                init_scale=0.1,
            )
        )

        if add_offset:
            guide.append(
                AutoLowRankMultivariateNormal(
                    poutine.block(
                        self._model,
                        expose=[
                            "alpha",
                            "beta",
                            "gamma",
                            "dt_switching",
                            "t0",
                            "u_scale",
                            "s_scale",
                            "u_offset",
                            "s_offset",
                        ],
                    ),
                    rank=10,
                    init_scale=0.1,
                )
            )
        else:
            guide.append(
                AutoLowRankMultivariateNormal(
                    poutine.block(
                        self._model,
                        expose=[
                            "alpha",
                            "beta",
                            "gamma",
                            "dt_switching",
                            "t0",
                            "u_scale",
                            "s_scale",
                        ],
                    ),
                    rank=10,
                    init_scale=0.1,
                )
            )
        self._guide = guide

    @property
    def model(self) -> VelocityModelAuto:
        return self._model

    @property
    def guide(self) -> AutoGuideList:
        return self._guide
