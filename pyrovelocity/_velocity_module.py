from typing import Optional
from typing import Union

from pyro import poutine
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.guides import AutoGuideList
from scvi.module.base import PyroBaseModuleClass

# from ._velocity_model import AuxCellVelocityModel
# from ._velocity_model import AuxTrajectoryModel
# from ._velocity_model import DecoderTimeModel
# from ._velocity_model import LatentFactor
# from ._velocity_model import MultiKineticsModelDirichlet
# from ._velocity_model import MultiKineticsModelDirichletLinear
# from ._velocity_model import VelocityModel
from ._velocity_model import VelocityModelAuto


# from ._velocity_guide import LatentGuide
# from ._velocity_guide import AutoDeltaRNAVelocityGuide
# from ._velocity_guide import AutoNormalRNAVelocityGuide
# from ._velocity_guide import AuxCellVelocityGuide
# from ._velocity_guide import DecoderTimeGuide
# from ._velocity_guide import MultiKineticsGuide
# from ._velocity_guide import TrajectoryGuide
# from ._velocity_guide import VelocityAutoGuideList
# from ._velocity_guide import VelocityGuide


class VelocityModule(PyroBaseModuleClass):
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
        self._get_fn_args_from_batch = self._model._get_fn_args_from_batch

    @property
    def model(self) -> VelocityModelAuto:
        return self._model

    @property
    def guide(self) -> AutoGuideList:
        return self._guide
