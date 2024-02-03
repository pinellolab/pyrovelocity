from typing import Optional
from typing import Union

from pyro import poutine
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.guides import AutoGuideList
from scvi.module.base import PyroBaseModuleClass

from pyrovelocity._velocity_model import VelocityModelAuto

from ._velocity_guide import AutoDeltaRNAVelocityGuide
from ._velocity_guide import AutoNormalRNAVelocityGuide
from ._velocity_guide import AuxCellVelocityGuide
from ._velocity_guide import DecoderTimeGuide
from ._velocity_guide import LatentGuide
from ._velocity_guide import MultiKineticsGuide
from ._velocity_guide import TrajectoryGuide
from ._velocity_guide import VelocityAutoGuideList
from ._velocity_guide import VelocityGuide
from ._velocity_model import AuxCellVelocityModel
from ._velocity_model import AuxTrajectoryModel
from ._velocity_model import DecoderTimeModel
from ._velocity_model import LatentFactor
from ._velocity_model import MultiKineticsModelDirichlet
from ._velocity_model import MultiKineticsModelDirichletLinear
from ._velocity_model import VelocityModel
from ._velocity_model import VelocityModelAuto


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
        **initial_values,
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
        if self.model_type == "multikinetics":
            # self._model = MultiKineticsModel(self.num_cells, self.num_genes, likelihood,
            self._model = MultiKineticsModelDirichlet(
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
                kinetics_num=kinetics_num,
                **initial_values,
            )
        if self.model_type == "multikinetics_linear":
            self._model = MultiKineticsModelDirichletLinear(
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
                kinetics_num=kinetics_num,
                **initial_values,
            )

        if self.model_type == "auto":
            # self._model = BlockedKineticsModel(self.num_cells,
            #                                   self.num_genes,
            #                                   likelihood,
            #                                   num_aux_cells=num_aux_cells,
            #                                   guide_type=self.guide_type,
            #                                   **initial_values)
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
                **initial_values,
            )
        if self.model_type == "traj":
            self._model = AuxTrajectoryModel(
                self.num_cells,
                self.num_genes,
                likelihood,
                num_aux_cells=num_aux_cells,
                **initial_values,
            )
        if self.model_type == "decoder_time":
            self._model = DecoderTimeModel(
                self.num_cells, self.num_genes, likelihood
            )
        elif self.model_type in ["velocity", "velocity2"]:
            if self.num_aux_cells >= 0:
                self._model = AuxCellVelocityModel(
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
                    **initial_values,
                )
            else:
                self._model = VelocityModel(
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
                    **initial_values,
                )
        elif self.model_type == "latentfactor":
            self._model = LatentFactor(
                self.num_cells,
                self.num_genes,
                likelihood,
                initial_values.get("mask", None),
                self.plate_size,
            )
        # create_plates is useful for keeping
        # batch_size same between model and guide
        if guide_type == "traj":
            self._guide = TrajectoryGuide(
                self._model,
                likelihood,
                shared_time,
                t_scale_on,
                latent_factor,
                latent_factor_operation,
                self.model_type,
                self.plate_size,
                **initial_values,
            )
        if guide_type == "autodelta":
            self._guide = AutoDeltaRNAVelocityGuide(
                self._model, use_gpu, **initial_values
            )
        elif guide_type == "autonormal":
            self._guide = AutoNormalRNAVelocityGuide(
                self._model, use_gpu, **initial_values
            )
        elif guide_type == "lowrank":
            self._guide = AutoLowRankMultivariateNormal(
                poutine.block(self._model, hide=["cell_gene_state"]), rank=20
            )
        elif guide_type == "inverse_t":
            if self.num_aux_cells >= 0:
                self._guide = AuxCellVelocityGuide(
                    self._model,
                    likelihood,
                    shared_time,
                    t_scale_on,
                    latent_factor,
                    latent_factor_operation,
                    self.model_type,
                    self.plate_size,
                    self.only_cell_times,
                    add_offset,
                    **initial_values,
                )
                ##guide = AutoGuideList(self._model)
                ##guide.append(AuxCellVelocityGuide(poutine.block(self._model, hide=["alpha", "beta", "gamma", "u_scale", "s_scale", "dt_switching", "cell_gene_state"]), likelihood, shared_time, t_scale_on, latent_factor, latent_factor_operation, self.model_type, self.plate_size, self.only_cell_times, add_offset, **initial_values))
                ##guide.append(AutoNormal(poutine.block(self._model, expose=["alpha", "beta", "gamma", "u_scale", "s_scale", "dt_switching"])))
                ##self._guide = guide
            else:
                self._guide = VelocityGuide(
                    self._model,
                    likelihood,
                    shared_time,
                    t_scale_on,
                    latent_factor,
                    latent_factor_operation,
                    self.model_type,
                    self.plate_size,
                    **initial_values,
                )
        elif guide_type == "decoder_time":
            self._guide = DecoderTimeGuide(self._model)
        elif guide_type == "multikinetics":
            self._guide = MultiKineticsGuide(
                self._model,
                likelihood,
                shared_time,
                t_scale_on,
                latent_factor,
                latent_factor_operation,
                self.model_type,
                self.plate_size,
                **initial_values,
            )
        elif (
            guide_type == "auto" or guide_type == "auto_t0_constraint"
        ):  # constraint on t0
            guide = AutoGuideList(
                self._model, create_plates=self._model.create_plates
            )
            if correct_library_size:
                if correct_library_size == "cell_size_regress":
                    guide.append(
                        AutoNormal(
                            poutine.block(
                                self._model,
                                expose=[
                                    "cell_time",
                                    "u_read_depth",
                                    "s_read_depth",
                                    "u_cell_size_coef",
                                    "ut_coef",
                                    "st_coef",
                                    "s_cell_size_coef",
                                    "kinetics_prob",
                                    "kinetics_weights",
                                ],
                            ),
                            init_scale=0.1,
                        )
                    )
                else:
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
            else:
                guide.append(
                    AutoNormal(
                        poutine.block(self._model, expose=["cell_time"]),
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
        elif (
            guide_type == "velocity_auto"
            or guide_type == "velocity_auto_t0_constraint"
        ):  # constraint on t0
            guide = VelocityAutoGuideList(
                self._model, create_plates=self._model.create_plates
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
            # def localguide(
            #    u_obs: Optional[torch.Tensor] = None,
            #    s_obs: Optional[torch.Tensor] = None,
            #    u_log_library: Optional[torch.Tensor] = None,
            #    s_log_library: Optional[torch.Tensor] = None,
            #    u_log_library_loc: Optional[torch.Tensor] = None,
            #    s_log_library_loc: Optional[torch.Tensor] = None,
            #    u_log_library_scale: Optional[torch.Tensor] = None,
            #    s_log_library_scale: Optional[torch.Tensor] = None,
            #    ind_x: Optional[torch.Tensor] = None,
            # ):
            #    u_lib_encoder = TimeEncoder2(u_obs.shape[1], 1, n_layers=1)
            #    s_lib_encoder = TimeEncoder2(s_obs.shape[1], 1, n_layers=1)
            #    pyro.module('u_lib_encoder', u_lib_encoder)
            #    pyro.module('s_lib_encoder', s_lib_encoder)
            #    samples = {}
            #    with guide.plates["cells"]:
            #        u_lib_loc, u_lib_var = u_lib_encoder(torch.log1p(u_obs))
            #        s_lib_loc, s_lib_var = s_lib_encoder(torch.log1p(s_obs))
            #        samples['u_read_depth'] = pyro.sample("u_read_depth", LogNormal(u_lib_loc, u_lib_var))
            #        samples['s_read_depth'] = pyro.sample("s_read_depth", LogNormal(s_lib_loc, s_lib_var))
            #    return samples
            # guide.append(localguide)
            self._guide = guide
        elif guide_type == "velocity_auto_depth":  # no constraint on t0
            guide = VelocityAutoGuideList(
                self._model, create_plates=self._model.create_plates
            )
            if correct_library_size:
                guide.append(
                    AutoNormal(
                        poutine.block(
                            self._model, expose=["u_read_depth", "s_read_depth"]
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
        else:
            if self.model_type == "latentfactor":
                self._guide = LatentGuide(
                    self._model,
                    plate_size=plate_size,
                    inducing_point_size=inducing_point_size,
                    **initial_values,
                )
        self._get_fn_args_from_batch = self._model._get_fn_args_from_batch

    @property
    def model(self) -> VelocityModelAuto:
        return self._model

    @property
    def guide(self) -> AutoGuideList:
        return self._guide
