import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import mlflow
import numpy as np
import pyro
import torch
from pyro.infer import TraceEnum_ELBO
from pyro.infer.autoguide.guides import AutoGuideList
from pyro.optim.clipped_adam import ClippedAdam
from pyro.optim.optim import PyroOptim
from scvi.dataloaders import DataSplitter
from scvi.train import PyroTrainingPlan
from scvi.train import TrainRunner

from ._velocity_module import VelocityModule


class VelocityAdam(ClippedAdam):
    def step(self, closure: Optional[Callable] = None) -> Optional[Any]:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            group["lr"] *= group["lrd"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad.clamp_(-group["clip_norm"], group["clip_norm"])
                # freeze non-velocity gene gradient
                grad[grad.isnan()] = 0.0
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


def VelocityClippedAdam(optim_args: Dict[str, float]) -> PyroOptim:
    """
    Wraps :class:`pyro.optim.clipped_adam.ClippedAdam` with :class:`~pyro.optim.optim.PyroOptim`.
    """
    return PyroOptim(VelocityAdam, optim_args)


class EnumTrainingPlan(PyroTrainingPlan):
    def __init__(
        self,
        pyro_velocity: VelocityModule,
        optim: Optional[pyro.optim.PyroOptim] = None,
    ):
        super().__init__(
            pyro_velocity, TraceEnum_ELBO(strict_enumeration_warning=True), optim
        )
        self.svi = pyro.infer.SVI(
            model=self.pyro_model,
            guide=self.pyro_guide,
            optim=self.optim,
            loss=self.loss_fn,
        )
        self.n_elem = self.module.num_genes * self.module.num_cells * 2

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        loss = self.svi.step(*args, **kwargs)
        return {
            "train_step_loss": loss,
            "num_elem": args[0].shape[0] * args[0].shape[1] * 2,
        }

    def training_epoch_end(self, outputs):
        n_batch, elbo = 0, 0
        n_elem = 0

        for tensors in outputs:
            elbo += tensors["train_step_loss"]
            n_batch += 1
            # n_elem += tensors['num_elem']
        self.log("elbo_train", elbo / n_batch, prog_bar=True, on_epoch=True)
        # self.log("elbo_train", elbo / self.n_elem, prog_bar=True, on_epoch=True)
        # self.log("elbo_train", elbo / n_elem, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        loss = self.svi.evaluate_loss(*args, **kwargs)
        return {
            "valid_step_loss": loss,
            "num_elem": args[0].shape[0] * args[0].shape[1] * 2,
        }

    def validation_epoch_end(self, outputs):
        """Aggregate validation step information."""
        n_batch, elbo = 0, 0
        n_elem = 0
        for tensors in outputs:
            elbo += tensors["valid_step_loss"]
            n_batch += 1
            ##n_elem += tensors['num_elem']
        self.log("elbo_validation", elbo / n_batch, prog_bar=True, on_epoch=True)
        # self.log("elbo_validation", elbo / self.n_elem, prog_bar=True, on_epoch=True)
        ##self.log("elbo_validation", elbo / n_elem, prog_bar=True, on_epoch=True)


class VelocityTrainingMixin:
    def train(
        self,
        use_gpu: Optional[Union[str, int, bool]] = 0,
        early_stopping: bool = False,
        seed: int = 99,
        lr: float = 1e-3,
        train_size: float = 1.0,
        valid_size: float = 0.0,
        batch_size: int = 256,
        max_epochs: int = 100,
        check_val_every_n_epoch: Optional[int] = 1,
        patience: int = 10,
        min_delta: float = 0.0,
        **kwargs,
    ):
        print("base train function")
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)
        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=valid_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = EnumTrainingPlan(
            self.module, VelocityClippedAdam({"lr": lr, "lrd": 0.9999})
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            check_val_every_n_epoch=check_val_every_n_epoch,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping=early_stopping,
            early_stopping_patience=patience,
            early_stopping_min_delta=min_delta,
            **kwargs,
        )
        return runner()

    def train_faster(
        self,
        use_gpu: Optional[Union[str, int, bool]] = 0,
        seed: int = 99,
        lr: float = 0.01,
        max_epochs: int = 5000,
        log_every: int = 100,
        patient_init: int = 45,
        patient_improve: float = 0.001,
    ) -> List[float]:
        """this method input all adata input gpu for faster IO,
        increase larger dataset by 5-6 fold, however should not
        be used for >20k cells with less than 40GB GPU memory,
        this ignores validation cells"""
        print("train_faster")
        if (use_gpu is False) or (use_gpu == "cpu") or (use_gpu == -1):
            device = "cpu"
        else:
            device = f"cuda:{use_gpu}"
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)
        pyro.enable_validation(True)
        optim = VelocityClippedAdam({"lr": lr, "lrd": 0.1 ** (1 / max_epochs)})
        self.module._model = self.module._model.to(device)
        print("TraceEnum")
        svi = pyro.infer.SVI(
            self.module._model,
            self.module._guide,
            optim,
            TraceEnum_ELBO(strict_enumeration_warning=True),
        )

        normalizer = self.adata.shape[0] * self.adata.shape[1] * 2
        u = torch.tensor(
            np.array(self.adata.layers["raw_unspliced"].toarray(), dtype="float32"),
            dtype=torch.float32,
        ).to(device)
        s = torch.tensor(
            np.array(self.adata.layers["raw_spliced"].toarray(), dtype="float32"),
            dtype=torch.float32,
        ).to(device)
        u_library = torch.tensor(
            np.array(self.adata.obs.u_lib_size, dtype="float32"), dtype=torch.float32
        ).to(device)
        s_library = torch.tensor(
            np.array(self.adata.obs.s_lib_size, dtype="float32"), dtype=torch.float32
        ).to(device)
        u_library_mean = torch.tensor(
            np.array(self.adata.obs.u_lib_size_mean, dtype="float32"),
            dtype=torch.float32,
        ).to(device)
        s_library_mean = torch.tensor(
            np.array(self.adata.obs.s_lib_size_mean, dtype="float32"),
            dtype=torch.float32,
        ).to(device)
        u_library_scale = torch.tensor(
            np.array(self.adata.obs.u_lib_size_scale, dtype="float32"),
            dtype=torch.float32,
        ).to(device)
        s_library_scale = torch.tensor(
            np.array(self.adata.obs.s_lib_size_scale, dtype="float32"),
            dtype=torch.float32,
        ).to(device)
        if "pyro_cell_state" in self.adata.obs.columns:
            cell_state = torch.tensor(
                np.array(self.adata.obs.pyro_cell_state, dtype="float32"),
                dtype=torch.float32,
            ).to(device)
        else:
            cell_state = None

        losses = []
        patience = patient_init
        for step in range(max_epochs):
            if cell_state is None:
                elbos = (
                    svi.step(
                        u,
                        s,
                        u_library.reshape(-1, 1),
                        s_library.reshape(-1, 1),
                        u_library_mean.reshape(-1, 1),
                        s_library_mean.reshape(-1, 1),
                        u_library_scale.reshape(-1, 1),
                        s_library_scale.reshape(-1, 1),
                        None,
                        None,
                    )
                    / normalizer
                )
            else:
                elbos = (
                    svi.step(
                        u,
                        s,
                        u_library.reshape(-1, 1),
                        s_library.reshape(-1, 1),
                        u_library_mean.reshape(-1, 1),
                        s_library_mean.reshape(-1, 1),
                        u_library_scale.reshape(-1, 1),
                        s_library_scale.reshape(-1, 1),
                        None,
                        cell_state.reshape(-1, 1),
                    )
                    / normalizer
                )
            if (step == 0) or (
                ((step + 1) % log_every == 0) and ((step + 1) < max_epochs)
            ):
                mlflow.log_metric("-ELBO", -elbos, step=step + 1)
                print(f"step {step + 1: >4d} loss = {elbos:0.6g} patience = {patience}")
            if step > log_every:
                if (losses[-1] - elbos) < losses[-1] * patient_improve:
                    patience -= 1
                else:
                    patience = patient_init
            if patience <= 0:
                break
            losses.append(elbos)
        mlflow.log_metric("-ELBO", -elbos, step=step + 1)
        mlflow.log_metric("real_epochs", step + 1)
        print(f"step {step + 1: >4d} loss = {elbos:0.6g} patience = {patience}")
        return losses

    def train_faster_with_batch(
        self,
        use_gpu: Optional[Union[str, int, bool]] = 0,
        seed: int = 99,
        lr: float = 1e-2,
        max_epochs: int = 5000,
        log_every: int = 100,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        new_valid_guide: Optional[AutoGuideList] = None,
        patient_init: int = 45,
        patient_improve: float = 0.0,
        elbo_name: str = "-ELBO",
    ):
        print("train_faster_with_batch")
        if (use_gpu is False) or (use_gpu == "cpu") or (use_gpu == -1):
            device = "cpu"
        else:
            device = f"cuda:{use_gpu}"
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)
        pyro.enable_validation(True)

        adata = self._validate_anndata(self.adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        optim = VelocityClippedAdam({"lr": lr, "lrd": 0.1 ** (1 / max_epochs)})
        self.module._model = self.module._model.to(device)

        if new_valid_guide is None:
            svi = pyro.infer.SVI(
                self.module._model,
                self.module._guide,
                optim,
                TraceEnum_ELBO(strict_enumeration_warning=True),
            )
        else:
            svi = pyro.infer.SVI(
                self.module._model,
                new_valid_guide,
                optim,
                TraceEnum_ELBO(strict_enumeration_warning=True),
            )

        losses = []
        patience = patient_init
        # normalizer = self.adata.shape[0] * self.adata.shape[1] * 2
        for step in range(max_epochs):
            n_batch = 0
            elbos = 0
            for tensor_dict in scdl:
                args, kwargs = self.module._get_fn_args_from_batch(tensor_dict)
                args = [a.to(device) if a is not None else a for a in args]
                loss = svi.step(*args, **kwargs)
                elbos += loss
                n_batch += 1
            # elbos = elbos / normalizer #n_batch
            elbos = elbos / n_batch
            if (step == 0) or (
                ((step + 1) % log_every == 0) and ((step + 1) < max_epochs)
            ):
                mlflow.log_metric("-ELBO", -elbos, step=step + 1)
                print(f"step {step + 1: >4d} loss = {elbos:0.6g} patience = {patience}")
            if step > log_every:
                if (losses[-1] - elbos) < losses[-1] * patient_improve:
                    patience -= 1
                    # print(f"step {step: >4d} loss = {elbos:0.6g} patience = {patience}")
                else:
                    patience = patient_init
            if patience <= 0:
                break
            losses.append(elbos)
        mlflow.log_metric("-ELBO", -elbos, step=step + 1)
        mlflow.log_metric("real_epochs", step + 1)
        print(f"step {step: >4d} loss = {elbos:0.6g} patience = {patience}")
        return losses
