"""
Tests for the interface definitions in PyroVelocity's modular architecture.

This module contains tests for the Protocol interfaces defined in interfaces.py,
verifying that they correctly define the contract that component implementations
must follow.
"""

from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import pyro
import pytest
import torch

from pyrovelocity.models.modular.interfaces import (
    BatchTensor,
    DynamicsModel,
    InferenceGuide,
    LikelihoodModel,
    ModelState,
    ObservationModel,
    ParamTensor,
    PriorModel,
)


class TestDynamicsModelInterface:
    """Tests for the DynamicsModel interface."""

    def test_dynamics_model_protocol(self):
        """Test that DynamicsModel is a runtime checkable Protocol."""
        assert isinstance(DynamicsModel, type)
        assert issubclass(DynamicsModel, Protocol)

    def test_dynamics_model_implementation(self):
        """Test that a class implementing the DynamicsModel interface is recognized."""
        class TestDynamicsModelImpl:
            def forward(
                self,
                u: BatchTensor,
                s: BatchTensor,
                alpha: ParamTensor,
                beta: ParamTensor,
                gamma: ParamTensor,
                scaling: Optional[ParamTensor] = None,
                t: Optional[BatchTensor] = None,
                **kwargs: Any,
            ) -> Tuple[BatchTensor, BatchTensor]:
                # Simple implementation for testing
                return u, s

            def steady_state(
                self,
                alpha: ParamTensor,
                beta: ParamTensor,
                gamma: ParamTensor,
                **kwargs: Any,
            ) -> Tuple[ParamTensor, ParamTensor]:
                # Simple implementation for testing
                return alpha / beta, alpha / gamma

        # Create an instance of the implementation
        impl = TestDynamicsModelImpl()

        # Check that it's recognized as implementing the Protocol
        assert isinstance(impl, DynamicsModel)


class TestPriorModelInterface:
    """Tests for the PriorModel interface."""

    def test_prior_model_protocol(self):
        """Test that PriorModel is a runtime checkable Protocol."""
        assert isinstance(PriorModel, type)
        assert issubclass(PriorModel, Protocol)

    def test_prior_model_implementation(self):
        """Test that a class implementing the PriorModel interface is recognized."""
        class TestPriorModelImpl:
            def forward(
                self,
                u_obs: BatchTensor,
                s_obs: BatchTensor,
                plate: pyro.plate,
                **kwargs: Any,
            ) -> ModelState:
                # Simple implementation for testing
                return {"alpha": None, "beta": None, "gamma": None}

        # Create an instance of the implementation
        impl = TestPriorModelImpl()

        # Check that it's recognized as implementing the Protocol
        assert isinstance(impl, PriorModel)


class TestLikelihoodModelInterface:
    """Tests for the LikelihoodModel interface."""

    def test_likelihood_model_protocol(self):
        """Test that LikelihoodModel is a runtime checkable Protocol."""
        assert isinstance(LikelihoodModel, type)
        assert issubclass(LikelihoodModel, Protocol)

    def test_likelihood_model_implementation(self):
        """Test that a class implementing the LikelihoodModel interface is recognized."""
        class TestLikelihoodModelImpl:
            def forward(
                self,
                u_obs: BatchTensor,
                s_obs: BatchTensor,
                u_logits: BatchTensor,
                s_logits: BatchTensor,
                plate: pyro.plate,
                **kwargs: Any,
            ) -> None:
                # Simple implementation for testing
                pass

        # Create an instance of the implementation
        impl = TestLikelihoodModelImpl()

        # Check that it's recognized as implementing the Protocol
        assert isinstance(impl, LikelihoodModel)


class TestObservationModelInterface:
    """Tests for the ObservationModel interface."""

    def test_observation_model_protocol(self):
        """Test that ObservationModel is a runtime checkable Protocol."""
        assert isinstance(ObservationModel, type)
        assert issubclass(ObservationModel, Protocol)

    def test_observation_model_implementation(self):
        """Test that a class implementing the ObservationModel interface is recognized."""
        class TestObservationModelImpl:
            def forward(
                self,
                u_obs: BatchTensor,
                s_obs: BatchTensor,
                **kwargs: Any,
            ) -> Tuple[BatchTensor, BatchTensor]:
                # Simple implementation for testing
                return u_obs, s_obs

        # Create an instance of the implementation
        impl = TestObservationModelImpl()

        # Check that it's recognized as implementing the Protocol
        assert isinstance(impl, ObservationModel)


class TestInferenceGuideInterface:
    """Tests for the InferenceGuide interface."""

    def test_inference_guide_protocol(self):
        """Test that InferenceGuide is a runtime checkable Protocol."""
        assert isinstance(InferenceGuide, type)
        assert issubclass(InferenceGuide, Protocol)

    def test_inference_guide_implementation(self):
        """Test that a class implementing the InferenceGuide interface is recognized."""
        class TestInferenceGuideImpl:
            def __call__(
                self,
                model: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                # Simple implementation for testing
                return lambda *args, **kwargs: None

        # Create an instance of the implementation
        impl = TestInferenceGuideImpl()

        # Check that it's recognized as implementing the Protocol
        assert isinstance(impl, InferenceGuide)
