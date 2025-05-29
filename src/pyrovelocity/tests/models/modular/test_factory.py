"""
Tests for the factory module in PyroVelocity's modular architecture.

This module contains tests for the factory functions and configuration management
provided by the factory module, verifying that models can be created from
configurations correctly.
"""

from typing import Any, Dict, Optional, Tuple, cast

import pyro
import pytest
import torch
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf

from pyrovelocity.models.modular.factory import (
    DynamicsModelConfig,
    InferenceGuideConfig,
    LikelihoodModelConfig,
    ObservationModelConfig,
    PriorModelConfig,
    PyroVelocityModelConfig,
    create_dynamics_model,
    create_inference_guide,
    create_likelihood_model,
    create_model,
    create_model_from_config,
    create_observation_model,
    create_prior_model,
    create_protocol_first_model,
    create_protocol_first_model_from_config,
    create_standard_model,
    standard_model_config,
)
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
from pyrovelocity.models.modular.model import PyroVelocityModel
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    InferenceGuideRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    PriorModelRegistry,
)


# Mock implementations for testing
class MockDynamicsModel:
    """Mock dynamics model for testing."""

    def __init__(self, param1=None, param2=None, shared_time=None, cell_specific_kinetics=None,
                 correct_library_size=None, kinetics_num=None, **kwargs):
        self.param1 = param1
        self.param2 = param2
        self.shared_time = shared_time
        self.cell_specific_kinetics = cell_specific_kinetics
        self.correct_library_size = correct_library_size
        self.kinetics_num = kinetics_num

    def forward(
        self,
        u: BatchTensor,
        s: BatchTensor,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        scaling: Optional[ParamTensor] = None,
        t: Optional[BatchTensor] = None,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """Implement the DynamicsModel protocol."""
        return u, s


class MockPriorModel:
    """Mock prior model for testing."""

    def __init__(self, param1=None, param2=None, scale_alpha=None, scale_beta=None,
                 scale_gamma=None, scale_u=None, scale_s=None, scale_dt=None, **kwargs):
        self.param1 = param1
        self.param2 = param2
        self.scale_alpha = scale_alpha
        self.scale_beta = scale_beta
        self.scale_gamma = scale_gamma
        self.scale_u = scale_u
        self.scale_s = scale_s
        self.scale_dt = scale_dt

    def forward(
        self,
        u_obs: BatchTensor,
        s_obs: BatchTensor,
        plate: pyro.plate,
        **kwargs: Any,
    ) -> ModelState:
        """Implement the PriorModel protocol."""
        return {"alpha": None, "beta": None, "gamma": None}


class MockLikelihoodModel:
    """Mock likelihood model for testing."""

    def __init__(self, param1=None, param2=None, use_observed_lib_size=True, **kwargs):
        self.param1 = param1
        self.param2 = param2
        self.use_observed_lib_size = use_observed_lib_size

    def forward(
        self,
        u_obs: BatchTensor,
        s_obs: BatchTensor,
        u_logits: BatchTensor,
        s_logits: BatchTensor,
        plate: pyro.plate,
        **kwargs: Any,
    ) -> None:
        """Implement the LikelihoodModel protocol."""
        pass


class MockObservationModel:
    """Mock observation model for testing."""

    def __init__(self, param1=None, param2=None, log_transform=None, normalize=None,
                 use_raw=None, **kwargs):
        self.param1 = param1
        self.param2 = param2
        self.log_transform = log_transform
        self.normalize = normalize
        self.use_raw = use_raw

    def forward(
        self,
        u_obs: BatchTensor,
        s_obs: BatchTensor,
        **kwargs: Any,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """Implement the ObservationModel protocol."""
        return u_obs, s_obs


class MockInferenceGuide:
    """Mock inference guide for testing."""

    def __init__(self, param1=None, param2=None, guide_type=None, init_loc_fn=None,
                 init_scale=None, **kwargs):
        self.param1 = param1
        self.param2 = param2
        self.guide_type = guide_type
        self.init_loc_fn = init_loc_fn
        self.init_scale = init_scale

    def __call__(
        self,
        model,
        *args: Any,
        **kwargs: Any,
    ):
        """Implement the InferenceGuide protocol."""

        def guide(*args, **kwargs):
            return None

        return guide


@pytest.fixture
def setup_registries():
    """Set up the registries with mock components for testing."""
    # Clear all registries
    DynamicsModelRegistry.clear()
    PriorModelRegistry.clear()
    LikelihoodModelRegistry.clear()
    ObservationModelRegistry.clear()
    InferenceGuideRegistry.clear()

    # Make the mock classes implement the required protocols
    # This is a hack to make beartype happy with our mock classes
    DynamicsModel.register(MockDynamicsModel)
    PriorModel.register(MockPriorModel)
    LikelihoodModel.register(MockLikelihoodModel)
    ObservationModel.register(MockObservationModel)
    InferenceGuide.register(MockInferenceGuide)

    # Register mock components
    @DynamicsModelRegistry.register("mock")
    class RegisteredMockDynamicsModel(MockDynamicsModel):
        pass

    @PriorModelRegistry.register("mock")
    class RegisteredMockPriorModel(MockPriorModel):
        pass

    @LikelihoodModelRegistry.register("mock")
    class RegisteredMockLikelihoodModel(MockLikelihoodModel):
        pass

    @ObservationModelRegistry.register("mock")
    class RegisteredMockObservationModel(MockObservationModel):
        pass

    @InferenceGuideRegistry.register("mock")
    class RegisteredMockInferenceGuide(MockInferenceGuide):
        pass

    # Register standard components (assuming these exist in the real codebase)
    @DynamicsModelRegistry.register("standard")
    class StandardDynamicsModel(MockDynamicsModel):
        pass

    @PriorModelRegistry.register("lognormal")
    class LogNormalPriorModel(MockPriorModel):
        pass

    @LikelihoodModelRegistry.register("poisson")
    class PoissonLikelihoodModel(MockLikelihoodModel):
        pass

    @ObservationModelRegistry.register("standard")
    class StandardObservationModel(MockObservationModel):
        pass

    @InferenceGuideRegistry.register("auto")
    class AutoGuide(MockInferenceGuide):
        pass


class TestFactoryFunctions:
    """Tests for the individual factory functions."""

    def test_create_dynamics_model(self, setup_registries):
        """Test creating a dynamics model from a configuration."""
        # Test with a DynamicsModelConfig object
        config = DynamicsModelConfig(
            name="mock", params={"param1": 42, "param2": "test"}
        )
        model = create_dynamics_model(config)
        assert isinstance(model, MockDynamicsModel)
        assert model.param1 == 42
        assert model.param2 == "test"

        # Test with a dictionary
        config_dict = {
            "name": "mock",
            "params": {"param1": 43, "param2": "test2"},
        }
        model = create_dynamics_model(config_dict)
        assert isinstance(model, MockDynamicsModel)
        assert model.param1 == 43
        assert model.param2 == "test2"

        # Test with a DictConfig
        config_dict_config = OmegaConf.create(config_dict)
        model = create_dynamics_model(config_dict_config)
        assert isinstance(model, MockDynamicsModel)
        assert model.param1 == 43
        assert model.param2 == "test2"

    def test_create_prior_model(self, setup_registries):
        """Test creating a prior model from a configuration."""
        # Test with a PriorModelConfig object
        config = PriorModelConfig(
            name="mock", params={"param1": 42, "param2": "test"}
        )
        model = create_prior_model(config)
        assert isinstance(model, MockPriorModel)
        assert model.param1 == 42
        assert model.param2 == "test"

        # Test with a dictionary
        config_dict = {
            "name": "mock",
            "params": {"param1": 43, "param2": "test2"},
        }
        model = create_prior_model(config_dict)
        assert isinstance(model, MockPriorModel)
        assert model.param1 == 43
        assert model.param2 == "test2"

        # Test with a DictConfig
        config_dict_config = OmegaConf.create(config_dict)
        model = create_prior_model(config_dict_config)
        assert isinstance(model, MockPriorModel)
        assert model.param1 == 43
        assert model.param2 == "test2"

    def test_create_likelihood_model(self, setup_registries):
        """Test creating a likelihood model from a configuration."""
        # Test with a LikelihoodModelConfig object
        config = LikelihoodModelConfig(
            name="mock", params={"param1": 42, "param2": "test"}
        )
        model = create_likelihood_model(config)
        assert isinstance(model, MockLikelihoodModel)
        assert model.param1 == 42
        assert model.param2 == "test"

        # Test with a dictionary
        config_dict = {
            "name": "mock",
            "params": {"param1": 43, "param2": "test2"},
        }
        model = create_likelihood_model(config_dict)
        assert isinstance(model, MockLikelihoodModel)
        assert model.param1 == 43
        assert model.param2 == "test2"

        # Test with a DictConfig
        config_dict_config = OmegaConf.create(config_dict)
        model = create_likelihood_model(config_dict_config)
        assert isinstance(model, MockLikelihoodModel)
        assert model.param1 == 43
        assert model.param2 == "test2"

    def test_create_observation_model(self, setup_registries):
        """Test creating an observation model from a configuration."""
        # Test with an ObservationModelConfig object
        config = ObservationModelConfig(
            name="mock", params={"param1": 42, "param2": "test"}
        )
        model = create_observation_model(config)
        assert isinstance(model, MockObservationModel)
        assert model.param1 == 42
        assert model.param2 == "test"

        # Test with a dictionary
        config_dict = {
            "name": "mock",
            "params": {"param1": 43, "param2": "test2"},
        }
        model = create_observation_model(config_dict)
        assert isinstance(model, MockObservationModel)
        assert model.param1 == 43
        assert model.param2 == "test2"

        # Test with a DictConfig
        config_dict_config = OmegaConf.create(config_dict)
        model = create_observation_model(config_dict_config)
        assert isinstance(model, MockObservationModel)
        assert model.param1 == 43
        assert model.param2 == "test2"

    def test_create_inference_guide(self, setup_registries):
        """Test creating an inference guide from a configuration."""
        # Test with an InferenceGuideConfig object
        config = InferenceGuideConfig(
            name="mock", params={"param1": 42, "param2": "test"}
        )
        model = create_inference_guide(config)
        assert isinstance(model, MockInferenceGuide)
        assert model.param1 == 42
        assert model.param2 == "test"

        # Test with a dictionary
        config_dict = {
            "name": "mock",
            "params": {"param1": 43, "param2": "test2"},
        }
        model = create_inference_guide(config_dict)
        assert isinstance(model, MockInferenceGuide)
        assert model.param1 == 43
        assert model.param2 == "test2"

        # Test with a DictConfig
        config_dict_config = OmegaConf.create(config_dict)
        model = create_inference_guide(config_dict_config)
        assert isinstance(model, MockInferenceGuide)
        assert model.param1 == 43
        assert model.param2 == "test2"


class TestModelCreation:
    """Tests for creating complete models from configurations."""

    def test_create_model(self, setup_registries):
        """Test creating a PyroVelocityModel from a configuration."""
        # Create a configuration
        config = PyroVelocityModelConfig(
            dynamics_model=DynamicsModelConfig(
                name="mock", params={"param1": 1}
            ),
            prior_model=PriorModelConfig(name="mock", params={"param1": 2}),
            likelihood_model=LikelihoodModelConfig(
                name="mock", params={"param1": 3}
            ),
            inference_guide=InferenceGuideConfig(
                name="mock", params={"param1": 5}
            ),
        )

        # Create the model
        model = create_model(config)

        # Check that the model is a PyroVelocityModel
        assert isinstance(model, PyroVelocityModel)

        # Check that the components are of the correct types
        assert isinstance(model.dynamics_model, MockDynamicsModel)
        assert isinstance(model.prior_model, MockPriorModel)
        assert isinstance(model.likelihood_model, MockLikelihoodModel)
        assert isinstance(model.guide_model, MockInferenceGuide)

        # Check that the parameters were passed correctly
        assert model.dynamics_model.param1 == 1
        assert model.prior_model.param1 == 2
        assert model.likelihood_model.param1 == 3
        assert model.guide_model.param1 == 5

    def test_create_model_from_dict(self, setup_registries):
        """Test creating a PyroVelocityModel from a dictionary configuration."""
        # Create a configuration dictionary
        config_dict = {
            "dynamics_model": {"name": "mock", "params": {"param1": 1}},
            "prior_model": {"name": "mock", "params": {"param1": 2}},
            "likelihood_model": {"name": "mock", "params": {"param1": 3}},
            "inference_guide": {"name": "mock", "params": {"param1": 5}},
        }

        # Create the model
        model = create_model(config_dict)

        # Check that the model is a PyroVelocityModel
        assert isinstance(model, PyroVelocityModel)

        # Check that the components are of the correct types
        assert isinstance(model.dynamics_model, MockDynamicsModel)
        assert isinstance(model.prior_model, MockPriorModel)
        assert isinstance(model.likelihood_model, MockLikelihoodModel)
        assert isinstance(model.guide_model, MockInferenceGuide)

        # Check that the parameters were passed correctly
        assert model.dynamics_model.param1 == 1
        assert model.prior_model.param1 == 2
        assert model.likelihood_model.param1 == 3
        assert model.guide_model.param1 == 5

    def test_create_model_from_dictconfig(self, setup_registries):
        """Test creating a PyroVelocityModel from a DictConfig configuration."""
        # Create a configuration dictionary
        config_dict = {
            "dynamics_model": {"name": "mock", "params": {"param1": 1}},
            "prior_model": {"name": "mock", "params": {"param1": 2}},
            "likelihood_model": {"name": "mock", "params": {"param1": 3}},
            "inference_guide": {"name": "mock", "params": {"param1": 5}},
        }

        # Convert to DictConfig
        config = OmegaConf.create(config_dict)

        # Create the model
        model = create_model(config)

        # Check that the model is a PyroVelocityModel
        assert isinstance(model, PyroVelocityModel)

        # Check that the components are of the correct types
        assert isinstance(model.dynamics_model, MockDynamicsModel)
        assert isinstance(model.prior_model, MockPriorModel)
        assert isinstance(model.likelihood_model, MockLikelihoodModel)
        assert isinstance(model.guide_model, MockInferenceGuide)

        # Check that the parameters were passed correctly
        assert model.dynamics_model.param1 == 1
        assert model.prior_model.param1 == 2
        assert model.likelihood_model.param1 == 3
        assert model.guide_model.param1 == 5


class TestPredefinedConfigurations:
    """Tests for the predefined configurations."""

    def test_standard_model_config(self, setup_registries):
        """Test the standard model configuration."""
        # Get the standard model configuration
        config = standard_model_config()

        # Check that it has the expected component configurations
        assert config.dynamics_model.name == "standard"
        assert config.prior_model.name == "lognormal"
        assert config.likelihood_model.name == "poisson"
        assert config.inference_guide.name == "auto"

    def test_create_standard_model(self, setup_registries):
        """Test creating a standard model."""
        # Create a standard model
        model = create_standard_model()

        # Check that the model is a PyroVelocityModel
        assert isinstance(model, PyroVelocityModel)

        # Check that the components are of the correct types
        assert isinstance(model.dynamics_model, MockDynamicsModel)
        assert isinstance(model.prior_model, MockPriorModel)
        assert isinstance(model.likelihood_model, MockLikelihoodModel)
        assert isinstance(model.guide_model, MockInferenceGuide)


class TestProtocolFirstFactoryFunctions:
    """Tests for the Protocol-First factory functions."""

    def test_create_protocol_first_model(self, setup_registries):
        """Test creating a model with Protocol-First components."""
        # Register Protocol-First components
        @DynamicsModelRegistry.register("standard_direct")
        class StandardDynamicsModel(MockDynamicsModel):
            pass

        @PriorModelRegistry.register("lognormal_direct")
        class LogNormalPriorModel(MockPriorModel):
            pass

        @LikelihoodModelRegistry.register("poisson_direct")
        class PoissonLikelihoodModel(MockLikelihoodModel):
            pass

        @ObservationModelRegistry.register("standard_direct")
        class StandardObservationModel(MockObservationModel):
            pass

        @InferenceGuideRegistry.register("auto_direct")
        class AutoGuideFactory(MockInferenceGuide):
            pass

        # Create a model with Protocol-First components
        model = create_protocol_first_model()

        # Check that the model is a PyroVelocityModel
        assert isinstance(model, PyroVelocityModel)

        # Check that the components are of the correct types
        assert isinstance(model.dynamics_model, MockDynamicsModel)
        assert isinstance(model.prior_model, MockPriorModel)
        assert isinstance(model.likelihood_model, MockLikelihoodModel)
        assert isinstance(model.guide_model, MockInferenceGuide)

    def test_create_protocol_first_model_from_config(self, setup_registries):
        """Test creating a model with Protocol-First components from a configuration."""
        # Register Protocol-First components
        @DynamicsModelRegistry.register("standard_direct")
        class StandardDynamicsModel(MockDynamicsModel):
            pass

        @PriorModelRegistry.register("lognormal_direct")
        class LogNormalPriorModel(MockPriorModel):
            pass

        @LikelihoodModelRegistry.register("poisson_direct")
        class PoissonLikelihoodModel(MockLikelihoodModel):
            pass

        @ObservationModelRegistry.register("standard_direct")
        class StandardObservationModel(MockObservationModel):
            pass

        @InferenceGuideRegistry.register("auto_direct")
        class AutoGuideFactory(MockInferenceGuide):
            pass

        # Create a configuration
        from pyrovelocity.models.modular.config import (
            ComponentConfig,
            ModelConfig,
        )
        config = ModelConfig(
            dynamics_model=ComponentConfig(name="standard"),
            prior_model=ComponentConfig(name="lognormal"),
            likelihood_model=ComponentConfig(name="poisson"),
            inference_guide=ComponentConfig(name="auto"),
        )

        # Create a model with Protocol-First components from the configuration
        model = create_protocol_first_model_from_config(config)

        # Check that the model is a PyroVelocityModel
        assert isinstance(model, PyroVelocityModel)

        # Check that the components are of the correct types
        assert isinstance(model.dynamics_model, MockDynamicsModel)
        assert isinstance(model.prior_model, MockPriorModel)
        assert isinstance(model.likelihood_model, MockLikelihoodModel)
        assert isinstance(model.guide_model, MockInferenceGuide)


class TestHydraZenIntegration:
    """Tests for the hydra-zen integration."""

    def test_hydra_zen_instantiate(self, setup_registries):
        """Test instantiating a model using hydra-zen."""
        # Create a configuration dictionary
        config_dict = {
            "_target_": "pyrovelocity.models.modular.factory.create_model",
            "config": {
                "dynamics_model": {"name": "mock", "params": {"param1": 1}},
                "prior_model": {"name": "mock", "params": {"param1": 2}},
                "likelihood_model": {"name": "mock", "params": {"param1": 3}},
                "inference_guide": {"name": "mock", "params": {"param1": 5}},
            },
        }

        # Convert to DictConfig
        config = OmegaConf.create(config_dict)

        # Instantiate the model using hydra-zen
        model = instantiate(config)

        # Check that the model is a PyroVelocityModel
        assert isinstance(model, PyroVelocityModel)

        # Check that the components are of the correct types
        assert isinstance(model.dynamics_model, MockDynamicsModel)
        assert isinstance(model.prior_model, MockPriorModel)
        assert isinstance(model.likelihood_model, MockLikelihoodModel)
        assert isinstance(model.guide_model, MockInferenceGuide)

        # Check that the parameters were passed correctly
        assert model.dynamics_model.param1 == 1
        assert model.prior_model.param1 == 2
        assert model.likelihood_model.param1 == 3
        assert model.guide_model.param1 == 5
