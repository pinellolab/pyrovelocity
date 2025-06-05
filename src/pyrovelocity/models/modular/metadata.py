"""
Parameter metadata definitions for PyroVelocity modular components.

This module contains parameter metadata definitions for different model components,
providing standardized information about parameter semantics, display formatting,
and biological interpretations.
"""

from typing import Dict

from pyrovelocity.models.modular.interfaces import (
    ComponentParameterMetadata,
    ParameterMetadata,
)


def create_piecewise_activation_prior_metadata() -> ComponentParameterMetadata:
    """
    Create parameter metadata for the piecewise activation prior model.
    
    This function defines metadata for all parameters in the piecewise activation
    prior model, based on the mathematical descriptions and biological interpretations
    from the validation study documentation.
    
    Returns:
        ComponentParameterMetadata for the piecewise activation prior model.
    """
    parameters = {
        # Hierarchical time parameters
        "T_M_star": ParameterMetadata(
            name="T_M_star",
            display_name=r"$T^*_M$",
            short_label="Max Time",
            description="Maximum dimensionless time coordinate for the population",
            units="dimensionless time",
            typical_range=(1.0, 10.0),
            biological_interpretation="Overall duration of the biological process being modeled",
            plot_order=1
        ),
        
        "t_loc": ParameterMetadata(
            name="t_loc",
            display_name=r"$t_{loc}$",
            short_label="Time Loc",
            description="Location parameter for hierarchical time distribution",
            units="dimensionless",
            typical_range=(0.1, 2.0),
            biological_interpretation="Central tendency of cell time coordinates in the population",
            plot_order=2
        ),
        
        "t_scale": ParameterMetadata(
            name="t_scale",
            display_name=r"$t_{scl}$",
            short_label="Time Scale",
            description="Scale parameter for hierarchical time distribution",
            units="dimensionless",
            typical_range=(0.1, 1.0),
            biological_interpretation="Spread of cell time coordinates around the population mean",
            plot_order=3
        ),
        
        "t_star": ParameterMetadata(
            name="t_star",
            display_name=r"$t^*$",
            short_label="Cell Time",
            description="Individual cell latent time coordinates",
            units="dimensionless time",
            typical_range=(0.0, 10.0),
            biological_interpretation="Progression of individual cells through the biological process",
            plot_order=4
        ),
        
        # Piecewise activation parameters
        "alpha_off": ParameterMetadata(
            name="alpha_off",
            display_name=r"$\alpha_{off}$",
            short_label="Basal Rate",
            description="Dimensionless basal transcription rate during inactive phase",
            units="dimensionless rate",
            typical_range=(0.01, 0.5),
            biological_interpretation="Low expression state transcription activity, representing repressed gene expression",
            plot_order=5
        ),
        
        "alpha_on": ParameterMetadata(
            name="alpha_on",
            display_name=r"$\alpha_{on}$",
            short_label="Active Rate",
            description="Dimensionless active transcription rate during activation phase",
            units="dimensionless rate",
            typical_range=(0.5, 5.0),
            biological_interpretation="High expression state transcription activity, representing activated gene expression",
            plot_order=6
        ),
        
        "gamma_star": ParameterMetadata(
            name="gamma_star",
            display_name=r"$\gamma^*$",
            short_label="Decay Rate",
            description="Relative degradation rate (splicing vs degradation balance)",
            units="dimensionless rate ratio",
            typical_range=(0.3, 3.0),
            biological_interpretation="Balance between mRNA splicing and degradation kinetics; γ*=1 represents balanced kinetics",
            plot_order=7
        ),
        
        "t_on_star": ParameterMetadata(
            name="t_on_star",
            display_name=r"$t^*_{on}$",
            short_label="Onset Time",
            description="Dimensionless activation onset time",
            units="dimensionless time",
            typical_range=(0.1, 0.8),
            biological_interpretation="When during the process each gene begins its activation phase",
            plot_order=8
        ),
        
        "delta_star": ParameterMetadata(
            name="delta_star",
            display_name=r"$\delta^*$",
            short_label="Duration",
            description="Dimensionless activation duration",
            units="dimensionless time",
            typical_range=(0.1, 1.0),
            biological_interpretation="How long each gene remains in its activated state",
            plot_order=9
        ),
        
        # Observation model parameters
        "U_0i": ParameterMetadata(
            name="U_0i",
            display_name=r"$U_{0i}$",
            short_label="Scale Factor",
            description="Gene-specific scaling factor for observation model",
            units="count scale",
            typical_range=(10.0, 1000.0),
            biological_interpretation="Gene-specific expression scale, accounting for differences in gene expression levels",
            plot_order=10
        ),
        
        "lambda_j": ParameterMetadata(
            name="lambda_j",
            display_name=r"$\lambda_j$",
            short_label="Capture Eff",
            description="Cell-specific capture efficiency",
            units="efficiency ratio",
            typical_range=(0.1, 2.0),
            biological_interpretation="Technical variation in RNA capture and sequencing efficiency across cells",
            plot_order=11
        ),
    }
    
    return ComponentParameterMetadata(
        component_name="piecewise_activation_prior",
        component_type="prior",
        parameters=parameters,
        description="Prior distributions for piecewise activation model with hierarchical time structure"
    )


def create_lognormal_prior_metadata() -> ComponentParameterMetadata:
    """
    Create parameter metadata for the standard log-normal prior model.
    
    Returns:
        ComponentParameterMetadata for the log-normal prior model.
    """
    parameters = {
        "alpha": ParameterMetadata(
            name="alpha",
            display_name=r"$\alpha$",
            short_label="Transcription",
            description="Transcription rate parameter",
            units="rate",
            typical_range=(0.1, 10.0),
            biological_interpretation="Rate of pre-mRNA transcription",
            plot_order=1
        ),
        
        "beta": ParameterMetadata(
            name="beta",
            display_name=r"$\beta$",
            short_label="Splicing",
            description="Splicing rate parameter",
            units="rate",
            typical_range=(0.1, 10.0),
            biological_interpretation="Rate of pre-mRNA splicing to mature mRNA",
            plot_order=2
        ),
        
        "gamma": ParameterMetadata(
            name="gamma",
            display_name=r"$\gamma$",
            short_label="Degradation",
            description="mRNA degradation rate parameter",
            units="rate",
            typical_range=(0.1, 10.0),
            biological_interpretation="Rate of mature mRNA degradation",
            plot_order=3
        ),
        
        "scaling": ParameterMetadata(
            name="scaling",
            display_name=r"$s$",
            short_label="Scaling",
            description="Global scaling factor",
            units="scale factor",
            typical_range=(0.1, 10.0),
            biological_interpretation="Overall expression scale adjustment",
            plot_order=4
        ),
    }
    
    return ComponentParameterMetadata(
        component_name="lognormal_prior",
        component_type="prior",
        parameters=parameters,
        description="Log-normal prior distributions for standard RNA velocity parameters"
    )


# Registry of metadata providers
PARAMETER_METADATA_REGISTRY: Dict[str, ComponentParameterMetadata] = {
    "piecewise_activation_prior": create_piecewise_activation_prior_metadata(),
    "lognormal_prior": create_lognormal_prior_metadata(),
}


def get_parameter_metadata(component_name: str) -> ComponentParameterMetadata:
    """
    Get parameter metadata for a named component.
    
    Args:
        component_name: Name of the component to get metadata for
        
    Returns:
        ComponentParameterMetadata for the component, or None if not found
        
    Raises:
        KeyError: If component_name is not found in the registry
    """
    if component_name not in PARAMETER_METADATA_REGISTRY:
        raise KeyError(f"No parameter metadata found for component '{component_name}'")
    
    return PARAMETER_METADATA_REGISTRY[component_name]


def get_parameter_short_labels(component_name: str) -> Dict[str, str]:
    """
    Get short labels for parameters of a named component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        Dictionary mapping parameter names to short labels
    """
    try:
        metadata = get_parameter_metadata(component_name)
        return metadata.get_short_labels()
    except KeyError:
        return {}


def get_parameter_display_names(component_name: str) -> Dict[str, str]:
    """
    Get LaTeX display names for parameters of a named component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        Dictionary mapping parameter names to LaTeX display names
    """
    try:
        metadata = get_parameter_metadata(component_name)
        return metadata.get_display_names()
    except KeyError:
        return {}
