"""Zooplankton Functional Group Factory.

This module provides tools to create configured FunctionalGroup instances
for the zooplankton model, supporting different vertical behaviors.
"""

from typing import Any, Literal

from seapopym_message.core.group import FunctionalGroup
from seapopym_message.kernels.sensing import diel_migration, extract_layer
from seapopym_message.kernels.zooplankton import (
    age_production,
    compute_mortality_unit,
    compute_recruitment,
    compute_tau_r_unit,
    update_biomass,
)


def zooplankton_group(
    name: str,
    behavior: Literal["epipelagic", "migrant"],
    params: dict[str, Any],
    forcing_map: dict[str, str] | None = None,
) -> FunctionalGroup:
    """Create a configured FunctionalGroup for zooplankton.

    Args:
        name: Name of the group (e.g., "epipelagic_zoo").
        behavior: Vertical behavior type:
            - "epipelagic": Stays in a fixed layer (defined by params['layer_index']).
            - "migrant": Migrates between day and night layers (defined by params).
        params: Dictionary of parameters. Must include:
            - For "epipelagic": layer_index
            - For "migrant": day_layer_index, night_layer_index
            - Biological params: tau_r0, gamma_tau_r, T_ref, lambda_0, gamma_lambda, n_ages, E
        forcing_map: Optional mapping for global forcings. Defaults to:
            - "forcing_3d": "forcing/temperature_3d"
            - "day_length": "forcing/day_length"
            - "npp": "forcing/npp"

    Returns:
        Configured FunctionalGroup instance.
    """
    if forcing_map is None:
        forcing_map = {}

    # Default forcing names
    global_temp_3d = forcing_map.get("forcing_3d", "forcing/temperature_3d")
    global_day_length = forcing_map.get("day_length", "forcing/day_length")
    global_npp = forcing_map.get("npp", "forcing/npp")

    units: list[Any] = []  # Type will be Unit, compatible with FunctionalGroup
    variable_map = {}

    # 1. Sensing Unit (Temperature Perception)
    # Maps global 3D temperature -> Group-specific effective temperature
    if behavior == "epipelagic":
        units.append(extract_layer)
        variable_map["forcing_nd"] = global_temp_3d  # Changed from forcing_3d
        variable_map["forcing_2d"] = f"{name}/temperature"  # Output effective temp

    elif behavior == "migrant":
        units.append(diel_migration)
        variable_map["forcing_nd"] = global_temp_3d  # Changed from forcing_3d
        variable_map["day_length"] = global_day_length
        variable_map["forcing_effective"] = f"{name}/temperature"  # Output effective temp

    else:
        raise ValueError(f"Unknown behavior: {behavior}")

    # 2. Physiology Units (Compute rates from temperature)
    units.append(compute_tau_r_unit)
    units.append(compute_mortality_unit)

    # Map inputs/outputs for physiology
    # They read 'temperature' (internal) -> mapped to '{name}/temperature'
    variable_map["temperature"] = f"{name}/temperature"
    variable_map["tau_r"] = f"{name}/tau_r"
    variable_map["mortality"] = f"{name}/mortality"

    # 3. Dynamics Units (Biomass & Production evolution)
    # Order is critical: Recruitment -> Age Production -> Biomass
    units.append(compute_recruitment)
    units.append(age_production)
    units.append(update_biomass)

    # Map inputs/outputs for dynamics
    variable_map["production"] = f"{name}/production"
    variable_map["recruitment"] = f"{name}/recruitment"
    variable_map["biomass"] = f"{name}/biomass"
    variable_map["npp"] = global_npp

    return FunctionalGroup(
        name=name,
        units=units,
        variable_map=variable_map,
        params=params,
    )
