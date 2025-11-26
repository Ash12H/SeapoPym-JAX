"""Configuration for LMTL model."""

from dataclasses import dataclass


@dataclass
class LMTLParams:
    """Parameters for a Low Mid Trophic Level functional group."""

    # Vertical migration
    day_layer: int | float
    night_layer: int | float

    # Recruitment age
    tau_r_0: float  # Maximum recruitment age at T_ref [days]
    gamma_tau_r: float  # Thermal sensitivity of recruitment age [1/degC]

    # Production transfer
    E: float  # Transfer efficiency from primary production

    # Mortality
    lambda_0: float  # Mortality rate at T_ref [1/day]
    gamma_lambda: float  # Thermal sensitivity of mortality [1/degC]

    # Reference temperature
    T_ref: float = 0.0  # [degC]
