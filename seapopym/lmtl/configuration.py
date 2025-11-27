"""Configuration for LMTL model."""

from dataclasses import dataclass

import pint


@dataclass
class LMTLParams:
    """Parameters for a Low Mid Trophic Level functional group.

    Units should be provided using pint.Quantity.
    """

    # Vertical migration
    day_layer: int | float
    night_layer: int | float

    # Recruitment age
    tau_r_0: pint.Quantity  # Maximum recruitment age at T_ref [time]
    gamma_tau_r: pint.Quantity  # Thermal sensitivity of recruitment age [1/temperature]

    # Production transfer
    E: float  # Transfer efficiency from primary production [dimensionless]

    # Mortality
    lambda_0: pint.Quantity  # Mortality rate at T_ref [1/time]
    gamma_lambda: pint.Quantity  # Thermal sensitivity of mortality [1/temperature]

    # Reference temperature
    T_ref: pint.Quantity  # [temperature]
