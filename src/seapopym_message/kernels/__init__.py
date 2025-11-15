"""Pre-defined computational kernels (Units): biology kernels.

Transport is now handled by the centralized TransportWorker.
See: seapopym_message.transport.TransportWorker
"""

from seapopym_message.kernels.biology import (
    compute_growth,
    compute_mortality,
    compute_recruitment,
    compute_recruitment_2d,
)

__all__ = [
    # Biology
    "compute_recruitment",
    "compute_recruitment_2d",
    "compute_mortality",
    "compute_growth",
]
