"""Utility functions: domain splitting, visualization, diagnostics."""

from seapopym_message.utils.domain import split_domain_2d, split_domain_2d_periodic_lon
from seapopym_message.utils.grid import GridInfo

__all__ = [
    "GridInfo",
    "split_domain_2d",
    "split_domain_2d_periodic_lon",
]
