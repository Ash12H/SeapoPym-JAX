"""Ray execution backend (Placeholder)."""

from typing import Any

import xarray as xr

from seapopym.backend.base import ComputeBackend
from seapopym.blueprint.nodes import ComputeNode


class RayBackend(ComputeBackend):
    """Distributed execution backend using Ray."""

    def execute(
        self,
        task_groups: list[tuple[str, list[ComputeNode]]],
        state: xr.Dataset,
    ) -> dict[str, Any]:
        """Execute tasks in parallel using Ray."""
        raise NotImplementedError("Ray backend is not yet implemented.")
