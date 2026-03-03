"""Common type aliases used across the seapopym package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Array = Any  # np.ndarray | jax.Array
State = dict[str, Array]
Params = dict[str, Array]
Forcings = dict[str, Array]
Outputs = dict[str, Array]

# Callable contracts for optimization
LossFn = Callable[[Params], Array]  # params -> scalar to minimize
PredictFn = Callable[[Params], Array]  # params -> prediction array
