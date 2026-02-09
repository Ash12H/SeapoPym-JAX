"""Common type aliases used across the seapopym package."""

from __future__ import annotations

from typing import Any

Array = Any  # np.ndarray | jax.Array
State = dict[str, Array]
Params = dict[str, Array]
Forcings = dict[str, Array]
Outputs = dict[str, Array]
