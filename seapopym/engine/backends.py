"""Backend implementations for time-stepping loops.

Provides two backends:
- JAXBackend: Uses jax.lax.scan for compiled execution
- NumpyBackend: Uses Python for loop for debugging
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol

import numpy as np

from .exceptions import BackendError


class Backend(Protocol):
    """Protocol for backend implementations."""

    def scan(
        self,
        step_fn: Callable[[Any, Any], tuple[Any, Any]],
        init: Any,
        xs: Any,
        length: int | None = None,
    ) -> tuple[Any, Any]:
        """Execute a fold operation over the time dimension.

        This performs the scan pattern: repeatedly applying step_fn
        to accumulate results over the time axis of xs.

        Args:
            step_fn: Function (carry, x) -> (new_carry, output).
            init: Initial carry value.
            xs: Input sequence to scan over (leading dim is time).
            length: Optional length if xs is None.

        Returns:
            Tuple of (final_carry, stacked_outputs).
        """
        ...


class JAXBackend:
    """Backend using jax.lax.scan for compiled execution."""

    def __init__(self) -> None:
        """Initialize JAX backend."""
        try:
            import jax  # noqa: F401
        except ImportError as e:
            raise BackendError("jax", "JAX is not installed") from e
        self._cached_scan_fn: Callable[..., Any] | None = None

    def scan(
        self,
        step_fn: Callable[[Any, Any], tuple[Any, Any]],
        init: Any,
        xs: Any,
        length: int | None = None,
    ) -> tuple[Any, Any]:
        """Execute scan using jax.lax.scan with JIT compilation.

        The scan is JIT-compiled on first call and cached for subsequent calls.
        This provides significant speedup when processing multiple chunks.

        Args:
            step_fn: Function (carry, x) -> (new_carry, output).
            init: Initial carry value (state dict).
            xs: Input sequence (forcings dict with time as leading dim).
            length: Optional length if xs is None.

        Returns:
            Tuple of (final_carry, stacked_outputs).
        """
        import jax
        import jax.lax as lax

        # Create JIT-compiled scan function if not cached
        # Note: We can't cache based on step_fn identity easily, so we recreate
        # if step_fn changes. In practice, step_fn is the same for all chunks.
        @jax.jit
        def jitted_scan(init_state: Any, inputs: Any) -> tuple[Any, Any]:
            return lax.scan(step_fn, init_state, inputs, length=length)

        return jitted_scan(init, xs)


class NumpyBackend:
    """Backend using Python for loop for debugging."""

    def scan(
        self,
        step_fn: Callable[[Any, Any], tuple[Any, Any]],
        init: Any,
        xs: Any,
        length: int | None = None,
    ) -> tuple[Any, Any]:
        """Execute scan using Python for loop.

        Args:
            step_fn: Function (carry, x) -> (new_carry, output).
            init: Initial carry value (state dict).
            xs: Input sequence (forcings dict with time as leading dim).
            length: Optional length if xs is None.

        Returns:
            Tuple of (final_carry, stacked_outputs).
        """
        carry = init

        # Determine length from xs
        if xs is None:
            if length is None:
                raise BackendError("numpy", "Either xs or length must be provided")
            n_steps = length
        elif isinstance(xs, dict):
            # Get length from first array in dict
            first_key = next(iter(xs))
            n_steps = xs[first_key].shape[0]
        else:
            n_steps = xs.shape[0]

        outputs_list: list[Any] = []

        for t in range(n_steps):
            # Slice inputs at time t
            if xs is None:
                x_t = None
            elif isinstance(xs, dict):
                x_t = {k: v[t] for k, v in xs.items()}
            else:
                x_t = xs[t]

            carry, output = step_fn(carry, x_t)
            outputs_list.append(output)

        # Stack outputs
        stacked_outputs: Any
        if len(outputs_list) == 0:
            stacked_outputs = {}
        elif isinstance(outputs_list[0], dict):
            stacked_outputs = {k: np.stack([o[k] for o in outputs_list], axis=0) for k in outputs_list[0]}
        else:
            stacked_outputs = np.stack(outputs_list, axis=0)

        return carry, stacked_outputs


def get_backend(name: Literal["jax", "numpy"]) -> Backend:
    """Get a backend by name.

    Args:
        name: Backend name ("jax" or "numpy").

    Returns:
        Backend instance.

    Raises:
        BackendError: If backend is not available.
    """
    if name == "jax":
        return JAXBackend()
    elif name == "numpy":
        return NumpyBackend()
    else:
        raise BackendError(name, f"Unknown backend '{name}'. Use 'jax' or 'numpy'.")
