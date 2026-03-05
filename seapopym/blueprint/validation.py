"""Validation pipeline for Blueprint and Config.

Blueprint validation pipeline:
1. Syntax — Pydantic parse (implicit, Blueprint already parsed).
2. Resolve functions — registry lookup.
3. Check signatures — required inputs, output count, variable existence.
4. Build nodes & validate core dimensions — build ComputeNode/DataNode
   list in process order; for each step, verify that every core_dim
   declared by the function exists in the (declared or propagated)
   dimensions of the corresponding input.
5. Validate units — propagate units along the process chain (Pint).
6. Validate tendencies — check sources reference produced derived variables.

Steps 4-6 are independent: a unit error does not block dimension checks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .exceptions import (
    BlueprintValidationError,
    ConfigValidationError,
    DimensionMismatchError,
    FunctionNotFoundError,
    MissingDataError,
    OutputCountMismatchError,
    SignatureMismatchError,
    ValidationError,
)
from .nodes import ComputeNode, DataNode
from .registry import FunctionMetadata, get_function
from .schema import Blueprint, Config, ProcessStep


@dataclass
class ValidationResult:
    """Result of the validation pipeline.

    Attributes:
        valid: Whether validation passed.
        errors: List of validation errors (if any).
        warnings: List of non-fatal warnings.
        compute_nodes: Ordered list of compute nodes (process steps).
        data_nodes: Dict mapping variable paths to DataNode metadata.
        resolved_functions: Map of function names to metadata.
    """

    valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    compute_nodes: list[ComputeNode] = field(default_factory=list)
    data_nodes: dict[str, DataNode] = field(default_factory=dict)
    resolved_functions: dict[str, FunctionMetadata] = field(default_factory=dict)

    def add_error(self, error: ValidationError) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)


class BlueprintValidator:
    """Validator for Blueprint definitions.

    Implements the validation pipeline for validating a Blueprint
    against its declarations and the function registry.
    """

    def validate(self, blueprint: Blueprint) -> ValidationResult:
        """Run the full validation pipeline on a Blueprint.

        Args:
            blueprint: The Blueprint to validate.

        Returns:
            ValidationResult with errors, warnings, and built nodes.
        """
        result = ValidationResult()

        # Step 1: Syntax already validated by Pydantic (Blueprint parsed)

        # Step 2: Resolve functions
        self._resolve_functions(blueprint, result)
        if not result.valid:
            return result

        # Step 3: Check signatures
        self._check_signatures(blueprint, result)
        if not result.valid:
            return result

        # Steps 4-6 are independent — errors in one do not block the others.

        # Step 4: Build nodes and validate core dimensions
        self._build_nodes_and_validate_dims(blueprint, result)

        # Step 5: Validate units
        self._validate_units(blueprint, result)

        # Step 6: Validate tendencies
        self._validate_tendencies(blueprint, result)

        return result

    def _resolve_functions(self, blueprint: Blueprint, result: ValidationResult) -> None:
        """Step 2: Resolve all functions from the registry."""
        for step in blueprint.process:
            try:
                metadata = get_function(step.func)
                result.resolved_functions[step.func] = metadata
            except FunctionNotFoundError as e:
                result.add_error(e)

    def _check_signatures(self, blueprint: Blueprint, result: ValidationResult) -> None:
        """Step 3: Check that function signatures match declared inputs."""
        all_vars = blueprint.declarations.get_all_variables()
        available_vars = set(all_vars.keys())

        # Track produced variables from process outputs
        produced_vars: set[str] = set()

        for step in blueprint.process:
            if step.func not in result.resolved_functions:
                continue  # Already reported as error

            metadata = result.resolved_functions[step.func]
            sig = metadata.get_signature()
            required_params = metadata.get_required_inputs()
            all_params = set(sig.parameters.keys())

            # Check that all required parameters are provided
            provided_inputs = set(step.inputs.keys())
            missing = [p for p in required_params if p not in provided_inputs]
            extra = sorted(provided_inputs - all_params)

            if missing or extra:
                result.add_error(SignatureMismatchError(step.func, missing=missing or None, extra=extra or None))

            # Check that all provided inputs reference valid variables
            for _arg_name, var_path in step.inputs.items():
                if var_path not in available_vars and var_path not in produced_vars:
                    result.add_error(MissingDataError(var_path, data_type=f"input for {step.func}"))

            # Check output count
            declared_outputs = len(step.outputs)
            if metadata.outputs:
                expected_outputs = len(metadata.outputs)
                if declared_outputs != expected_outputs:
                    result.add_error(OutputCountMismatchError(step.func, expected_outputs, declared_outputs))

            # Register produced variables (outputs are now simple strings)
            for target in step.outputs.values():
                produced_vars.add(target)

    def _validate_units(self, blueprint: Blueprint, result: ValidationResult) -> None:
        """Validate unit compatibility using Pint (strict checking)."""
        from .units import UnitValidator

        validator = UnitValidator()
        unit_errors = validator.validate_process_chain(blueprint, result.resolved_functions)

        for error in unit_errors:
            result.add_error(error)

    def _build_nodes_and_validate_dims(self, blueprint: Blueprint, result: ValidationResult) -> None:
        """Build ComputeNode/DataNode lists and validate core dimensions.

        Iterates over process steps in declaration order:
        1. Resolve each input's dimensions (from declarations or from a
           previously produced derived variable).
        2. Validate that every core_dim declared by the function exists
           in the resolved dimensions of the corresponding input.
        3. Propagate output dimensions for derived variables so that
           subsequent steps can validate against them.

        Note: Forcings are declared with a T (time) dimension, but the
        engine slices forcings per-timestep before passing them to
        functions. T is therefore excluded from dimension checks here.
        """
        from seapopym.dims import get_canonical_order

        data_nodes: dict[str, DataNode] = {}

        # Seed data_nodes from Blueprint declarations
        all_vars = blueprint.declarations.get_all_variables()
        for var_path, var_decl in all_vars.items():
            data_nodes[var_path] = DataNode(
                name=var_path,
                dims=tuple(var_decl.dims) if var_decl.dims else None,
                units=var_decl.units,
                is_state=var_path.startswith("state."),
                is_parameter=var_path.startswith("parameters."),
            )

        compute_nodes: list[ComputeNode] = []

        for step in blueprint.process:
            if step.func not in result.resolved_functions:
                continue

            metadata = result.resolved_functions[step.func]

            # --- Resolve input dimensions ---
            input_dims = self._resolve_input_dims(step, data_nodes, get_canonical_order)

            # --- Validate core_dims ---
            self._check_core_dims(step, metadata, input_dims, result)

            # --- Build ComputeNode ---
            targets_suffix = ",".join(t.split(".", 1)[-1] for t in step.outputs.values())
            compute_node = ComputeNode(
                func=metadata.func,
                name=f"{step.func}[{targets_suffix}]",
                output_mapping=dict(step.outputs),
                input_mapping=dict(step.inputs),
                core_dims=metadata.core_dims,
                input_dims=input_dims,
                out_dims=metadata.out_dims,
            )
            compute_nodes.append(compute_node)

            # --- Propagate output dims for downstream steps ---
            output_dims = self._compute_output_dims(input_dims, metadata, get_canonical_order)
            for target in step.outputs.values():
                data_nodes[target] = DataNode(
                    name=target,
                    dims=output_dims if output_dims else None,
                )

        result.compute_nodes = compute_nodes
        result.data_nodes = data_nodes

    @staticmethod
    def _resolve_input_dims(
        step: ProcessStep,
        data_nodes: dict[str, DataNode],
        get_canonical_order: Callable[..., Any],
    ) -> dict[str, tuple[str, ...]]:
        """Resolve the dimension tuple for each input of a process step.

        Forcings have their T dimension stripped (the engine slices per-timestep).
        """
        input_dims: dict[str, tuple[str, ...]] = {}
        for arg_name, var_path in step.inputs.items():
            node_dims = data_nodes.get(var_path, DataNode(name=var_path)).dims
            if node_dims is None:
                input_dims[arg_name] = ()
                continue
            dims_list = [str(d) for d in node_dims]
            # T is consumed by the engine's time loop, not seen by functions
            if var_path.startswith("forcings.") and "T" in dims_list:
                dims_list.remove("T")
            input_dims[arg_name] = get_canonical_order(dims_list)
        return input_dims

    @staticmethod
    def _check_core_dims(
        step: ProcessStep,
        metadata: FunctionMetadata,
        input_dims: dict[str, tuple[str, ...]],
        result: ValidationResult,
    ) -> None:
        """Verify that every core_dim declared by the function exists in the input's dims."""
        for arg_name, core_dims in metadata.core_dims.items():
            if arg_name not in step.inputs:
                continue
            var_path = step.inputs[arg_name]
            resolved = input_dims.get(arg_name, ())
            if not resolved:
                result.add_warning(
                    f"Variable '{var_path}' has no dimensions, "
                    f"but function '{step.func}' expects core_dims {core_dims}"
                )
                continue
            for dim in core_dims:
                if dim not in resolved:
                    result.add_error(
                        DimensionMismatchError(var_path, expected=core_dims, actual=list(resolved))
                    )

    @staticmethod
    def _compute_output_dims(
        input_dims: dict[str, tuple[str, ...]],
        metadata: FunctionMetadata,
        get_canonical_order: Callable[..., Any],
    ) -> tuple[str, ...]:
        """Compute the output dimensions of a process step.

        output_dims = broadcast_dims (all input dims minus core_dims) + out_dims.
        """
        all_dims: set[str] = set()
        for dims in input_dims.values():
            all_dims.update(dims)
        core: set[str] = set()
        for c in metadata.core_dims.values():
            core.update(c)
        broadcast = get_canonical_order(list(all_dims - core))
        if metadata.out_dims:
            return broadcast + tuple(metadata.out_dims)
        return broadcast

    def _validate_tendencies(self, blueprint: Blueprint, result: ValidationResult) -> None:
        """Validate that tendency sources reference produced derived variables."""
        # Collect all produced derived paths
        produced_derived: set[str] = set()
        for step in blueprint.process:
            for target in step.outputs.values():
                produced_derived.add(target)

        for state_var, sources in blueprint.tendencies.items():
            for src in sources:
                if src.source not in produced_derived:
                    result.add_warning(
                        f"Tendency source '{src.source}' for state '{state_var}' "
                        f"is not produced by any process step."
                    )


def validate_blueprint(blueprint: Blueprint) -> ValidationResult:
    """Validate a Blueprint.

    Convenience function that creates a validator and runs validation.

    Args:
        blueprint: The Blueprint to validate.

    Returns:
        ValidationResult with errors, warnings, and built nodes.

    Raises:
        BlueprintValidationError: If validation fails (contains all errors).
    """
    validator = BlueprintValidator()
    result = validator.validate(blueprint)
    if not result.valid:
        raise BlueprintValidationError(result.errors, result.warnings)
    return result


def validate_config(
    config: Config,
    blueprint: Blueprint,
) -> ValidationResult:
    """Validate a Config against a Blueprint.

    This validates that:
    - All required parameters are provided
    - All required forcings are provided
    - All required initial state variables are provided
    - Data dimensions match declarations

    Args:
        config: The Config to validate.
        blueprint: The Blueprint to validate against.

    Returns:
        ValidationResult with errors and warnings.

    Raises:
        ConfigValidationError: If validation fails (contains all errors).
    """
    result = ValidationResult()
    all_vars = blueprint.declarations.get_all_variables()

    # Check required parameters
    for var_path, _var_decl in all_vars.items():
        if not var_path.startswith("parameters."):
            continue

        param_path = var_path.replace("parameters.", "")
        param_value = config.get_parameter_value(param_path)

        if param_value is None:
            result.add_error(MissingDataError(var_path, data_type="parameter"))

    # Check required forcings
    for var_path, _var_decl in all_vars.items():
        if not var_path.startswith("forcings."):
            continue

        forcing_name = var_path.replace("forcings.", "")
        if forcing_name not in config.forcings:
            result.add_error(MissingDataError(var_path, data_type="forcing"))

    # Check required initial state
    for var_path, _var_decl in all_vars.items():
        if not var_path.startswith("state."):
            continue

        state_name = var_path.replace("state.", "")
        parts = state_name.split(".")

        current = config.initial_state
        found = True
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break

        if not found:
            result.add_error(MissingDataError(var_path, data_type="initial_state"))

    # Check dimension consistency between data and blueprint declarations
    _validate_data_dims(config, all_vars, result)

    if not result.valid:
        raise ConfigValidationError(result.errors, result.warnings)
    return result


def _validate_data_dims(
    config: Config,
    all_vars: dict[str, Any],
    result: ValidationResult,
) -> None:
    """Validate that data dimensions match blueprint declarations.

    For forcings and initial_state provided as xr.DataArray, checks that
    the dimension names match the blueprint declarations (ignoring order).
    The config's ``dimension_mapping`` is applied to data dims before
    comparison so that non-canonical names (e.g. ``lat`` → ``Y``) are
    accepted.

    Args:
        config: The Config to validate.
        all_vars: All variable declarations from the blueprint.
        result: ValidationResult to accumulate errors.
    """
    import xarray as xr

    dim_map = config.dimension_mapping or {}

    def _mapped_dims(da: xr.DataArray) -> set[str]:
        return {dim_map.get(str(d), str(d)) for d in da.dims}

    # Validate forcings
    for var_path, var_decl in all_vars.items():
        if not var_path.startswith("forcings."):
            continue
        if var_decl.dims is None:
            continue

        forcing_name = var_path.removeprefix("forcings.")
        source = config.forcings.get(forcing_name)
        if not isinstance(source, xr.DataArray):
            continue

        expected = set(var_decl.dims)
        actual = _mapped_dims(source)
        if expected != actual:
            result.add_error(
                DimensionMismatchError(var_path, expected=sorted(expected), actual=sorted(actual))
            )

    # Validate initial_state
    for var_path, var_decl in all_vars.items():
        if not var_path.startswith("state."):
            continue
        if var_decl.dims is None:
            continue

        state_name = var_path.removeprefix("state.")
        source = config.initial_state.get(state_name)
        if not isinstance(source, xr.DataArray):
            continue

        expected = set(var_decl.dims)
        actual = _mapped_dims(source)
        if expected != actual:
            result.add_error(
                DimensionMismatchError(var_path, expected=sorted(expected), actual=sorted(actual))
            )
