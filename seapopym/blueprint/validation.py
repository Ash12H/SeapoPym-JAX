"""Validation pipeline for Blueprint and Config.

This module implements the validation pipeline:
1. Parse syntax (Blueprint/Config)
2. Resolve functions (registry lookup)
3. Check signatures
4. Validate dimensions
5. Validate units (Pint)
6. Build compute/data nodes (no graph needed — process order is authoritative)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .exceptions import (
    DimensionMismatchError,
    FunctionNotFoundError,
    MissingDataError,
    OutputCountMismatchError,
    SignatureMismatchError,
    ValidationError,
)
from .nodes import ComputeNode, DataNode
from .registry import FunctionMetadata, get_function
from .schema import Blueprint, Config


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

    def __init__(self, backend: str = "jax") -> None:
        """Initialize the validator.

        Args:
            backend: Target backend for function resolution.
        """
        self.backend = backend

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

        # Step 4: Validate dimensions
        self._validate_dimensions(blueprint, result)

        # Step 5: Validate units
        self._validate_units(blueprint, result)

        # Step 6: Build compute and data nodes
        if result.valid:
            self._build_nodes(blueprint, result)

        # Step 7: Validate tendencies
        if result.valid:
            self._validate_tendencies(blueprint, result)

        return result

    def _resolve_functions(self, blueprint: Blueprint, result: ValidationResult) -> None:
        """Step 2: Resolve all functions from the registry."""
        for step in blueprint.process:
            try:
                metadata = get_function(step.func, self.backend)
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
            metadata.get_signature()
            required_params = metadata.get_required_inputs()

            # Check that all required parameters are provided
            provided_inputs = set(step.inputs.keys())
            missing = [p for p in required_params if p not in provided_inputs]

            if missing:
                result.add_error(SignatureMismatchError(step.func, missing=missing))

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

    def _validate_dimensions(self, blueprint: Blueprint, result: ValidationResult) -> None:
        """Step 4: Validate dimension compatibility."""
        all_vars = blueprint.declarations.get_all_variables()

        for step in blueprint.process:
            if step.func not in result.resolved_functions:
                continue

            metadata = result.resolved_functions[step.func]

            # Check core_dims match declared dims
            for arg_name, core_dims in metadata.core_dims.items():
                if arg_name not in step.inputs:
                    continue

                var_path = step.inputs[arg_name]
                if var_path not in all_vars:
                    continue

                var_decl = all_vars[var_path]
                if var_decl.dims is None:
                    result.add_warning(
                        f"Variable '{var_path}' has no declared dimensions, "
                        f"but function '{step.func}' expects core_dims {core_dims}"
                    )
                    continue

                # Check that core_dims are subset of declared dims
                for dim in core_dims:
                    if dim not in var_decl.dims:
                        result.add_error(
                            DimensionMismatchError(
                                var_path,
                                expected=core_dims,
                                actual=var_decl.dims,
                            )
                        )

    def _validate_units(self, blueprint: Blueprint, result: ValidationResult) -> None:
        """Step 5: Validate unit compatibility using Pint (strict checking)."""
        from seapopym.compiler.units import UnitValidator

        validator = UnitValidator()
        unit_errors = validator.validate_process_chain(blueprint, result.resolved_functions)

        for error in unit_errors:
            result.add_error(error)  # type: ignore[arg-type]

    def _build_nodes(self, blueprint: Blueprint, result: ValidationResult) -> None:
        """Step 6: Build compute and data nodes from process steps.

        Process order in the Blueprint is authoritative — no graph or
        topological sort needed.
        """
        data_nodes: dict[str, DataNode] = {}

        # Create DataNodes for all declared variables
        all_vars = blueprint.declarations.get_all_variables()
        for var_path, var_decl in all_vars.items():
            is_state = var_path.startswith("state.")
            is_parameter = var_path.startswith("parameters.")

            node = DataNode(
                name=var_path,
                dims=tuple(var_decl.dims) if var_decl.dims else None,
                units=var_decl.units,
                is_state=is_state,
                is_parameter=is_parameter,
            )
            data_nodes[var_path] = node

        compute_nodes: list[ComputeNode] = []

        for step in blueprint.process:
            if step.func not in result.resolved_functions:
                continue

            metadata = result.resolved_functions[step.func]

            # Output mapping: output_key → derived path
            output_mapping = dict(step.outputs)

            # Resolve input dimensions from DataNodes
            from seapopym.compiler.transpose import get_canonical_order

            input_dims: dict[str, tuple[str, ...]] = {}
            for arg_name, var_path in step.inputs.items():
                node_dims = data_nodes.get(var_path, DataNode(name=var_path)).dims
                if node_dims is not None:
                    dims_list = [str(d) for d in node_dims]
                    if var_path.startswith("forcings.") and "T" in dims_list:
                        dims_list.remove("T")
                    input_dims[arg_name] = get_canonical_order(dims_list)
                else:
                    input_dims[arg_name] = ()

            # Derive unique name
            output_targets = list(step.outputs.values())
            targets_suffix = ",".join(t.split(".", 1)[-1] for t in output_targets)
            _unique_name = f"{step.func}[{targets_suffix}]"

            compute_node = ComputeNode(
                func=metadata.func,
                name=_unique_name,
                output_mapping=output_mapping,
                input_mapping=dict(step.inputs),
                core_dims=metadata.core_dims,
                input_dims=input_dims,
                out_dims=metadata.out_dims,
            )
            compute_nodes.append(compute_node)

            # Infer output dimensions
            all_input_dims: set[str] = set()
            for dims in input_dims.values():
                all_input_dims.update(dims)
            all_core_dims: set[str] = set()
            for core in metadata.core_dims.values():
                all_core_dims.update(core)
            broadcast_dims_set = all_input_dims - all_core_dims
            inferred_broadcast_dims = get_canonical_order(list(broadcast_dims_set))
            if metadata.out_dims:
                inferred_output_dims = inferred_broadcast_dims + tuple(metadata.out_dims)
            else:
                inferred_output_dims = inferred_broadcast_dims

            # Create output DataNodes
            for target in step.outputs.values():
                out_node = DataNode(
                    name=target,
                    dims=inferred_output_dims if inferred_output_dims else None,
                )
                data_nodes[target] = out_node

        result.compute_nodes = compute_nodes
        result.data_nodes = data_nodes

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


def validate_blueprint(blueprint: Blueprint, backend: str = "jax") -> ValidationResult:
    """Validate a Blueprint.

    Convenience function that creates a validator and runs validation.

    Args:
        blueprint: The Blueprint to validate.
        backend: Target backend for function resolution.

    Returns:
        ValidationResult with errors, warnings, and built nodes.
    """
    validator = BlueprintValidator(backend=backend)
    return validator.validate(blueprint)


def validate_config(
    config: Config,
    blueprint: Blueprint,
    backend: str = "jax",  # noqa: ARG001
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
        backend: Target backend.

    Returns:
        ValidationResult with errors and warnings.
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

    return result
