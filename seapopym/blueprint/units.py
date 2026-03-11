"""Unit validation using Pint for strict dimensional consistency checking.

This module provides utilities to validate that units declared in Blueprints
match the expected units in function signatures, and that the entire process
chain maintains unit consistency.

Key principle: Units must be EXACTLY identical (canonical forms), not just
dimensionally compatible. For example:
    - "m/s" == "meter/second"  → OK (same canonical form)
    - "m/s" == "km/s"           → ERROR (different forms)
    - "1/d" == "1/s"            → ERROR (different forms)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pint

if TYPE_CHECKING:
    from seapopym.blueprint import Blueprint
    from seapopym.blueprint.registry import FunctionMetadata

from seapopym.blueprint.exceptions import UnitError


class UnitValidator:
    """Validator for strict unit checking using Pint.

    This validator ensures that all units in a Blueprint are:
    1. Valid Pint unit strings
    2. Exactly matching (canonical form) between declarations and expectations
    3. Consistent across the entire process chain

    Attributes:
        ureg: Pint unit registry for parsing and comparing units.
    """

    def __init__(self, ureg: pint.UnitRegistry | None = None) -> None:
        """Initialize the unit validator.

        Args:
            ureg: Optional custom Pint registry. If None, creates default registry.
        """
        self.ureg = ureg or pint.UnitRegistry()

    def parse_unit(self, unit_str: str | None) -> Any:
        """Parse a unit string to a Pint Unit object.

        Args:
            unit_str: Unit string (e.g., "m/s", "1/d", "dimensionless").

        Returns:
            Parsed Pint Unit object.

        Raises:
            UnitError: If unit string is invalid.
        """
        if unit_str == "dimensionless" or unit_str is None:
            return self.ureg.dimensionless

        try:
            return self.ureg(unit_str).units
        except (pint.UndefinedUnitError, pint.DimensionalityError) as e:
            raise UnitError(f"Invalid unit string '{unit_str}': {e}") from e

    def check_exact_match(
        self,
        unit1: str | None,
        unit2: str | None,
        context: str = "",
    ) -> None:
        """Check if two units are exactly identical (canonical forms).

        This is a STRICT check: units must have the same canonical form,
        not just be dimensionally compatible.

        Examples:
            check_exact_match("m/s", "meter/second", ...)   # OK
            check_exact_match("m * s**-1", "m/s", ...)      # OK
            check_exact_match("m/s", "km/s", ...)           # RAISES UnitError
            check_exact_match("1/d", "1/s", ...)            # RAISES UnitError

        Args:
            unit1: First unit string (can be None for dimensionless).
            unit2: Second unit string (can be None for dimensionless).
            context: Context string for error message (e.g., "function:arg").

        Raises:
            UnitError: If canonical forms differ.
        """
        # Parse units to canonical forms
        u1 = self.parse_unit(unit1 or "dimensionless")
        u2 = self.parse_unit(unit2 or "dimensionless")

        # Compare canonical forms
        if u1 != u2:
            context_str = f" in {context}" if context else ""
            raise UnitError(
                f"Unit mismatch{context_str}: '{unit1}' != '{unit2}'\n"
                f"  Canonical forms: {u1} != {u2}\n"
                f"  Units must be identical (not just dimensionally compatible).\n"
                f"  → If you need time units, use only seconds (/s, not /d or /h)."
            )

    def validate_process_chain(
        self,
        blueprint: Blueprint,
        resolved_functions: dict[str, FunctionMetadata],
    ) -> list[UnitError]:
        """Validate unit consistency across the entire process chain.

        This method checks:
        1. Input units: declared in Blueprint match expected by functions
        2. Output units: returned by functions match target declarations (if declared)
        3. Chain consistency: output of function A → input of function B

        Args:
            blueprint: The Blueprint to validate.
            resolved_functions: Map of function names to their metadata.

        Returns:
            List of UnitError objects (empty if all valid).
        """
        errors: list[UnitError] = []
        all_vars = blueprint.declarations.get_all_variables()

        # Track produced variables and their units (for chain validation)
        produced_vars: dict[str, str] = {}

        for step in blueprint.process:
            if step.func not in resolved_functions:
                continue  # Already reported as function not found

            metadata = resolved_functions[step.func]

            # Validate input units
            for arg_name, var_path in step.inputs.items():
                # Get expected unit from function metadata
                expected_unit = metadata.units.get(arg_name)
                if expected_unit is None or expected_unit == "return":
                    continue  # No unit constraint or it's the return unit

                # Get declared unit from Blueprint (or from previously produced var)
                if var_path in all_vars:
                    declared_unit = all_vars[var_path].units
                elif var_path in produced_vars:
                    declared_unit = produced_vars[var_path]
                else:
                    continue  # Variable not found (reported elsewhere)

                # Check exact match
                try:
                    self.check_exact_match(
                        declared_unit,
                        expected_unit,
                        context=f"{step.func}:{arg_name}",
                    )
                except UnitError as e:
                    errors.append(e)

            # Validate output units
            for out_key, target_path in step.outputs.items():
                # multi-output → use output key name, single-output → use "return"
                out_unit = metadata.units.get(out_key) if metadata.is_multi_output else metadata.units.get("return")

                if out_unit is None:
                    continue  # No unit declared, chain stops here

                # Check against declared vars if target exists in Blueprint
                if target_path in all_vars:
                    target_unit = all_vars[target_path].units
                    if target_unit is not None:
                        try:
                            self.check_exact_match(
                                out_unit,
                                target_unit,
                                context=f"{step.func} output '{out_key}' → {target_path}",
                            )
                        except UnitError as e:
                            errors.append(e)

                # Register this variable for future chain validation
                produced_vars[target_path] = out_unit

        # Validate tendency sources: check that each tendency source has time^-1 dimension
        for state_var, sources in blueprint.tendencies.items():
            for src in sources:
                source_unit = produced_vars.get(src.source)
                if source_unit is None:
                    continue  # Not produced or no unit info

                try:
                    unit_obj = self.parse_unit(source_unit)
                    if "[time]" not in str(unit_obj.dimensionality):
                        errors.append(
                            UnitError(
                                f"Tendency source '{src.source}' for state '{state_var}' has unit '{source_unit}' "
                                f"which lacks a time dimension.\n"
                                f"  Tendencies must have units like 'X/s' (per second) for Euler solver.\n"
                                f"  Example: 'individuals/s', 'g/s', 'kg*m/s', etc."
                            )
                        )
                except UnitError:
                    pass  # Already caught by parse_unit

        return errors


def validate_units(
    blueprint: Blueprint,
    resolved_functions: dict[str, FunctionMetadata],
) -> list[UnitError]:
    """Convenience function for unit validation.

    Args:
        blueprint: The Blueprint to validate.
        resolved_functions: Map of function names to their metadata.

    Returns:
        List of UnitError objects (empty if all valid).
    """
    validator = UnitValidator()
    return validator.validate_process_chain(blueprint, resolved_functions)
