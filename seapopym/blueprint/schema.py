"""Pydantic schemas for Blueprint and Config.

This module defines the declarative data structures for model definition (Blueprint)
and experiment configuration (Config) as per SPEC_01.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# === Variable Declarations ===


class VariableDeclaration(BaseModel):
    """Declaration of a variable (state, parameter, forcing, or derived).

    Attributes:
        units: Physical units (Pint-compatible string). Example: "g", "1/d", "degC".
        dims: Dimension names. Example: ["Y", "X", "C"].
        description: Human-readable description.
    """

    model_config = ConfigDict(extra="forbid")

    units: str | None = None
    dims: list[str] | None = None
    description: str | None = None


class ParameterValue(BaseModel):
    """Parameter value specification for Config.

    Attributes:
        value: The parameter value (scalar or array).
        trainable: Whether this parameter can be optimized (Phase 5).
        bounds: Optional bounds for optimization [min, max].
    """

    model_config = ConfigDict(extra="forbid")

    value: float | int | list[float] | list[int]
    trainable: bool = False
    bounds: tuple[float, float] | None = None

    @field_validator("bounds", mode="before")
    @classmethod
    def validate_bounds(cls, v: Any) -> tuple[float, float] | None:
        """Convert list to tuple for bounds."""
        if v is None:
            return None
        if isinstance(v, list | tuple) and len(v) == 2:
            return (float(v[0]), float(v[1]))
        raise ValueError("bounds must be a list/tuple of two numbers [min, max]")


# === Process Definitions ===


class ProcessOutput(BaseModel):
    """Output specification for a process step.

    Attributes:
        target: Target variable path (e.g., "tendencies.growth", "state.biomass").
        type: Output type determining how the value is used.
    """

    model_config = ConfigDict(extra="forbid")

    target: str
    type: Literal["tendency", "derived", "diagnostic", "state"]


class ProcessStep(BaseModel):
    """A single process step in the computation graph.

    Attributes:
        func: Function identifier in format "namespace:function_name".
        inputs: Mapping from function argument names to variable paths.
        outputs: Mapping from output keys to ProcessOutput specifications.
    """

    model_config = ConfigDict(extra="forbid")

    func: str
    inputs: dict[str, str]
    outputs: dict[str, ProcessOutput]

    @field_validator("func")
    @classmethod
    def validate_func_format(cls, v: str) -> str:
        """Ensure function name follows namespace:name format."""
        if ":" not in v:
            raise ValueError(f"Function name must be in format 'namespace:name', got '{v}'")
        return v


# === Declarations Container ===


# Type for nested variable declarations (supports functional groups)
NestedVariableDeclaration = dict[str, VariableDeclaration | dict[str, VariableDeclaration]]


class Declarations(BaseModel):
    """Container for all variable declarations.

    Supports both flat and hierarchical (functional group) structures.

    Example (flat):
        state:
          biomass:
            units: "g"
            dims: ["Y", "X"]

    Example (hierarchical):
        state:
          tuna:
            biomass:
              units: "g"
              dims: ["Y", "X", "C"]
    """

    model_config = ConfigDict(extra="forbid")

    state: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    forcings: dict[str, Any] = Field(default_factory=dict)
    derived: dict[str, Any] = Field(default_factory=dict)

    def get_all_variables(self) -> dict[str, VariableDeclaration]:
        """Flatten all declarations into a single dict with dotted paths.

        Returns:
            Dict mapping "category.group.var" or "category.var" to VariableDeclaration.
        """
        result: dict[str, VariableDeclaration] = {}

        for category_name, category in [
            ("state", self.state),
            ("parameters", self.parameters),
            ("forcings", self.forcings),
            ("derived", self.derived),
        ]:
            result.update(self._flatten_category(category_name, category))

        return result

    def _flatten_category(self, prefix: str, category: dict[str, Any]) -> dict[str, VariableDeclaration]:
        """Recursively flatten a category dict."""
        result: dict[str, VariableDeclaration] = {}

        for key, value in category.items():
            full_key = f"{prefix}.{key}"

            if isinstance(value, VariableDeclaration):
                result[full_key] = value
            elif isinstance(value, dict):
                # Check if it's a VariableDeclaration dict or a group
                if "units" in value or "dims" in value or "description" in value:
                    # It's a variable declaration as dict
                    result[full_key] = VariableDeclaration(**value)
                else:
                    # It's a functional group - recurse
                    for sub_key, sub_value in value.items():
                        sub_full_key = f"{full_key}.{sub_key}"
                        if isinstance(sub_value, VariableDeclaration):
                            result[sub_full_key] = sub_value
                        elif isinstance(sub_value, dict):
                            result[sub_full_key] = VariableDeclaration(**sub_value)

        return result


# === Blueprint ===


class Blueprint(BaseModel):
    """Model definition (topology and contracts).

    A Blueprint defines WHAT to compute without any concrete values.
    It specifies the variables, their dimensions/units, and the process chain.

    Attributes:
        id: Unique model identifier.
        version: Semantic version string.
        declarations: Variable declarations (state, parameters, forcings, derived).
        process: Ordered list of process steps forming the computation graph.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    declarations: Declarations
    process: list[ProcessStep]

    @classmethod
    def load(cls, source: str | Path | dict[str, Any]) -> Blueprint:
        """Load a Blueprint from file path or dict.

        Args:
            source: Path to YAML/JSON file, or dict.

        Returns:
            Parsed Blueprint instance.
        """
        if isinstance(source, dict):
            return cls.from_dict(source)

        path = Path(source)
        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif path.suffix == ".json":
            return cls.from_json(path)
        else:
            # Try YAML by default
            return cls.from_yaml(path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Blueprint:
        """Create Blueprint from a dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Blueprint:
        """Load Blueprint from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> Blueprint:
        """Load Blueprint from a JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_variable(self, path: str) -> VariableDeclaration | None:
        """Get a variable declaration by its dotted path.

        Args:
            path: Variable path like "state.biomass" or "state.tuna.biomass".

        Returns:
            VariableDeclaration if found, None otherwise.
        """
        all_vars = self.declarations.get_all_variables()
        return all_vars.get(path)


# === Execution Parameters ===


class ExecutionParams(BaseModel):
    """Execution parameters for a simulation run.

    Attributes:
        time_range: Start and end dates as strings.
        dt: Timestep as string (e.g., "1d", "6h").
        output_path: Optional path for output files.
        forcing_interpolation: Method for temporal interpolation of forcings.
            - "constant": Broadcast static forcings (default, existing behavior)
            - "nearest": Nearest neighbor for under-sampled forcings
            - "linear": Linear interpolation between points
            - "ffill": Forward fill (repeat last known value)
    """

    model_config = ConfigDict(extra="forbid")

    time_range: tuple[str, str] | None = None
    dt: str = "1d"
    output_path: str | None = None
    forcing_interpolation: Literal["constant", "nearest", "linear", "ffill"] = "constant"

    @field_validator("time_range", mode="before")
    @classmethod
    def validate_time_range(cls, v: Any) -> tuple[str, str] | None:
        """Convert list to tuple for time_range."""
        if v is None:
            return None
        if isinstance(v, list | tuple) and len(v) == 2:
            return (str(v[0]), str(v[1]))
        raise ValueError("time_range must be a list/tuple of two date strings")


# === Config ===


class Config(BaseModel):
    """Experiment configuration (concrete values).

    A Config provides the actual data for a simulation run:
    parameter values, forcing data paths/arrays, initial state, etc.

    Attributes:
        model: Optional path to the Blueprint file.
        parameters: Parameter values (hierarchical, matching Blueprint).
        forcings: Forcing data (paths or arrays).
        initial_state: Initial state data (paths or arrays).
        execution: Execution parameters (timestep, time range, etc.).
        dimension_mapping: Optional mapping from data dimension names to canonical names.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    forcings: dict[str, Any] = Field(default_factory=dict)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    execution: ExecutionParams = Field(default_factory=ExecutionParams)
    dimension_mapping: dict[str, str] | None = None

    @classmethod
    def load(cls, source: str | Path | dict[str, Any]) -> Config:
        """Load a Config from file path or dict.

        Args:
            source: Path to YAML/JSON file, or dict.

        Returns:
            Parsed Config instance.
        """
        if isinstance(source, dict):
            return cls.from_dict(source)

        path = Path(source)
        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif path.suffix == ".json":
            return cls.from_json(path)
        else:
            return cls.from_yaml(path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create Config from a dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load Config from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> Config:
        """Load Config from a JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_parameter_value(self, path: str) -> Any:
        """Get a parameter value by dotted path.

        Args:
            path: Parameter path like "growth_rate" or "tuna.growth_rate".

        Returns:
            Parameter value or ParameterValue object.
        """
        parts = path.split(".")
        current = self.parameters

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current
