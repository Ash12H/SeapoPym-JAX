"""Pydantic schemas for Blueprint and Config.

This module defines the declarative data structures for model definition (Blueprint)
and experiment configuration (Config).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

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
    """

    model_config = ConfigDict(extra="forbid")

    value: float | int | list[float] | list[int]


# === Process Definitions ===


class TendencySource(BaseModel):
    """A single source contributing to a tendency for a state variable.

    Attributes:
        source: Path to a derived variable (e.g., "derived.mortality_flux").
        sign: Sign multiplier (default +1.0). Use -1.0 to invert the contribution.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    sign: float = 1.0


class ProcessStep(BaseModel):
    """A single process step in the computation graph.

    Attributes:
        func: Function identifier in format "namespace:function_name".
        inputs: Mapping from function argument names to variable paths.
        outputs: Mapping from output keys to derived variable paths.
    """

    model_config = ConfigDict(extra="forbid")

    func: str
    inputs: dict[str, str]
    outputs: dict[str, str]

    @field_validator("func")
    @classmethod
    def validate_func_format(cls, v: str) -> str:
        """Ensure function name follows namespace:name format."""
        if ":" not in v:
            raise ValueError(f"Function name must be in format 'namespace:name', got '{v}'")
        return v

    @field_validator("outputs")
    @classmethod
    def validate_output_targets(cls, v: dict[str, str]) -> dict[str, str]:
        """Ensure all output targets start with 'derived.'."""
        for key, target in v.items():
            if not target.startswith("derived."):
                raise ValueError(
                    f"Output target '{target}' (key '{key}') must start with 'derived.'. "
                    f"All process outputs are intermediate derived values."
                )
        return v


# === Declarations Container ===


class Declarations(BaseModel):
    """Container for all variable declarations.

    Example:
        state:
          biomass:
            units: "g"
            dims: ["Y", "X"]
    """

    model_config = ConfigDict(extra="forbid")

    state: dict[str, VariableDeclaration] = Field(default_factory=dict)
    parameters: dict[str, VariableDeclaration] = Field(default_factory=dict)
    forcings: dict[str, VariableDeclaration] = Field(default_factory=dict)
    derived: dict[str, VariableDeclaration] = Field(default_factory=dict)

    @field_validator("state", "parameters", "forcings", "derived", mode="before")
    @classmethod
    def coerce_declarations(cls, v: Any) -> dict[str, Any]:
        """Coerce raw dicts to VariableDeclaration-compatible dicts."""
        if not isinstance(v, dict):
            return v
        result = {}
        for key, value in v.items():
            if isinstance(value, VariableDeclaration | dict):
                result[key] = value  # Pydantic will validate as VariableDeclaration
            else:
                raise ValueError(f"Invalid declaration for '{key}': expected dict, got {type(value)}")
        return result

    def get_all_variables(self) -> dict[str, VariableDeclaration]:
        """Flatten all declarations into a single dict with dotted paths.

        Returns:
            Dict mapping "category.var" to VariableDeclaration.
        """
        result: dict[str, VariableDeclaration] = {}
        for category_name, category in [
            ("state", self.state),
            ("parameters", self.parameters),
            ("forcings", self.forcings),
            ("derived", self.derived),
        ]:
            for key, var_decl in category.items():
                result[f"{category_name}.{key}"] = var_decl
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
        tendencies: Mapping from state variable names to lists of tendency sources.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    declarations: Declarations
    process: list[ProcessStep]
    tendencies: dict[str, list[TendencySource]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tendencies(self) -> Self:
        """Validate that tendency state keys exist and sources reference derived paths."""
        all_vars = self.declarations.get_all_variables()
        state_keys = {k.removeprefix("state.") for k in all_vars if k.startswith("state.")}

        for state_var, sources in self.tendencies.items():
            if state_var not in state_keys:
                raise ValueError(
                    f"Tendency target '{state_var}' is not a declared state variable. "
                    f"Available: {sorted(state_keys)}"
                )
            for src in sources:
                if not src.source.startswith("derived."):
                    raise ValueError(
                        f"Tendency source '{src.source}' for '{state_var}' must start with 'derived.'."
                    )
        return self

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

        from .loaders import load_file

        data = load_file(source)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Blueprint:
        """Create Blueprint from a dictionary."""
        return cls.model_validate(data)

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
        time_start: Simulation start time (ISO format, e.g., "2000-01-01").
        time_end: Simulation end time (ISO format, e.g., "2020-12-31").
        dt: Timestep duration (e.g., "1d", "0.05d", "6h", "30min").
        forcing_interpolation: Method for temporal interpolation of forcings.
            - "constant": Broadcast static forcings (default).
            - "nearest": Nearest neighbor for under-sampled forcings.
            - "linear": Linear interpolation between points.
            - "ffill": Forward fill (repeat last known value).

        output_path: Optional path for output files (Zarr format).
            - None: Outputs returned in memory.
            - str/Path: Write outputs to disk asynchronously.
    """

    model_config = ConfigDict(extra="forbid")

    time_start: str
    time_end: str
    dt: str = "1d"
    forcing_interpolation: Literal["constant", "nearest", "linear", "ffill"] = "constant"
    output_path: str | None = None

    @field_validator("dt")
    @classmethod
    def validate_dt(cls, v: str) -> str:
        """Validate timestep format using pint (e.g. '1d', '6h', '30min', '0.05d')."""
        import pint

        _ureg = pint.UnitRegistry()
        try:
            quantity = _ureg(v)
            quantity.to("seconds")
        except (pint.UndefinedUnitError, pint.DimensionalityError, Exception):
            raise ValueError(
                f"Invalid dt format: '{v}'. Expected a time duration "
                f"(e.g. '1d', '6h', '30min', '1.5h')."
            )
        return v

    @field_validator("time_start", "time_end")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        """Validate that datetime strings are parseable.

        Args:
            v: Datetime string to validate.

        Returns:
            The validated datetime string.

        Raises:
            ValueError: If the datetime string cannot be parsed.
        """
        import pandas as pd

        try:
            pd.to_datetime(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid datetime format: {v}. Use ISO format (e.g., '2000-01-01').") from e


    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        """Validate that time_end is after time_start.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If time_end <= time_start.
        """
        import pandas as pd

        start = pd.to_datetime(self.time_start)
        end = pd.to_datetime(self.time_end)

        if end <= start:
            raise ValueError(f"time_end ({self.time_end}) must be after time_start ({self.time_start})")

        return self


# === Config ===


class Config(BaseModel):
    """Experiment configuration (concrete values).

    A Config provides the actual data for a simulation run:
    parameter values, forcing data paths/arrays, initial state, etc.

    Attributes:
        parameters: Parameter values (flat, matching Blueprint declarations).
        forcings: Forcing data (paths or arrays).
        initial_state: Initial state data (paths or arrays).
        execution: Execution parameters (timestep, time range, etc.).
        dimension_mapping: Optional mapping from data dimension names to canonical names.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    parameters: dict[str, ParameterValue] = Field(default_factory=dict)
    forcings: dict[str, Any] = Field(default_factory=dict)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    execution: ExecutionParams  # REQUIRED: time_start, time_end are mandatory
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

        from .loaders import load_file

        data = load_file(source)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create Config from a dictionary."""
        return cls.model_validate(data)

    def get_parameter_value(self, path: str) -> ParameterValue | None:
        """Get a parameter value by name.

        Args:
            path: Parameter name (e.g. "growth_rate").

        Returns:
            ParameterValue if found, None otherwise.
        """
        return self.parameters.get(path)
