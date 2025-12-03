# Implementation Plan - Explicit State Variables in Blueprint

This plan outlines the introduction of explicit state variable declaration within functional groups in the Blueprint. This resolves naming conflicts and dependency cycles for variables that persist between timesteps (e.g., biomass, production).

## Architectural Principle

**Blueprint = Logic Declaration Only**
- The Blueprint declares the model structure (what variables exist, what functions compute what)
- The Blueprint does NOT contain or validate actual data values
- The Controller is responsible for validating that provided initial_state matches the Blueprint contract

## 1. Modify Blueprint API

### 1.1 Update `register_group` signature
- **File**: `seapopym/blueprint/core.py`
- **Method**: `register_group`
- **Change**: Add `state_variables` argument.
    ```python
    def register_group(
        self,
        group_prefix: str,
        units: list[dict],
        parameters: dict[str, dict[str, Any]] | None = None,
        state_variables: dict[str, dict[str, Any]] | None = None,  # NEW
    ) -> None:
        """
        Args:
            state_variables: State variables maintained by this group.
                Format: {var_name: {"dims": (...), "units": "..."}}
                Example: {"biomass": {"dims": (Coordinates.T, Y, X), "units": "gC/m²"}}
                Note: No initial_value here - that's provided via Controller.setup()
        """
    ```

### 1.2 Implementation logic
- Iterate over `state_variables`
- For each state variable:
    - Prefix name with `group_prefix` (e.g., `Micronekton/biomass`)
    - Create a `DataNode` marked as state variable (add `is_state=True` flag)
    - Register in graph and `_data_nodes` registry
    - Store in `registered_variables` set

### 1.3 Add DataNode flag
- **File**: `seapopym/blueprint/nodes.py`
- **Change**: Add `is_state: bool = False` field to `DataNode`
- **Purpose**: Distinguish state variables from forcings and parameters for validation

## 2. Add Strict Validation in build()

### 2.1 Validate tendency targets
- **File**: `seapopym/blueprint/core.py`
- **Method**: `build()`
- **Logic**:
    ```python
    # Collect all state variable names
    declared_states = {name for name, node in self._data_nodes.items() if node.is_state}

    # Collect all tendency targets from registered units
    tendency_targets = set()
    for node in self.graph.nodes():
        if isinstance(node, ComputeNode) and node.output_tendencies:
            tendency_targets.update(node.output_tendencies.values())

    # Strict validation: every tendency target must be a declared state
    invalid_targets = tendency_targets - declared_states
    if invalid_targets:
        raise ConfigurationError(
            f"The following variables are tendency targets but not declared as state variables: {invalid_targets}. "
            f"Declare them in register_group(..., state_variables={{...}})"
        )
    ```

### 2.2 Purpose
- **Strict enforcement**: If a function produces a tendency for a variable, that variable MUST be declared as state
- **Clear error messages**: Guides users to add missing state declarations
- **No implicit behavior**: Forces explicit declaration of all state variables

## 3. Update Graph Construction

### 3.1 Treat state variables as roots
- **File**: `seapopym/blueprint/core.py` (possibly `execution.py`)
- **Logic**:
    - When building dependency graph, state variables are treated as "available at start of timestep" (like forcings)
    - This breaks dependency cycles: `compute_mortality(biomass)` can use `biomass` as input even though it produces a tendency for `biomass`
    - State variables become roots in the DAG for each timestep
    - The Time Integrator closes the loop by applying tendencies to update state for next timestep

### 3.2 Add helper method
- **File**: `seapopym/blueprint/core.py`
- **Method**: `get_state_variables() -> set[str]`
    ```python
    def get_state_variables(self) -> set[str]:
        """Return the set of all declared state variable names."""
        return {name for name, node in self._data_nodes.items() if node.is_state}
    ```

## 4. Update Controller Validation

### 4.1 Validate initial_state coverage
- **File**: `seapopym/controller/core.py`
- **Method**: `setup()`
- **Logic**:
    ```python
    # After building execution plan
    declared_states = self.blueprint.get_state_variables()
    provided_states = set(initial_state.data_vars)

    # Check coverage
    missing_states = declared_states - provided_states
    if missing_states:
        raise StateValidationError(
            f"Initial state is missing required state variables: {missing_states}"
        )

    # Optionally warn about extra variables (not required, might be diagnostics)
    extra_states = provided_states - declared_states - declared_forcings - declared_params
    if extra_states:
        logger.warning(f"Initial state contains undeclared variables: {extra_states}")
    ```

### 4.2 Purpose
- Controller validates that user-provided `initial_state` matches Blueprint contract
- Clear error if state variables are missing
- This is where data/logic separation is enforced

## 5. Update Demo Notebook

### 5.1 Remove old forcing declarations
- **File**: `notebook/05_LMTL_Model_Demo.ipynb`
- **Remove**:
    ```python
    bp.register_forcing("Micronekton/production")  # DELETE
    bp.register_forcing("Micronekton/biomass")      # DELETE
    ```

### 5.2 Update register_group call
- **Add**:
    ```python
    bp.register_group(
        "Micronekton",
        units=[...],  # existing units
        parameters={...},  # existing parameters
        state_variables={  # NEW
            "production": {
                "dims": ("cohort", Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
                "units": "gC/m²/s",
            },
            "biomass": {
                "dims": (Coordinates.T.value, Coordinates.Y.value, Coordinates.X.value),
                "units": "gC/m²",
            },
        }
    )
    ```

### 5.3 Ensure initial_state keys match
- Verify that `initial_state` Dataset contains variables with prefixed names:
    - `Micronekton/production`
    - `Micronekton/biomass`

## 6. Migration Notes (Breaking Change)

**This is a breaking change** - no backward compatibility:
- Old code using `bp.register_forcing("group/state_var")` for state variables will need to migrate
- Users must update to use `state_variables` parameter in `register_group`
- Clear migration path: Move state variable declarations from `register_forcing` to `state_variables`

## 7. Verification

### 7.1 Unit tests
- **Test**: `tests/test_blueprint_state.py` (New test file)
- **Scenarios**:
    1. **Basic registration**: Register state variables, verify they appear in graph with correct prefixes
    2. **Root node behavior**: Verify state variables are treated as roots (no incoming edges from compute nodes)
    3. **Tendency validation - success**: Function with `output_tendencies` pointing to declared state → OK
    4. **Tendency validation - error**: Function with `output_tendencies` pointing to undeclared variable → ConfigurationError
    5. **Controller validation - success**: Provide complete initial_state → OK
    6. **Controller validation - error**: Missing state variable in initial_state → StateValidationError
    7. **Units propagation**: Verify state variable units are stored in DataNode

### 7.2 Integration test
- **Test**: Use LMTL model with new syntax
- **Verify**: Simulation runs successfully with state variables declared explicitly
