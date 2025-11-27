---
description: Implementation plan for Parameters in State and Unit Standardization
---

# Plan: Parameters in State & Unit Standardization

This plan details the steps to refactor `seapopym` to treat parameters as state variables and implement automatic unit standardization using `pint`.

## 1. Dependencies
- [ ] Add `pint` and `pint-xarray` to `pyproject.toml`.

## 2. Blueprint Refactoring (`seapopym/blueprint`)
The Blueprint acts as the contract for the model, defining expected variables and their units.

- [ ] **Modify `DataNode`**: Add `units: str | None = None` field.
- [ ] **Update `register_forcing`**: Add `units` argument to `Blueprint.register_forcing`.
- [ ] **Add `register_parameter`**: Add `Blueprint.register_parameter(name, units, ...)` method.
    - This behaves similarly to `register_forcing` but explicitly marks the variable as a parameter (useful for validation).
- [ ] **Update `register_unit`**: Ensure `input_mapping` can correctly link function arguments to these new parameter nodes.

## 3. Controller Refactoring (`seapopym/controller`)
The Controller enforces the contract defined by the Blueprint during setup.

- [ ] **Update `SimulationController.setup`**:
    - Add `parameters: Any | None = None` argument (accepts Dataclass or Dict).
- [ ] **Implement `_ingest_parameters`**:
    - Convert the input `parameters` object into an `xarray.Dataset`.
    - Handle Dataclasses (using `dataclasses.asdict`) and Dictionaries.
    - Assign default names based on the Blueprint contract (or use a naming convention like `{group}_{param}`).
- [ ] **Implement `_standardize_units`**:
    - Iterate over the combined state (Initial State + Forcings + Parameters).
    - Check against `Blueprint`'s registered units.
    - Use `pint` and `pint-xarray` to validate and convert units.
    - Raise informative errors if units are incompatible.

## 4. LMTL Model Update (`seapopym/lmtl`)
(To be done in a subsequent phase, but planned here)
- [ ] Update `LMTLParams` to use `pint` types for documentation/typing.
- [ ] Update model registration to use `register_parameter` with units.

## 5. Verification
- [ ] Create a test case with mismatched but compatible units (e.g., days vs seconds) and verify automatic conversion.
- [ ] Create a test case with incompatible units and verify error raising.
