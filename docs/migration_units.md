# Migration Guide: Unit System Changes

**Date:** 2026-01-26
**Version:** v1.0.0 (Unit Strictness Update)

## Summary of Changes

To ensure numerical stability and avoid silent errors, Seapopym now enforces **strict unit validation** at compilation time. Additionally, a universal time convention has been adopted.

### Key Changes

1. **Compilation Errors**: Unit mismatches now raise `UnitError` (code E205) instead of warnings.
2. **Universal Time Seconds**: All time-dependent parameters and tendencies must be expressed in **seconds** (/s).
3. **Pint Integration**: Units are parsed and compared using canonical forms (e.g., `1/d` is NOT equal to `1/s`).

---

## How to Migrate

### 1. Update your Blueprint

If your Blueprint declared rates in days (`1/d`), you must update them to seconds (`1/s`).

**Before:**

```json
"parameters": {
    "growth_rate": {"units": "1/d"}
}
```

**After:**

```json
"parameters": {
    "growth_rate": {"units": "1/s"}
}
```

### 2. Update your Configuration Values

You must manually convert your parameter values to match the new units. Seapopym does **NOT** perform implicit conversion to avoid ambiguity using JAX.

**Before:**

```json
"growth_rate": {"value": 0.05}  # Implicitly 0.05 per day
```

**After:**

```json
"growth_rate": {"value": 5.787e-7}  # 0.05 / 86400 (per second)
```

_Tip: You can perform the division in your Python script: `0.05 / 86400`._

### 3. Usage of 'count' vs 'individuals'

For population counts, prefer the standard unit **`count`** (or `dimensionless`).
The unit `individuals` is not standard in Pint and requires custom registry definition.

**Before:**

```json
"state": {"biomass": {"units": "individuals"}}
```

**After:**

```json
"state": {"biomass": {"units": "count"}}
```

---

## FAQ

**Q: Why do I get `Unit mismatch: '1/d' != '1/s'`?**
A: You are using a function that expects seconds (likely a biological function or the solver itself) but passing a parameter declared as 1/day. Change the declaration to 1/s.

**Q: Can I keep using days?**
A: No. The internal Euler solver multiplies tendencies by `dt` in seconds. Providing tendencies in per-day would lead to results 86400x too large. This change ensures safety.
