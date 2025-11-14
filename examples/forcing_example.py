"""Forcing System Example: Temperature and Recruitment.

This example demonstrates how to use the ForcingManager to:
1. Load base forcings from synthetic datasets
2. Create derived forcings with @derived_forcing
3. Use forcings in a simple simulation

The example simulates:
- Temperature forcing (time, depth, lat, lon)
- Primary production forcing (time, lat, lon)
- Derived recruitment = f(primary_production, transfer_coefficient)
"""

import jax.numpy as jnp
import numpy as np
import xarray as xr

from seapopym_message.forcing import ForcingConfig, ForcingManager, derived_forcing


def create_synthetic_datasets():
    """Create synthetic forcing datasets for demonstration."""
    print("Creating synthetic forcing datasets...")

    # Time coordinates: 0 to 24 hours (hourly)
    times = np.arange(0, 25 * 3600, 3600, dtype=float)  # 0, 1h, 2h, ..., 24h

    # Spatial coordinates
    lats = np.linspace(-10, 10, 40)
    lons = np.linspace(-10, 10, 40)

    # Depth layers (for temperature)
    depths = np.array([0, 150, 400], dtype=float)  # Surface, mid, deep

    # === Temperature dataset (4D: time, depth, lat, lon) ===
    # Create gradient: warmer at surface, cooler at depth
    # Temporal variation: warmer during day (t=12h), cooler at night
    temp_data = np.zeros((len(times), len(depths), len(lats), len(lons)))

    for t_idx, time in enumerate(times):
        hour = time / 3600
        # Daily cycle: +5°C at noon, -5°C at midnight
        daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)

        for d_idx, depth in enumerate(depths):
            # Depth gradient: 25°C at surface, 5°C at 400m
            base_temp = 25 - (depth / 400) * 20

            # Spatial gradient: warmer near equator
            for lat_idx, lat in enumerate(lats):
                lat_factor = 1 - abs(lat) / 10 * 0.3
                temp_data[t_idx, d_idx, lat_idx, :] = (base_temp + daily_variation) * lat_factor

    temperature_ds = xr.Dataset(
        {
            "temperature": (["time", "depth", "lat", "lon"], temp_data),
        },
        coords={
            "time": times,
            "depth": depths,
            "lat": lats,
            "lon": lons,
        },
        attrs={"units": "°C", "description": "Sea surface temperature"},
    )

    # === Primary Production dataset (3D: time, lat, lon) ===
    # Higher near equator, daily cycle
    pp_data = np.zeros((len(times), len(lats), len(lons)))

    for t_idx, time in enumerate(times):
        hour = time / 3600
        # Daily cycle: peak at noon
        daily_cycle = 1 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 24)

        for lat_idx, lat in enumerate(lats):
            # Equatorial enhancement
            lat_factor = 1 + 0.5 * np.exp(-((lat / 5) ** 2))
            pp_data[t_idx, lat_idx, :] = 50 * lat_factor * daily_cycle

    primary_production_ds = xr.Dataset(
        {
            "primary_production": (["time", "lat", "lon"], pp_data),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
        attrs={"units": "mg C/m³/day", "description": "Primary production"},
    )

    print(f"  Temperature: {temperature_ds['temperature'].shape}")
    print(f"  Primary Production: {primary_production_ds['primary_production'].shape}")

    return temperature_ds, primary_production_ds


def main():
    """Run forcing system example."""
    print("=== Forcing System Example ===\n")

    # Step 1: Create synthetic datasets
    temperature_ds, pp_ds = create_synthetic_datasets()

    # Step 2: Configure ForcingManager
    print("\nConfiguring ForcingManager...")

    config = {
        "temperature": ForcingConfig(
            source=temperature_ds,
            dims=["time", "depth", "lat", "lon"],
            units="°C",
            interpolation_method="linear",
        ),
        "primary_production": ForcingConfig(
            source=pp_ds,
            dims=["time", "lat", "lon"],
            units="mg C/m³/day",
            interpolation_method="linear",
        ),
    }

    manager = ForcingManager(config)
    print(f"  {manager}")

    # Step 3: Define derived forcings
    print("\nDefining derived forcings...")

    @derived_forcing(
        name="recruitment",
        inputs=["primary_production"],
        params=["transfer_coefficient", "day_of_year"],
    )
    def compute_recruitment(primary_production, transfer_coefficient, day_of_year):
        """Compute recruitment from primary production with seasonal modulation.

        Args:
            primary_production: Surface PP (lat, lon)
            transfer_coefficient: Trophic transfer efficiency
            day_of_year: Current day (for seasonality)

        Returns:
            Recruitment rate (lat, lon)
        """
        # Seasonal modulation (stronger recruitment in spring/summer)
        seasonal_factor = 1.0 + 0.3 * jnp.sin(2 * jnp.pi * day_of_year / 365)

        # Trophic transfer
        recruitment = primary_production * transfer_coefficient * seasonal_factor

        return recruitment

    manager.register_derived(compute_recruitment)
    print(f"  Registered: {compute_recruitment}")

    # Step 4: Prepare forcings at different times
    print("\nPreparing forcings at different simulation times...")

    times_to_test = [0.0, 6 * 3600, 12 * 3600, 18 * 3600]  # 0h, 6h, 12h, 18h
    params = {"transfer_coefficient": 0.15, "day_of_year": 180}  # Summer

    for time in times_to_test:
        forcings = manager.prepare_timestep(time=time, params=params)

        print(f"\n  Time = {time / 3600:.0f} hours:")
        print(f"    Temperature shape: {forcings['temperature'].shape}")
        print(
            f"    Temperature (surface, center): {float(forcings['temperature'][0, 20, 20]):.2f} °C"
        )
        print(f"    Primary Production shape: {forcings['primary_production'].shape}")
        print(f"    PP (center): {float(forcings['primary_production'][20, 20]):.2f} mg C/m³/day")
        print(f"    Recruitment shape: {forcings['recruitment'].shape}")
        print(f"    Recruitment (center): {float(forcings['recruitment'][20, 20]):.2f} (derived)")

        # Verify recruitment calculation
        # seasonal_factor = 1 + 0.3 * sin(2*pi*180/365) ≈ 1.0128
        seasonal_factor = 1.0 + 0.3 * float(jnp.sin(2 * jnp.pi * params["day_of_year"] / 365))
        expected_recruitment = (
            forcings["primary_production"] * params["transfer_coefficient"] * seasonal_factor
        )
        assert jnp.allclose(forcings["recruitment"], expected_recruitment, rtol=1e-5)

    # Step 5: Demonstrate temporal interpolation
    print("\n\nTemporal interpolation demo:")
    print("  Comparing values at exact hour vs interpolated half-hour")

    # At exact hour (t=12h)
    forcings_12h = manager.prepare_timestep(time=12 * 3600, params=params)
    temp_12h = forcings_12h["temperature"][0, 20, 20]

    # At interpolated time (t=12.5h)
    forcings_12_5h = manager.prepare_timestep(time=12.5 * 3600, params=params)
    temp_12_5h = forcings_12_5h["temperature"][0, 20, 20]

    # At next hour (t=13h)
    forcings_13h = manager.prepare_timestep(time=13 * 3600, params=params)
    temp_13h = forcings_13h["temperature"][0, 20, 20]

    print(f"    T(12h) = {float(temp_12h):.3f} °C")
    print(f"    T(12.5h) = {float(temp_12_5h):.3f} °C (interpolated)")
    print(f"    T(13h) = {float(temp_13h):.3f} °C")
    print(
        f"    Interpolation check: T(12.5h) ≈ (T(12h) + T(13h))/2 = "
        f"{float((temp_12h + temp_13h) / 2):.3f} °C"
    )

    # Step 6: Statistics across space
    print("\n\nSpatial statistics at t=12h:")
    forcings_noon = manager.prepare_timestep(time=12 * 3600, params=params)

    for var_name in ["temperature", "primary_production", "recruitment"]:
        data = forcings_noon[var_name]
        if data.ndim == 3:  # (depth, lat, lon) - use surface
            data = data[0]

        print(f"  {var_name}:")
        print(f"    Mean: {float(jnp.mean(data)):.2f}")
        print(f"    Min: {float(jnp.min(data)):.2f}")
        print(f"    Max: {float(jnp.max(data)):.2f}")
        print(f"    Std: {float(jnp.std(data)):.2f}")

    print("\n=== Example completed successfully! ===")
    print("\nKey takeaways:")
    print("  1. ForcingManager handles multi-dimensional forcings (time + space)")
    print("  2. Temporal interpolation is automatic (via xarray)")
    print("  3. Derived forcings (@derived_forcing) enable custom computations")
    print("  4. Dependency resolution ensures correct computation order")
    print("  5. Ready for distributed simulations (Ray object store)")


if __name__ == "__main__":
    main()
