temperature_forcing = Forcing(...)
group1 = FunctionalGroup(
    name="group1",
    param={
        "growth_rate": 0.1,
        "mortality_rate": 0.05,
    },
    kernel=Kernel(
        GrowthUnitInstance,
        MortalityUnit,
    ),
    derived_forcing={
        "temperature": Forcing("temperature").apply(avg_dvm, kwargs={"day_layer":1, "night_layer":0}),
        "salinity": SelectForcing(name="salinity", dim="Z", value=1),
    }
)
