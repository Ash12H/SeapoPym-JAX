import pytest
import xarray as xr

from seapopym.time_integrator import TimeIntegrator


def test_euler_simple():
    """Test Euler simple : x(t+1) = x(t) + dt * dx/dt."""
    # État initial
    state = xr.Dataset({"biomass": (("x"), [10.0])})

    # Tendance : +2 par seconde
    all_results = {"growth_rate": xr.DataArray([2.0], dims=["x"])}

    # Mapping
    tendency_map = {"biomass": ["growth_rate"]}

    # Intégration
    integrator = TimeIntegrator()
    new_state = integrator.integrate(state, all_results, tendency_map, dt=1.0)

    # Vérification : 10 + 1*2 = 12
    assert new_state["biomass"].item() == 12.0


def test_multiple_tendencies():
    """Test avec plusieurs tendances affectant la même variable."""
    state = xr.Dataset({"biomass": (("x"), [100.0])})

    all_results = {
        "growth_rate": xr.DataArray([5.0], dims=["x"]),
        "mortality_rate": xr.DataArray([-3.0], dims=["x"]),
        "transport_flux": xr.DataArray([-1.0], dims=["x"]),
    }

    tendency_map = {"biomass": ["growth_rate", "mortality_rate", "transport_flux"]}

    integrator = TimeIntegrator()
    new_state = integrator.integrate(state, all_results, tendency_map, dt=1.0)

    # Vérification : 100 + 1*(5-3-1) = 101
    assert new_state["biomass"].item() == 101.0


def test_positivity_constraint():
    """Test que la contrainte de positivité est appliquée."""
    state = xr.Dataset({"biomass": (("x"), [2.0])})

    # Tendance négative forte
    all_results = {"mortality_rate": xr.DataArray([-5.0], dims=["x"])}

    tendency_map = {"biomass": ["mortality_rate"]}

    # Sans contrainte de positivité : 2 + 1*(-5) = -3
    # Avec contrainte : clip à 0
    integrator = TimeIntegrator(positive_vars=["biomass"])
    new_state = integrator.integrate(state, all_results, tendency_map, dt=1.0)

    assert new_state["biomass"].item() == 0.0


def test_no_tendency_for_variable():
    """Test qu'une variable sans tendance reste inchangée."""
    state = xr.Dataset({"biomass": (("x"), [10.0]), "temperature": (("x"), [15.0])})

    all_results = {"growth_rate": xr.DataArray([2.0], dims=["x"])}

    # Seulement biomass a une tendance
    tendency_map = {"biomass": ["growth_rate"]}

    integrator = TimeIntegrator()
    new_state = integrator.integrate(state, all_results, tendency_map, dt=1.0)

    # biomass change, temperature reste
    assert new_state["biomass"].item() == 12.0
    assert new_state["temperature"].item() == 15.0


def test_missing_tendency_in_results():
    """Test qu'une tendance absente des résultats est ignorée (pas d'erreur)."""
    state = xr.Dataset({"biomass": (("x"), [10.0])})

    # missing_tendency n'est pas dans all_results
    all_results = {"growth_rate": xr.DataArray([2.0], dims=["x"])}

    tendency_map = {"biomass": ["growth_rate", "missing_tendency"]}

    integrator = TimeIntegrator()
    new_state = integrator.integrate(state, all_results, tendency_map, dt=1.0)

    # Seul growth_rate est appliqué
    assert new_state["biomass"].item() == 12.0


def test_unknown_scheme():
    """Test qu'un schéma inconnu lève une erreur."""
    state = xr.Dataset({"biomass": (("x"), [10.0])})
    integrator = TimeIntegrator(scheme="rk4")  # Pas encore implémenté

    with pytest.raises(ValueError, match="Unknown scheme"):
        integrator.integrate(state, {}, {}, 1.0)
