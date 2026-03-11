"""Pre-defined model catalog.

Each blueprint is a strict contract: it defines the process chain, the
variables (state, parameters, forcings) **and their dimensions**.  Data
provided via ``Config`` must match those dimensions exactly — the engine
uses them to build the ``vmap`` dispatch.

One blueprint = one topology.  For instance, the LMTL blueprints expect
a 2-D spatial grid ``(Y, X)`` with forcings dimensioned accordingly.
Running a 0-D (box) simulation requires a dedicated blueprint with
scalar declarations, not the same blueprint fed with scalar data.
"""

from importlib.resources import as_file, files

from seapopym.blueprint import Blueprint

_MODELS_DIR = files("seapopym.models")


def load_model(name: str) -> Blueprint:
    """Load a blueprint from the catalog.

    Args:
        name: Model name (e.g. "seapodym_lmtl", "seapodym_lmtl_no_transport").
    """
    with as_file(_MODELS_DIR / f"{name}.yaml") as path:
        return Blueprint.load(path)


LMTL_NO_TRANSPORT = load_model("seapodym_lmtl_no_transport")
LMTL = load_model("seapodym_lmtl")
