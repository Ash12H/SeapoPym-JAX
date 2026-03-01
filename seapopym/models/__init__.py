"""Catalogue de modeles pre-definis."""

from importlib.resources import as_file, files

from seapopym.blueprint import Blueprint

_MODELS_DIR = files("seapopym.models")


def load_model(name: str) -> Blueprint:
    """Charge un blueprint depuis le catalogue.

    Args:
        name: Nom du modele (ex: "seapodym_lmtl", "seapodym_lmtl_no_transport").
    """
    with as_file(_MODELS_DIR / f"{name}.yaml") as path:
        return Blueprint.load(path)


LMTL_NO_TRANSPORT = load_model("seapodym_lmtl_no_transport")
LMTL = load_model("seapodym_lmtl")
