"""Catalogue de modeles pre-definis."""

from importlib.resources import files

from seapopym.blueprint import Blueprint

_MODELS_DIR = files("seapopym.models")


def load_model(name: str) -> Blueprint:
    """Charge un blueprint depuis le catalogue.

    Args:
        name: Nom du modele (ex: "lmtl_0d", "lmtl_2d").
    """
    path = _MODELS_DIR / f"{name}.yaml"
    return Blueprint.from_yaml(path)


LMTL_0D = load_model("lmtl_0d")
LMTL_2D = load_model("lmtl_2d")
