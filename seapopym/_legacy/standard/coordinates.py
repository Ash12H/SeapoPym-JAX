"""Standard coordinate names used across the application."""

from enum import Enum


class Coordinates(str, Enum):
    """Standard coordinate names."""

    T = "time"
    X = "x"
    Y = "y"
    Z = "z"


class GridPosition(str, Enum):
    """Grid positions following Xgcm conventions.

    Xgcm uses named positions to handle staggered grids:
    - center: cell centers (where state variables typically live)
    - left: left/west/south faces (one more than centers)
    - right: right/east/north faces (one more than centers)
    - inner: inner faces (for nested grids)
    - outer: outer faces (for nested grids)

    For a typical 2D lat/lon grid:
    - X axis: center (lon centers), left (lon faces at west edges)
    - Y axis: center (lat centers), left (lat faces at south edges)

    References:
        https://xgcm.readthedocs.io/en/latest/grid_topology.html
    """

    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    INNER = "inner"
    OUTER = "outer"

    @staticmethod
    def get_face_dim(axis: Coordinates, position: "GridPosition | None" = None) -> str:
        """Get dimension name for a face position.

        Args:
            axis: Coordinate axis (Coordinates.X or Coordinates.Y)
            position: Grid position (default: LEFT)

        Returns:
            Dimension name following Xgcm convention (e.g., "x_left", "y_left")

        Examples:
            >>> GridPosition.get_face_dim(Coordinates.X, GridPosition.LEFT)
            'x_left'
            >>> GridPosition.get_face_dim(Coordinates.Y)
            'y_left'
        """
        if position is None:
            position = GridPosition.LEFT
        return f"{axis.value}_{position.value}"
