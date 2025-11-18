"""Transport configuration: declarative specification of fields to transport.

This module provides dataclasses to configure which fields should be transported
during distributed simulations. The configuration is domain-agnostic and extensible.
"""

from dataclasses import dataclass, field


@dataclass
class FieldConfig:
    """Configuration for a single field to transport.

    Args:
        name: Name of the field in the state dict.
        dims: List of dimension names for this field.
             Spatial dimensions (latitude and longitude) should be named 'Y' and 'X'.
             Other dimensions (age, depth, species, etc.) will be iterated over
             during transport.

    Example:
        >>> # 2D field (lat, lon)
        >>> biomass = FieldConfig(name='biomass', dims=['Y', 'X'])
        >>>
        >>> # 3D field (age, lat, lon) - will loop over age dimension
        >>> production = FieldConfig(name='production', dims=['age', 'Y', 'X'])
        >>>
        >>> # 3D field (depth, lat, lon) - will loop over depth dimension
        >>> temperature = FieldConfig(name='temperature', dims=['depth', 'Y', 'X'])
    """

    name: str
    dims: list[str]

    def __post_init__(self) -> None:
        """Validate that spatial dimensions are present."""
        if "Y" not in self.dims or "X" not in self.dims:
            raise ValueError(
                f"Field '{self.name}' must include spatial dimensions 'Y' and 'X'. "
                f"Got dims: {self.dims}"
            )


@dataclass
class TransportConfig:
    """Configuration for transport of multiple fields.

    Args:
        fields: List of FieldConfig instances specifying which fields to transport.
        spatial_dims: Names of spatial dimensions (default: ['Y', 'X']).
                     Transport is performed in 2D over these dimensions.

    Example:
        >>> config = TransportConfig(
        ...     fields=[
        ...         FieldConfig(name='biomass', dims=['Y', 'X']),
        ...         FieldConfig(name='production', dims=['age', 'Y', 'X']),
        ...         FieldConfig(name='temperature', dims=['depth', 'Y', 'X']),
        ...     ]
        ... )
        >>>
        >>> # Query configuration
        >>> config.get_field_names()  # ['biomass', 'production', 'temperature']
        >>> config.get_non_spatial_dims('production')  # ['age']
    """

    fields: list[FieldConfig]
    spatial_dims: list[str] = field(default_factory=lambda: ["Y", "X"])

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.fields:
            raise ValueError("TransportConfig must have at least one field")

        # Check for duplicate field names
        field_names = [f.name for f in self.fields]
        if len(field_names) != len(set(field_names)):
            duplicates = [name for name in field_names if field_names.count(name) > 1]
            raise ValueError(f"Duplicate field names found: {set(duplicates)}")

    def get_field_names(self) -> list[str]:
        """Get list of field names to transport.

        Returns:
            List of field names.

        Example:
            >>> config = TransportConfig(fields=[
            ...     FieldConfig(name='biomass', dims=['Y', 'X']),
            ...     FieldConfig(name='production', dims=['age', 'Y', 'X']),
            ... ])
            >>> config.get_field_names()
            ['biomass', 'production']
        """
        return [f.name for f in self.fields]

    def get_field_config(self, field_name: str) -> FieldConfig:
        """Get configuration for a specific field.

        Args:
            field_name: Name of the field.

        Returns:
            FieldConfig for the requested field.

        Raises:
            ValueError: If field not found in configuration.

        Example:
            >>> config = TransportConfig(fields=[
            ...     FieldConfig(name='biomass', dims=['Y', 'X']),
            ... ])
            >>> field_cfg = config.get_field_config('biomass')
            >>> field_cfg.dims
            ['Y', 'X']
        """
        for field_cfg in self.fields:
            if field_cfg.name == field_name:
                return field_cfg

        raise ValueError(
            f"Field '{field_name}' not found in transport config. "
            f"Available fields: {self.get_field_names()}"
        )

    def get_field_dims(self, field_name: str) -> list[str]:
        """Get dimension names for a field.

        Args:
            field_name: Name of the field.

        Returns:
            List of dimension names.

        Example:
            >>> config = TransportConfig(fields=[
            ...     FieldConfig(name='production', dims=['age', 'Y', 'X']),
            ... ])
            >>> config.get_field_dims('production')
            ['age', 'Y', 'X']
        """
        field_cfg = self.get_field_config(field_name)
        return field_cfg.dims

    def get_non_spatial_dims(self, field_name: str) -> list[str]:
        """Get non-spatial dimensions for a field.

        Non-spatial dimensions are those not in self.spatial_dims.
        These are the dimensions we need to loop over during transport.

        Args:
            field_name: Name of the field.

        Returns:
            List of non-spatial dimension names (for looping during transport).

        Example:
            >>> config = TransportConfig(fields=[
            ...     FieldConfig(name='biomass', dims=['Y', 'X']),
            ...     FieldConfig(name='production', dims=['age', 'Y', 'X']),
            ...     FieldConfig(name='temperature', dims=['species', 'depth', 'Y', 'X']),
            ... ])
            >>>
            >>> config.get_non_spatial_dims('biomass')  # Pure 2D field
            []
            >>> config.get_non_spatial_dims('production')  # Loop over age
            ['age']
            >>> config.get_non_spatial_dims('temperature')  # Loop over species and depth
            ['species', 'depth']
        """
        field_dims = self.get_field_dims(field_name)
        return [d for d in field_dims if d not in self.spatial_dims]

    def is_2d_field(self, field_name: str) -> bool:
        """Check if a field is purely 2D (only spatial dimensions).

        Args:
            field_name: Name of the field.

        Returns:
            True if field has only spatial dimensions, False otherwise.

        Example:
            >>> config = TransportConfig(fields=[
            ...     FieldConfig(name='biomass', dims=['Y', 'X']),
            ...     FieldConfig(name='production', dims=['age', 'Y', 'X']),
            ... ])
            >>>
            >>> config.is_2d_field('biomass')
            True
            >>> config.is_2d_field('production')
            False
        """
        return len(self.get_non_spatial_dims(field_name)) == 0

    def __repr__(self) -> str:
        """String representation."""
        field_names = self.get_field_names()
        return f"TransportConfig(fields={field_names}, spatial_dims={self.spatial_dims})"
