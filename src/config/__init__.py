"""Configuration: parameter registry (name, symbol, unit, range, source)."""

from .defaults import (  # noqa: F401
    Parameter,
    ParameterSource,
    PARAMETER_REGISTRY,
    register,
    get,
    default,
)
