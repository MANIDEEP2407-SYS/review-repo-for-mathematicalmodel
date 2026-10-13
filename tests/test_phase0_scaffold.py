"""Phase 0 smoke tests: scaffold imports and parameter-registry schema."""

import pytest

from src.config.defaults import (
    Parameter,
    ParameterSource,
    PARAMETER_REGISTRY,
    register,
    get,
)


def test_registry_populated_on_import():
    """After Phase 1, defaults.py registers the standard parameters at import."""
    assert isinstance(PARAMETER_REGISTRY, dict)
    assert len(PARAMETER_REGISTRY) >= 20
    for p in PARAMETER_REGISTRY.values():
        assert isinstance(p.source, ParameterSource)
        assert p.unit != ""


def test_parameter_records_provenance():
    p = Parameter("t", "X", "m", 1.0, (0.0, 10.0), ParameterSource.ASSUMED, "n")
    assert p.source is ParameterSource.ASSUMED


def test_parameter_range_check_accepts_valid():
    p = Parameter("t2", "Y", "W/m^2", 800.0, (0.0, 1500.0), ParameterSource.LITERATURE)
    assert p.validate(900.0) == 900.0


def test_parameter_range_check_rejects_out_of_range():
    p = Parameter("t3", "Z", "-", 0.5, (0.0, 1.0), ParameterSource.ASSUMED)
    with pytest.raises(ValueError):
        p.validate(1.5)


def test_register_and_get_roundtrip():
    p = Parameter("test_only_roundtrip", "R", "-", 0.1, (0.0, 1.0), ParameterSource.ASSUMED)
    register(p)
    try:
        assert get("test_only_roundtrip") is p
    finally:
        PARAMETER_REGISTRY.pop("test_only_roundtrip", None)


def test_register_rejects_duplicate():
    p = Parameter("test_only_dup", "D", "-", 0.0, (0.0, 1.0), ParameterSource.ASSUMED)
    register(p)
    try:
        with pytest.raises(KeyError):
            register(p)
    finally:
        PARAMETER_REGISTRY.pop("test_only_dup", None)
