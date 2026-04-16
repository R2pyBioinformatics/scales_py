"""Shared pytest fixtures for scales_py."""

import pytest


@pytest.fixture(autouse=True)
def _reset_number_options():
    """Snapshot + restore the global number_options store around every
    test so state set by one test (decimal_mark, big_mark, …) cannot
    leak into later tests."""
    from scales.labels import _NUMBER_OPTIONS

    snapshot = dict(_NUMBER_OPTIONS)
    try:
        yield
    finally:
        _NUMBER_OPTIONS.clear()
        _NUMBER_OPTIONS.update(snapshot)


@pytest.fixture(autouse=True)
def _reset_palette_registry():
    """Snapshot + restore the palette registry so `register_palette()`
    calls in one test do not leak to the next."""
    from scales.palettes import _KNOWN_PALETTES

    snapshot = dict(_KNOWN_PALETTES)
    try:
        yield
    finally:
        _KNOWN_PALETTES.clear()
        _KNOWN_PALETTES.update(snapshot)
