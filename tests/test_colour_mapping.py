"""Tests for scales colour mapping functions."""

import re

import numpy as np
import pytest

import scales


HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6,8}$")


def is_hex_color(s):
    return bool(HEX_RE.match(s))


# ---------------------------------------------------------------------------
# col_numeric
# ---------------------------------------------------------------------------

class TestColNumeric:
    def test_basic_mapping(self):
        cn = scales.col_numeric(["red", "blue"], domain=(0, 1))
        r0 = cn(0)
        r1 = cn(1)
        assert isinstance(r0, list)
        assert isinstance(r1, list)
        assert len(r0) == 1
        assert len(r1) == 1
        assert is_hex_color(r0[0])
        assert is_hex_color(r1[0])

    def test_endpoints(self):
        cn = scales.col_numeric(["red", "blue"], domain=(0, 1))
        assert "ff0000" in cn(0)[0].lower()
        assert "0000ff" in cn(1)[0].lower()

    def test_midpoint(self):
        cn = scales.col_numeric(["red", "blue"], domain=(0, 1))
        mid = cn(0.5)
        assert len(mid) == 1
        assert is_hex_color(mid[0])
        # Midpoint should be neither pure red nor pure blue
        assert mid[0].lower() != "#ff0000ff"
        assert mid[0].lower() != "#0000ffff"

    def test_vector_input(self):
        cn = scales.col_numeric(["red", "blue"], domain=(0, 1))
        result = cn([0, 0.5, 1])
        assert len(result) == 3

    def test_auto_domain(self):
        cn = scales.col_numeric(["red", "blue"])
        # Without domain it should still be callable
        assert callable(cn)

    def test_na_handling(self):
        cn = scales.col_numeric(["red", "blue"], domain=(0, 1))
        result = cn([np.nan, 0.5])
        assert len(result) == 2
        # NaN should map to a neutral/grey color
        assert is_hex_color(result[0]) or result[0] == "#808080"
        assert is_hex_color(result[1])

    def test_single_nan(self):
        cn = scales.col_numeric(["red", "blue"], domain=(0, 1))
        result = cn(np.nan)
        assert len(result) == 1
        # NA color is typically grey
        assert "#808080" in result[0]

    def test_three_colors(self):
        cn = scales.col_numeric(["red", "white", "blue"], domain=(0, 1))
        r0 = cn(0)
        rmid = cn(0.5)
        r1 = cn(1)
        assert "ff0000" in r0[0].lower()
        assert "0000ff" in r1[0].lower()


# ---------------------------------------------------------------------------
# col_bin
# ---------------------------------------------------------------------------

class TestColBin:
    def test_basic(self):
        cb = scales.col_bin(["red", "blue"], domain=(0, 1), bins=2)
        r_low = cb(0.25)
        r_high = cb(0.75)
        assert isinstance(r_low, list)
        assert isinstance(r_high, list)
        assert len(r_low) == 1
        assert len(r_high) == 1

    def test_different_bins(self):
        cb = scales.col_bin(["red", "blue"], domain=(0, 1), bins=2)
        r_low = cb(0.25)
        r_high = cb(0.75)
        # The two bins should map to different colors
        assert r_low[0] != r_high[0]

    def test_vector_input(self):
        cb = scales.col_bin(["red", "blue"], domain=(0, 1), bins=2)
        result = cb([0.1, 0.3, 0.7, 0.9])
        assert len(result) == 4

    def test_na_handling(self):
        cb = scales.col_bin(["red", "blue"], domain=(0, 1), bins=2)
        result = cb([np.nan, 0.5])
        assert len(result) == 2
        assert "#808080" in result[0]

    def test_hex_output(self):
        cb = scales.col_bin(["red", "blue"], domain=(0, 1), bins=2)
        result = cb(0.5)
        assert is_hex_color(result[0])


# ---------------------------------------------------------------------------
# col_factor
# ---------------------------------------------------------------------------

class TestColFactor:
    def test_basic(self):
        cf = scales.col_factor(["red", "blue"], domain=["a", "b"])
        ra = cf("a")
        rb = cf("b")
        assert isinstance(ra, list)
        assert len(ra) == 1
        assert len(rb) == 1

    def test_maps_correctly(self):
        cf = scales.col_factor(["red", "blue"], domain=["a", "b"])
        assert "ff0000" in cf("a")[0].lower()
        assert "0000ff" in cf("b")[0].lower()

    def test_vector_input(self):
        cf = scales.col_factor(["red", "blue"], domain=["a", "b"])
        result = cf(["a", "b", "a"])
        assert len(result) == 3
        assert result[0] == result[2]  # same category, same color

    def test_na_handling(self):
        cf = scales.col_factor(["red", "blue"], domain=["a", "b"])
        result = cf([None, "a"])
        assert len(result) == 2
        assert "#808080" in result[0]

    def test_unknown_level(self):
        cf = scales.col_factor(["red", "blue"], domain=["a", "b"])
        result = cf(["c", "a"])
        assert len(result) == 2
        # Unknown level should get NA color
        assert "#808080" in result[0]


# ---------------------------------------------------------------------------
# col_quantile
# ---------------------------------------------------------------------------

class TestColQuantile:
    def test_basic(self):
        cq = scales.col_quantile(["red", "blue"], domain=np.arange(100))
        r_low = cq(10)
        r_high = cq(90)
        assert isinstance(r_low, list)
        assert isinstance(r_high, list)
        assert len(r_low) == 1
        assert len(r_high) == 1

    def test_different_quantiles(self):
        cq = scales.col_quantile(["red", "blue"], domain=np.arange(100))
        r_low = cq(10)
        r_high = cq(90)
        # Low and high quantile values should map to different colors
        assert r_low[0] != r_high[0]

    def test_na_handling(self):
        cq = scales.col_quantile(["red", "blue"], domain=np.arange(100))
        result = cq(np.nan)
        assert len(result) == 1
        assert "#808080" in result[0]

    def test_vector_input(self):
        cq = scales.col_quantile(["red", "blue"], domain=np.arange(100))
        result = cq([10, 50, 90])
        assert len(result) == 3

    def test_hex_output(self):
        cq = scales.col_quantile(["red", "blue"], domain=np.arange(100))
        result = cq(50)
        assert is_hex_color(result[0])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestColourMappingEdgeCases:
    def test_col_numeric_all_nan(self):
        cn = scales.col_numeric(["red", "blue"], domain=(0, 1))
        result = cn([np.nan, np.nan])
        assert len(result) == 2
        for c in result:
            assert "#808080" in c

    def test_col_factor_single_level(self):
        cf = scales.col_factor(["red", "blue"], domain=["a"])
        result = cf("a")
        assert len(result) == 1
        assert is_hex_color(result[0])

    def test_col_numeric_out_of_domain(self):
        cn = scales.col_numeric(["red", "blue"], domain=(0, 1))
        # Values outside domain may be clamped or produce NA
        result = cn([-0.5, 1.5])
        assert len(result) == 2

    def test_col_bin_boundary_values(self):
        cb = scales.col_bin(["red", "blue"], domain=(0, 1), bins=2)
        result = cb([0, 0.5, 1])
        assert len(result) == 3
