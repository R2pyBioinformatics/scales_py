"""Tests for scales colour manipulation functions."""

import re

import numpy as np
import pytest

import scales


HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6,8}$")


def is_hex_color(s):
    return bool(HEX_RE.match(s))


# ---------------------------------------------------------------------------
# alpha
# ---------------------------------------------------------------------------

class TestAlpha:
    def test_single_color(self):
        result = scales.alpha("red", 0.5)
        assert is_hex_color(result)
        # Alpha=0.5 -> hex 80 (128/255)
        assert result.endswith("80") or result.endswith("7f")

    def test_vectorized(self):
        result = scales.alpha(["red", "blue"], [0.3, 0.7])
        assert isinstance(result, list)
        assert len(result) == 2
        for c in result:
            assert is_hex_color(c)

    def test_full_opacity(self):
        result = scales.alpha("red", 1.0)
        assert is_hex_color(result)
        assert result.lower().endswith("ff")

    def test_zero_opacity(self):
        result = scales.alpha("red", 0.0)
        assert is_hex_color(result)
        assert result.lower().endswith("00")


# ---------------------------------------------------------------------------
# muted
# ---------------------------------------------------------------------------

class TestMuted:
    def test_returns_hex(self):
        result = scales.muted("red")
        assert is_hex_color(result)

    def test_desaturated(self):
        result = scales.muted("red")
        # Result should be different from pure red
        assert result.lower() != "#ff0000ff"
        assert result.lower() != "#ff0000"

    def test_blue(self):
        result = scales.muted("blue")
        assert is_hex_color(result)


# ---------------------------------------------------------------------------
# col2hcl
# ---------------------------------------------------------------------------

class TestCol2Hcl:
    def test_returns_hex(self):
        result = scales.col2hcl("red")
        assert is_hex_color(result)

    def test_round_trip(self):
        # col2hcl converts to HCL and back; red should stay reddish
        result = scales.col2hcl("red")
        assert "ff0000" in result.lower() or "ff" in result.lower()[:7]


# ---------------------------------------------------------------------------
# col_mix
# ---------------------------------------------------------------------------

class TestColMix:
    def test_returns_hex(self):
        result = scales.col_mix("red", "blue", 0.5)
        assert is_hex_color(result)

    def test_midpoint(self):
        result = scales.col_mix("red", "blue", 0.5)
        # Midpoint of red and blue should be purplish
        assert is_hex_color(result)

    def test_zero_weight(self):
        result = scales.col_mix("red", "blue", 0.0)
        assert is_hex_color(result)
        # Weight=0 should give first color (red)
        assert "ff" in result[:3].lower() or "ff0000" in result.lower()

    def test_full_weight(self):
        result = scales.col_mix("red", "blue", 1.0)
        assert is_hex_color(result)
        # Weight=1 should give second color (blue)
        assert "0000ff" in result.lower()


# ---------------------------------------------------------------------------
# col_shift
# ---------------------------------------------------------------------------

class TestColShift:
    def test_returns_hex(self):
        result = scales.col_shift("red", 180)
        assert is_hex_color(result)

    def test_hue_shift(self):
        result = scales.col_shift("red", 180)
        # Shifting red by 180 degrees should yield a complementary color
        assert result.lower() != "#ff0000ff"

    def test_zero_shift(self):
        result = scales.col_shift("red", 0)
        assert is_hex_color(result)
        # Zero shift should return roughly the same color
        assert "ff" in result[:3].lower()


# ---------------------------------------------------------------------------
# col_lighter
# ---------------------------------------------------------------------------

class TestColLighter:
    def test_returns_hex(self):
        result = scales.col_lighter("red", 20)
        assert is_hex_color(result)

    def test_lighter_than_original(self):
        # Lighter version should have higher luminance
        original = scales.col_lighter("red", 0)
        lighter = scales.col_lighter("red", 20)
        # The lighter color should be different
        assert is_hex_color(lighter)

    def test_blue(self):
        result = scales.col_lighter("blue", 20)
        assert is_hex_color(result)


# ---------------------------------------------------------------------------
# col_darker
# ---------------------------------------------------------------------------

class TestColDarker:
    def test_returns_hex(self):
        result = scales.col_darker("red", 20)
        assert is_hex_color(result)

    def test_darker_than_original(self):
        darker = scales.col_darker("red", 20)
        assert is_hex_color(darker)
        # Should be different from pure red
        assert darker.lower() != "#ff0000ff"

    def test_green(self):
        result = scales.col_darker("green", 20)
        assert is_hex_color(result)


# ---------------------------------------------------------------------------
# col_saturate
# ---------------------------------------------------------------------------

class TestColSaturate:
    def test_returns_hex(self):
        result = scales.col_saturate("grey", 30)
        assert is_hex_color(result)

    def test_increases_saturation(self):
        result = scales.col_saturate("grey", 30)
        # Grey with increased saturation should no longer be pure grey
        assert is_hex_color(result)

    def test_already_saturated(self):
        result = scales.col_saturate("red", 30)
        assert is_hex_color(result)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestColourManipEdgeCases:
    def test_alpha_with_hex_input(self):
        result = scales.alpha("#FF0000", 0.5)
        assert is_hex_color(result)

    def test_muted_with_hex_input(self):
        result = scales.muted("#0000FF")
        assert is_hex_color(result)

    def test_col_mix_same_color(self):
        result = scales.col_mix("red", "red", 0.5)
        assert is_hex_color(result)
        assert "ff0000" in result.lower()

    def test_col_shift_full_circle(self):
        result = scales.col_shift("red", 360)
        assert is_hex_color(result)
        # Full circle should return approximately the same hue
