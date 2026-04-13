"""Tests for CIELAB colour space interpolation in colour_ramp and pal_gradient_n.

Validates that the Python implementation matches R's farver-based LAB
interpolation, not RGB interpolation.
"""

import numpy as np
import pytest

from scales import colour_ramp, pal_gradient_n, pal_div_gradient, pal_seq_gradient
from scales.colour_manip import _rgb_to_lab
from scales._colors import to_rgba


# ---------------------------------------------------------------------------
# colour_ramp: LAB interpolation correctness
# ---------------------------------------------------------------------------

class TestColourRampLab:
    """Verify that colour_ramp interpolates in CIELAB, not RGB."""

    def test_red_blue_midpoint_is_lab_not_rgb(self):
        """The midpoint of red→blue in LAB should be a bright purple,
        not the dark grey/magenta you get from RGB interpolation.

        In RGB: (1,0,0) + (0,0,1) / 2 = (0.5, 0, 0.5) → dark purple ~#800080
        In LAB: the midpoint preserves perceptual brightness → brighter ~#ca28xx
        """
        ramp = colour_ramp(["red", "blue"])
        mid = ramp([0.5])[0]
        rgba = to_rgba(mid)

        # RGB midpoint would give (0.5, 0, 0.5) → dark.
        # LAB midpoint should be significantly brighter (higher luminance).
        L_mid, _, _ = _rgb_to_lab(rgba[0], rgba[1], rgba[2])

        # In LAB, L of red ≈ 53, L of blue ≈ 32, midpoint L ≈ 43.
        # In RGB interpolation, L of #800080 ≈ 30.
        # The LAB midpoint should have L > 35 (distinguishing it from RGB).
        assert L_mid > 35, f"LAB midpoint too dark (L={L_mid}), likely RGB interpolation"

        # LAB midpoint is significantly brighter than RGB midpoint.
        # RGB midpoint #800080 has R=0.5, LAB midpoint has R>0.7.
        assert rgba[0] > 0.7, (
            f"Red channel too low ({rgba[0]}), looks like RGB interpolation"
        )

    def test_endpoints_exact(self):
        """Endpoints should be exactly the input colours."""
        ramp = colour_ramp(["#2B6788", "#90503F"])
        result = ramp([0.0, 1.0])
        assert result[0].lower().startswith("#2b6788")
        assert result[1].lower().startswith("#90503f")

    def test_three_colour_gradient(self):
        """Three-colour gradient: ensure middle colour is reached at 0.5."""
        ramp = colour_ramp(["#2B6788", "#CBCBCB", "#90503F"])
        result = ramp([0.0, 0.5, 1.0])
        # At x=0.5, should be very close to #CBCBCB
        mid_rgba = to_rgba(result[1])
        expected_rgba = to_rgba("#CBCBCB")
        for i in range(3):
            assert abs(mid_rgba[i] - expected_rgba[i]) < 0.02, (
                f"Channel {i}: {mid_rgba[i]} vs {expected_rgba[i]}"
            )

    def test_monotonic_luminance(self):
        """A white→black ramp should have monotonically decreasing L."""
        ramp = colour_ramp(["white", "black"])
        x = np.linspace(0, 1, 11)
        result = ramp(x)
        L_values = []
        for c in result:
            rgba = to_rgba(c)
            L, _, _ = _rgb_to_lab(rgba[0], rgba[1], rgba[2])
            L_values.append(L)
        for i in range(len(L_values) - 1):
            assert L_values[i] >= L_values[i + 1] - 0.1


# ---------------------------------------------------------------------------
# colour_ramp: edge cases
# ---------------------------------------------------------------------------

class TestColourRampEdgeCases:

    def test_single_colour_constant(self):
        ramp = colour_ramp(["#FF8800"])
        result = ramp([0.0, 0.25, 0.5, 0.75, 1.0])
        assert len(result) == 5
        for c in result:
            assert c.lower().startswith("#ff8800")

    def test_na_handling(self):
        ramp = colour_ramp(["red", "blue"], na_color="#999999")
        result = ramp([0.0, float("nan"), 1.0])
        assert result[0] is not None
        assert result[1] == "#999999"
        assert result[2] is not None

    def test_na_default_none(self):
        ramp = colour_ramp(["red", "blue"])
        result = ramp([float("nan")])
        assert result[0] is None

    def test_alpha_interpolation(self):
        """Alpha should be interpolated between endpoints."""
        ramp = colour_ramp(["#FF000080", "#0000FFFF"], alpha=True)
        result = ramp([0.0, 0.5, 1.0])
        # At 0: alpha = 0x80/255 ≈ 0.502
        # At 1: alpha = 0xFF/255 = 1.0
        # At 0.5: alpha ≈ 0.75
        mid_rgba = to_rgba(result[1])
        assert 0.65 < mid_rgba[3] < 0.85

    def test_no_alpha_mode(self):
        ramp = colour_ramp(["red", "blue"], alpha=False)
        result = ramp([0.0, 0.5, 1.0])
        for c in result:
            # Without alpha, hex should be 7 chars (#RRGGBB)
            assert len(c) == 7

    def test_out_of_range_returns_na(self):
        """Values outside [0,1] return na_color (matching R's approxfun rule=1)."""
        ramp = colour_ramp(["red", "blue"])
        result = ramp([-0.5, 0.0, 1.0, 1.5])
        assert result[0] is None  # -0.5 → na_color (None by default)
        assert result[3] is None  # 1.5 → na_color
        assert result[1] is not None  # 0.0 is valid
        assert result[2] is not None  # 1.0 is valid

    def test_out_of_range_with_na_color(self):
        """Out-of-range maps to explicit na_color when provided."""
        ramp = colour_ramp(["red", "blue"], na_color="#999999")
        result = ramp([-0.1, 1.1])
        assert result[0] == "#999999"
        assert result[1] == "#999999"


# ---------------------------------------------------------------------------
# pal_gradient_n: delegation to colour_ramp
# ---------------------------------------------------------------------------

class TestPalGradientNLab:

    def test_matches_colour_ramp(self):
        """pal_gradient_n without values should match colour_ramp."""
        ramp = colour_ramp(["red", "green", "blue"])
        pal = pal_gradient_n(["red", "green", "blue"])
        x = np.linspace(0, 1, 7)
        ramp_result = ramp(x)
        pal_result = pal(x)
        for r, p in zip(ramp_result, pal_result):
            assert r == p

    def test_values_remapping(self):
        """With values, colours should appear at specified positions."""
        # Place green at 0.2 instead of 0.5
        pal = pal_gradient_n(
            ["red", "green", "blue"],
            values=[0.0, 0.2, 1.0],
        )
        result = pal(np.array([0.0, 0.2, 1.0]))
        # At x=0.2, should be close to green
        rgba = to_rgba(result[1])
        assert rgba[1] > 0.4, f"Expected greenish at x=0.2, got {result[1]}"

    def test_div_gradient_uses_lab(self):
        """pal_div_gradient should also use LAB interpolation."""
        pal = pal_div_gradient()
        result = pal(np.array([0.0, 0.5, 1.0]))
        assert len(result) == 3
        # Midpoint should be close to #CBCBCB
        mid_rgba = to_rgba(result[1])
        expected_rgba = to_rgba("#CBCBCB")
        for i in range(3):
            assert abs(mid_rgba[i] - expected_rgba[i]) < 0.02

    def test_seq_gradient_uses_lab(self):
        """pal_seq_gradient should also use LAB interpolation."""
        pal = pal_seq_gradient()
        result = pal(np.array([0.0, 1.0]))
        assert result[0].lower().startswith("#2b6788")
        assert result[1].lower().startswith("#90503f")

    def test_empty_input(self):
        pal = pal_gradient_n(["red", "blue"])
        result = pal(np.array([]))
        assert result == []
