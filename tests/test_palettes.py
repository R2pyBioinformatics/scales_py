"""Tests for scales palette system."""

import re

import numpy as np
import pytest

import scales


HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6,8}$")


def is_hex_color(s):
    return bool(HEX_RE.match(s))


# ---------------------------------------------------------------------------
# pal_brewer
# ---------------------------------------------------------------------------

class TestPalBrewer:
    def test_returns_discrete_palette(self):
        p = scales.pal_brewer("qual", "Set1")
        assert isinstance(p, scales.DiscretePalette)

    def test_returns_n_colors(self):
        p = scales.pal_brewer("qual", "Set1")
        colors = p(5)
        assert len(colors) == 5

    def test_colors_are_hex(self):
        p = scales.pal_brewer("qual", "Set1")
        for c in p(5):
            assert is_hex_color(c), f"{c} is not a valid hex color"

    def test_different_palette_types(self):
        for ptype in ["qual", "seq", "div"]:
            p = scales.pal_brewer(ptype)
            colors = p(3)
            assert len(colors) == 3

    def test_is_colour_pal(self):
        p = scales.pal_brewer("qual", "Set1")
        assert scales.is_colour_pal(p)


# ---------------------------------------------------------------------------
# pal_viridis
# ---------------------------------------------------------------------------

class TestPalViridis:
    def test_returns_discrete_palette(self):
        p = scales.pal_viridis()
        assert isinstance(p, scales.DiscretePalette)

    def test_returns_n_colors(self):
        p = scales.pal_viridis()
        colors = p(5)
        assert len(colors) == 5

    def test_colors_are_hex(self):
        p = scales.pal_viridis()
        for c in p(5):
            assert is_hex_color(c), f"{c} is not a valid hex color"

    def test_known_values(self):
        p = scales.pal_viridis()
        colors = p(5)
        assert colors[0].lower() == "#440154"
        assert colors[-1].lower() == "#fde725"


# ---------------------------------------------------------------------------
# pal_hue
# ---------------------------------------------------------------------------

class TestPalHue:
    def test_returns_discrete_palette(self):
        p = scales.pal_hue()
        assert isinstance(p, scales.DiscretePalette)

    def test_returns_n_colors(self):
        p = scales.pal_hue()
        colors = p(3)
        assert len(colors) == 3

    def test_colors_are_hex(self):
        p = scales.pal_hue()
        for c in p(4):
            assert is_hex_color(c)

    def test_evenly_spaced(self):
        # Should return distinct colors
        p = scales.pal_hue()
        colors = p(3)
        assert len(set(colors)) == 3


# ---------------------------------------------------------------------------
# pal_grey
# ---------------------------------------------------------------------------

class TestPalGrey:
    def test_returns_discrete_palette(self):
        p = scales.pal_grey()
        assert isinstance(p, scales.DiscretePalette)

    def test_returns_hex_greys(self):
        p = scales.pal_grey()
        colors = p(3)
        assert len(colors) == 3
        for c in colors:
            assert is_hex_color(c)

    def test_grey_bounds(self):
        p = scales.pal_grey()
        colors = p(3)
        # Grey colors have equal R, G, B channels
        for c in colors:
            r, g, b = c[1:3], c[3:5], c[5:7]
            assert r == g == b


# ---------------------------------------------------------------------------
# pal_manual
# ---------------------------------------------------------------------------

class TestPalManual:
    def test_returns_discrete_palette(self):
        p = scales.pal_manual(["red", "blue", "green"])
        assert isinstance(p, scales.DiscretePalette)

    def test_returns_provided_colors(self):
        p = scales.pal_manual(["red", "blue", "green"])
        colors = p(3)
        assert colors == ["red", "blue", "green"]

    def test_fewer_than_available(self):
        p = scales.pal_manual(["red", "blue", "green"])
        colors = p(2)
        assert len(colors) == 2
        assert colors == ["red", "blue"]


# ---------------------------------------------------------------------------
# pal_identity
# ---------------------------------------------------------------------------

class TestPalIdentity:
    def test_returns_discrete_palette(self):
        p = scales.pal_identity()
        assert isinstance(p, scales.DiscretePalette)


# ---------------------------------------------------------------------------
# pal_shape
# ---------------------------------------------------------------------------

class TestPalShape:
    def test_returns_integers(self):
        p = scales.pal_shape()
        shapes = p(3)
        assert len(shapes) == 3
        for s in shapes:
            assert isinstance(s, (int, np.integer))

    def test_known_values(self):
        p = scales.pal_shape()
        # R default: c(16, 17, 15) = filled circle, triangle, square
        assert p(3) == [16, 17, 15]


# ---------------------------------------------------------------------------
# pal_linetype
# ---------------------------------------------------------------------------

class TestPalLinetype:
    def test_returns_strings(self):
        p = scales.pal_linetype()
        types = p(3)
        assert len(types) == 3
        for t in types:
            assert isinstance(t, str)

    def test_known_values(self):
        p = scales.pal_linetype()
        types = p(3)
        assert types == ["solid", "dashed", "dotted"]


# ---------------------------------------------------------------------------
# pal_gradient_n (continuous)
# ---------------------------------------------------------------------------

class TestPalGradientN:
    def test_returns_continuous_palette(self):
        p = scales.pal_gradient_n(["red", "white", "blue"])
        assert isinstance(p, scales.ContinuousPalette)

    def test_endpoints(self):
        p = scales.pal_gradient_n(["red", "white", "blue"])
        result = p([0, 0.5, 1.0])
        assert len(result) == 3
        # At 0 should be reddish, at 1 should be bluish
        assert "FF0000" in result[0].upper() or "ff0000" in result[0].lower()
        assert "0000FF" in result[2].upper() or "0000ff" in result[2].lower()

    def test_midpoint(self):
        p = scales.pal_gradient_n(["red", "white", "blue"])
        result = p([0.5])
        # Midpoint should be close to white
        assert "FE" in result[0].upper() or "FF" in result[0].upper()


# ---------------------------------------------------------------------------
# pal_div_gradient
# ---------------------------------------------------------------------------

class TestPalDivGradient:
    def test_returns_continuous_palette(self):
        p = scales.pal_div_gradient()
        assert isinstance(p, scales.ContinuousPalette)

    def test_returns_colors(self):
        p = scales.pal_div_gradient()
        result = p([0, 0.5, 1.0])
        assert len(result) == 3
        for c in result:
            assert is_hex_color(c)


# ---------------------------------------------------------------------------
# pal_seq_gradient
# ---------------------------------------------------------------------------

class TestPalSeqGradient:
    def test_returns_continuous_palette(self):
        p = scales.pal_seq_gradient()
        assert isinstance(p, scales.ContinuousPalette)

    def test_returns_colors(self):
        p = scales.pal_seq_gradient()
        result = p([0, 0.5, 1.0])
        assert len(result) == 3
        for c in result:
            assert is_hex_color(c)


# ---------------------------------------------------------------------------
# pal_area
# ---------------------------------------------------------------------------

class TestPalArea:
    def test_returns_continuous_palette(self):
        p = scales.pal_area()
        assert isinstance(p, scales.ContinuousPalette)

    def test_returns_numeric(self):
        p = scales.pal_area()
        result = p([0, 0.5, 1.0])
        assert len(result) == 3
        # Should be numeric
        np.testing.assert_allclose(result[0], 1.0, atol=0.1)

    def test_monotonically_increasing(self):
        p = scales.pal_area()
        result = p([0, 0.25, 0.5, 0.75, 1.0])
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]


# ---------------------------------------------------------------------------
# pal_rescale
# ---------------------------------------------------------------------------

class TestPalRescale:
    def test_returns_continuous_palette(self):
        p = scales.pal_rescale()
        assert isinstance(p, scales.ContinuousPalette)

    def test_rescales_to_range(self):
        p = scales.pal_rescale()
        result = p([0, 0.5, 1.0])
        assert len(result) == 3
        # Default range is (0.1, 1)
        np.testing.assert_allclose(result[0], 0.1, atol=0.01)
        np.testing.assert_allclose(result[2], 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# abs_area
# ---------------------------------------------------------------------------

class TestAbsArea:
    def test_returns_continuous_palette(self):
        p = scales.abs_area(10)
        assert isinstance(p, scales.ContinuousPalette)

    def test_zero_maps_to_zero(self):
        p = scales.abs_area(10)
        result = p([0, 0.5, 1.0])
        np.testing.assert_allclose(result[0], 0.0, atol=0.01)

    def test_values_increase(self):
        p = scales.abs_area(10)
        result = p([0, 0.5, 1.0])
        assert result[0] < result[1] < result[2]


# ---------------------------------------------------------------------------
# Palette introspection
# ---------------------------------------------------------------------------

class TestPaletteIntrospection:
    def test_is_pal(self):
        assert scales.is_pal(scales.pal_brewer("qual", "Set1"))
        assert scales.is_pal(scales.pal_gradient_n(["red", "blue"]))
        assert not scales.is_pal(lambda x: x)

    def test_is_continuous_pal(self):
        assert scales.is_continuous_pal(scales.pal_gradient_n(["red", "blue"]))
        assert not scales.is_continuous_pal(scales.pal_brewer("qual", "Set1"))

    def test_is_discrete_pal(self):
        assert scales.is_discrete_pal(scales.pal_brewer("qual", "Set1"))
        assert not scales.is_discrete_pal(scales.pal_gradient_n(["red", "blue"]))

    def test_palette_type_colour(self):
        assert scales.palette_type(scales.pal_brewer("qual", "Set1")) == "colour"

    def test_palette_type_numeric(self):
        assert scales.palette_type(scales.pal_area()) == "numeric"

    def test_palette_nlevels(self):
        p = scales.pal_brewer("qual", "Set1")
        # May be None or an integer
        result = scales.palette_nlevels(p)
        assert result is None or isinstance(result, int)

    def test_is_colour_pal(self):
        assert scales.is_colour_pal(scales.pal_brewer("qual", "Set1"))
        assert not scales.is_colour_pal(scales.pal_area())

    def test_is_numeric_pal(self):
        assert scales.is_numeric_pal(scales.pal_area())
        assert not scales.is_numeric_pal(scales.pal_brewer("qual", "Set1"))


# ---------------------------------------------------------------------------
# as_discrete_pal / as_continuous_pal
# ---------------------------------------------------------------------------

class TestPaletteCoercion:
    def test_as_discrete_pal(self):
        p = scales.pal_brewer("qual", "Set1")
        dp = scales.as_discrete_pal(p)
        assert isinstance(dp, scales.DiscretePalette)

    def test_as_continuous_pal(self):
        p = scales.pal_gradient_n(["red", "blue"])
        cp = scales.as_continuous_pal(p)
        assert isinstance(cp, scales.ContinuousPalette)

    def test_as_discrete_from_continuous(self):
        p = scales.pal_gradient_n(["red", "blue"])
        dp = scales.as_discrete_pal(p)
        assert isinstance(dp, scales.DiscretePalette)
        # Should still produce colors
        colors = dp(3)
        assert len(colors) == 3

    def test_as_continuous_from_discrete(self):
        p = scales.pal_brewer("qual", "Set1")
        cp = scales.as_continuous_pal(p)
        assert isinstance(cp, scales.ContinuousPalette)
