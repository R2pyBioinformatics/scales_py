"""Coverage tests for scales/palettes.py – targeting uncovered lines."""

import numpy as np
import pytest

from scales.palettes import (
    ContinuousPalette,
    DiscretePalette,
    new_continuous_palette,
    new_discrete_palette,
    is_pal,
    is_continuous_pal,
    is_discrete_pal,
    is_colour_pal,
    is_numeric_pal,
    palette_nlevels,
    palette_na_safe,
    palette_type,
    as_discrete_pal,
    as_continuous_pal,
    pal_brewer,
    pal_hue,
    pal_viridis,
    pal_grey,
    pal_shape,
    pal_linetype,
    pal_identity,
    pal_manual,
    pal_dichromat,
    pal_gradient_n,
    pal_div_gradient,
    pal_seq_gradient,
    pal_area,
    pal_rescale,
    abs_area,
    # Legacy aliases
    brewer_pal,
    hue_pal,
    viridis_pal,
    grey_pal,
    shape_pal,
    linetype_pal,
    identity_pal,
    manual_pal,
    dichromat_pal,
    gradient_n_pal,
    div_gradient_pal,
    seq_gradient_pal,
    area_pal,
    rescale_pal,
)


# ---------------------------------------------------------------------------
# ContinuousPalette repr (line 251)
# ---------------------------------------------------------------------------

class TestContinuousPaletteRepr:
    def test_repr(self):
        pal = ContinuousPalette(lambda x: x, "numeric")
        r = repr(pal)
        assert "ContinuousPalette" in r
        assert "numeric" in r


# ---------------------------------------------------------------------------
# DiscretePalette repr (line 286)
# ---------------------------------------------------------------------------

class TestDiscretePaletteRepr:
    def test_repr(self):
        pal = DiscretePalette(lambda n: list(range(n)), "numeric", nlevels=5)
        r = repr(pal)
        assert "DiscretePalette" in r
        assert "5" in r


# ---------------------------------------------------------------------------
# new_continuous_palette / new_discrete_palette (lines 316, 340)
# ---------------------------------------------------------------------------

class TestPaletteConstructors:
    def test_new_continuous(self):
        pal = new_continuous_palette(lambda x: x, "numeric")
        assert is_continuous_pal(pal)

    def test_new_discrete(self):
        pal = new_discrete_palette(lambda n: list(range(n)), "colour", nlevels=3)
        assert is_discrete_pal(pal)
        assert palette_nlevels(pal) == 3


# ---------------------------------------------------------------------------
# palette_type (line 401)
# ---------------------------------------------------------------------------

class TestPaletteType:
    def test_colour(self):
        pal = ContinuousPalette(lambda x: x, "colour")
        assert palette_type(pal) == "colour"

    def test_numeric(self):
        pal = DiscretePalette(lambda n: list(range(n)), "numeric")
        assert palette_type(pal) == "numeric"


# ---------------------------------------------------------------------------
# as_discrete_pal (lines 448, 455-464)
# ---------------------------------------------------------------------------

class TestAsDiscretePal:
    def test_already_discrete(self):
        pal = pal_grey()
        assert as_discrete_pal(pal) is pal

    def test_from_continuous(self):
        cpal = pal_gradient_n(["#000000", "#FFFFFF"])
        dpal = as_discrete_pal(cpal)
        assert isinstance(dpal, DiscretePalette)
        # Test sampling 1 vs n
        assert len(dpal(1)) == 1
        assert len(dpal(5)) == 5

    def test_from_string_dichromat(self):
        dpal = as_discrete_pal("Categorical.12")
        assert isinstance(dpal, DiscretePalette)

    def test_from_string_viridis(self):
        dpal = as_discrete_pal("D")
        assert isinstance(dpal, DiscretePalette)

    def test_unknown_string(self):
        with pytest.raises(ValueError):
            as_discrete_pal("nonexistent_palette")

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            as_discrete_pal(42)


# ---------------------------------------------------------------------------
# as_continuous_pal (lines 492-496)
# ---------------------------------------------------------------------------

class TestAsContinuousPal:
    def test_already_continuous(self):
        pal = pal_gradient_n(["#000000", "#FFFFFF"])
        assert as_continuous_pal(pal) is pal

    def test_from_discrete(self):
        dpal = pal_grey()
        cpal = as_continuous_pal(dpal)
        assert isinstance(cpal, ContinuousPalette)

    def test_from_string(self):
        cpal = as_continuous_pal("Categorical.12")
        assert isinstance(cpal, ContinuousPalette)

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            as_continuous_pal(42)


# ---------------------------------------------------------------------------
# pal_brewer edge cases (lines 554-555, 565)
# ---------------------------------------------------------------------------

class TestPalBrewerEdge:
    def test_string_palette_name(self):
        pal = pal_brewer(palette="Set1")
        result = pal(3)
        assert len(result) == 3

    def test_direction_reverse(self):
        pal_fwd = pal_brewer(palette="Blues")
        pal_rev = pal_brewer(palette="Blues", direction=-1)
        fwd = pal_fwd(3)
        rev = pal_rev(3)
        assert fwd[0] == rev[-1]

    def test_div_type(self):
        pal = pal_brewer(type="div")
        result = pal(5)
        assert len(result) == 5

    def test_qual_type(self):
        pal = pal_brewer(type="qual")
        result = pal(5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# pal_hue (line 604, 609)
# ---------------------------------------------------------------------------

class TestPalHueEdge:
    def test_zero_raises(self):
        # R: `if (n == 0) cli::cli_abort("Must request at least one ...")`.
        pal = pal_hue()
        with pytest.raises(ValueError):
            pal(0)

    def test_direction_reverse(self):
        pal = pal_hue(direction=-1)
        result = pal(3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# pal_viridis edge cases (lines 651, 654-655, 658, 668)
# ---------------------------------------------------------------------------

class TestPalViridisEdge:
    def test_zero(self):
        pal = pal_viridis()
        result = pal(0)
        assert result == []

    def test_alpha(self):
        pal = pal_viridis(alpha=0.5)
        result = pal(3)
        # With alpha < 1, hex should be 8 chars (plus #)
        assert all(len(c) == 9 for c in result)

    def test_direction_reverse(self):
        pal = pal_viridis(direction=-1)
        result = pal(3)
        assert len(result) == 3

    def test_option_magma(self):
        pal = pal_viridis(option="A")
        result = pal(3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# pal_grey edge (lines 710, 712)
# ---------------------------------------------------------------------------

class TestPalGreyEdge:
    def test_zero(self):
        pal = pal_grey()
        result = pal(0)
        assert result == []

    def test_one(self):
        pal = pal_grey()
        result = pal(1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# pal_shape (lines 742, 748)
# ---------------------------------------------------------------------------

class TestPalShapeEdge:
    def test_not_solid(self):
        # Per R: c(1, 2, 0, 3, 7, 8) for solid=FALSE.
        pal = pal_shape(solid=False)
        assert pal(3) == [1, 2, 0]
        assert pal(6) == [1, 2, 0, 3, 7, 8]

    def test_too_many(self):
        # Per R: warn rather than abort; positions past max_n come back
        # as NA (None in Python).
        pal = pal_shape()
        with pytest.warns(UserWarning):
            result = pal(8)
        assert result[:6] == [16, 17, 15, 3, 7, 8]
        assert result[6] is None and result[7] is None


# ---------------------------------------------------------------------------
# pal_linetype (lines 771)
# ---------------------------------------------------------------------------

class TestPalLinetypeEdge:
    def test_too_many(self):
        # Per R: warn (via pal_manual) rather than abort; pad with None.
        pal = pal_linetype()
        max_n = pal.nlevels
        with pytest.warns(UserWarning):
            result = pal(max_n + 2)
        assert result[max_n] is None and result[max_n + 1] is None


# ---------------------------------------------------------------------------
# pal_identity (line 791)
# ---------------------------------------------------------------------------

class TestPalIdentity:
    def test_basic(self):
        # Mirrors R: pal_identity()(3) -> 3 (pass-through).
        pal = pal_identity()
        assert pal(5) == 5
        assert pal([1, 2, 3]) == [1, 2, 3]
        assert pal("red") == "red"


# ---------------------------------------------------------------------------
# pal_manual (lines 816, 824)
# ---------------------------------------------------------------------------

class TestPalManualEdge:
    def test_dict_input(self):
        pal = pal_manual({"a": "#FF0000", "b": "#00FF00"})
        result = pal(2)
        assert len(result) == 2

    def test_too_many(self):
        # Per R: warn (not raise) and pad positions past the palette
        # length with NA / None.
        pal = pal_manual(["#FF0000"])
        with pytest.warns(UserWarning):
            result = pal(2)
        assert result[0] == "#FF0000" and result[1] is None


# ---------------------------------------------------------------------------
# pal_dichromat (lines 848-865)
# ---------------------------------------------------------------------------

class TestPalDichromat:
    def test_unknown(self):
        with pytest.raises(ValueError):
            pal_dichromat(name="Unknown")

    def test_too_many(self):
        # Per R: pal_dichromat wraps pal_manual, which warns + pads
        # rather than raising.
        pal = pal_dichromat(name="SteppedSequential.5")
        with pytest.warns(UserWarning):
            result = pal(100)
        max_n = pal.nlevels
        assert result[max_n] is None

    def test_basic(self):
        pal = pal_dichromat()
        result = pal(5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# pal_gradient_n (lines 900, 907)
# ---------------------------------------------------------------------------

class TestPalGradientNEdge:
    def test_single_colour(self):
        pal = pal_gradient_n(["#FF0000"])
        result = pal(np.array([0.0, 0.5, 1.0]))
        assert len(result) == 3
        for c in result:
            assert c is not None

    def test_values_mismatch(self):
        with pytest.raises(ValueError):
            pal_gradient_n(["#FF0000", "#00FF00"], values=[0, 0.5, 1])

    def test_with_values(self):
        pal = pal_gradient_n(["#FF0000", "#00FF00", "#0000FF"],
                             values=[0, 0.3, 1])
        result = pal(np.array([0.0, 0.5, 1.0]))
        assert len(result) == 3

    def test_nan_handling(self):
        pal = pal_gradient_n(["#FF0000", "#0000FF"])
        result = pal(np.array([0.0, float("nan"), 1.0]))
        assert result[1] is None


# ---------------------------------------------------------------------------
# pal_area / abs_area / pal_rescale (line 923)
# ---------------------------------------------------------------------------

class TestPalAreaRescale:
    def test_pal_area(self):
        pal = pal_area(range=(1, 6))
        result = pal(np.array([0, 0.5, 1.0]))
        assert len(result) == 3

    def test_abs_area(self):
        pal = abs_area(max_val=10)
        result = pal(np.array([0, 0.5, 1.0]))
        assert len(result) == 3

    def test_pal_rescale(self):
        pal = pal_rescale(range=(1, 6))
        result = pal(np.array([0, 0.5, 1.0]))
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------

class TestLegacyAliases:
    def test_brewer_pal(self):
        assert brewer_pal is pal_brewer

    def test_hue_pal(self):
        assert hue_pal is pal_hue

    def test_viridis_pal(self):
        assert viridis_pal is pal_viridis

    def test_grey_pal(self):
        assert grey_pal is pal_grey

    def test_shape_pal(self):
        assert shape_pal is pal_shape

    def test_linetype_pal(self):
        assert linetype_pal is pal_linetype

    def test_identity_pal(self):
        assert identity_pal is pal_identity

    def test_manual_pal(self):
        assert manual_pal is pal_manual

    def test_dichromat_pal(self):
        assert dichromat_pal is pal_dichromat

    def test_gradient_n_pal(self):
        assert gradient_n_pal is pal_gradient_n

    def test_div_gradient_pal(self):
        assert div_gradient_pal is pal_div_gradient

    def test_seq_gradient_pal(self):
        assert seq_gradient_pal is pal_seq_gradient

    def test_area_pal(self):
        assert area_pal is pal_area

    def test_rescale_pal(self):
        assert rescale_pal is pal_rescale
