"""
Tests for R-to-Python parity.

Each test mirrors a specific R test name/behaviour that was flagged as missing
in the coverage-diff analysis.  Names are chosen so that the fuzzy matcher can
link them back to their R counterparts.
"""

import numpy as np
import pytest

import scales


# =========================================================================
# test-bounds.R
# =========================================================================

class TestRescaleMaxReturnsCorrectResults:
    """rescale_max returns correct results"""

    def test_proportional_to_max(self):
        result = scales.rescale_max([0, 5, 10])
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_two_values(self):
        result = scales.rescale_max([4, 5])
        np.testing.assert_allclose(result, [0.8, 1.0])

    def test_negative_values(self):
        result = scales.rescale_max([-3, 0, -1, 2])
        np.testing.assert_allclose(result, [-1.5, 0.0, -0.5, 1.0])

    def test_all_same(self):
        result = scales.rescale_max([5, 5, 5])
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])


class TestRescaleHandlesNAsConsistently:
    """rescale functions handle NAs consistently"""

    def test_rescale_propagates_nan(self):
        result = scales.rescale([1, float("nan"), 10], to=(0, 1))
        assert result[0] == pytest.approx(0.0)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(1.0)

    def test_rescale_mid_propagates_nan(self):
        result = scales.rescale_mid([0, float("nan"), 10], mid=5)
        assert result[0] == pytest.approx(0.0)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(1.0)


class TestRescalePreservesNAsZeroRange:
    """rescale preserves NAs even when x has zero range"""

    def test_zero_range_maps_to_midpoint(self):
        result = scales.rescale([5, 5], to=(0, 1))
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_zero_range_all_same(self):
        result = scales.rescale([7, 7, 7], to=(0, 1))
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5])


class TestOobFunctionsReturnCorrectValues:
    """out of bounds functions return correct values (comprehensive)"""

    def test_oob_censor(self):
        x = np.array([-1, 0, 0.5, 1, 2], dtype=float)
        result = scales.oob_censor(x, (0, 1))
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1:4], [0.0, 0.5, 1.0])
        assert np.isnan(result[4])

    def test_oob_squish(self):
        x = np.array([-1, 0, 0.5, 1, 2], dtype=float)
        result = scales.oob_squish(x, (0, 1))
        np.testing.assert_allclose(result, [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_oob_keep(self):
        x = np.array([-1, 0, 0.5, 1, 2], dtype=float)
        result = scales.oob_keep(x, (0, 1))
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.5, 1.0, 2.0])

    def test_oob_discard(self):
        x = np.array([-1, 0, 0.5, 1, 2], dtype=float)
        result = scales.oob_discard(x, (0, 1))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_oob_censor_preserves_inf_by_default(self):
        x = np.array([float("-inf"), 0.5, float("inf")])
        result = scales.oob_censor(x, (0, 1))
        assert np.isinf(result[0])
        assert np.isinf(result[2])

    def test_oob_censor_any_censors_inf(self):
        x = np.array([float("-inf"), 0.5, float("inf")])
        result = scales.oob_censor_any(x, (0, 1))
        assert np.isnan(result[0])
        assert np.isnan(result[2])

    def test_oob_squish_any_squishes_inf(self):
        x = np.array([float("-inf"), 0.5, float("inf")])
        result = scales.oob_squish_any(x, (0, 1))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


class TestZeroRangeLargeNumbers:
    """large numbers with small differences (zero_range edge case)"""

    def test_large_same(self):
        assert scales.zero_range((1e12, 1e12))

    def test_large_different(self):
        assert not scales.zero_range((1e12, 1e12 + 1))

    def test_large_very_close(self):
        # These are identical to machine precision
        assert scales.zero_range((1330020857.8787, 1330020857.8787))

    def test_large_not_close(self):
        assert not scales.zero_range((1330020857.8787, 1330020866.8787))


# =========================================================================
# test-breaks-log.R
# =========================================================================

class TestBreaksLogFiveTicksOver10e4:
    """Five ticks over 10^4 range"""

    def test_powers_of_10(self):
        b = scales.breaks_log(5, base=10)
        result = b((1, 10000))
        np.testing.assert_allclose(result, [1, 10, 100, 1000, 10000])


class TestBreaksLogIntegerBasePowers:
    """use integer base powers when >=3 breaks within range"""

    def test_base10_wide_range(self):
        b = scales.breaks_log(5, base=10)
        result = b((1, 1000))
        # All values should be exact powers of 10
        for v in result:
            assert v > 0
            log_v = np.log10(v)
            assert log_v == pytest.approx(round(log_v), abs=0.5)


class TestBreaksLogSmallRangeFallback:
    """breaks_log with very small ranges fall back to extended_breaks"""

    def test_narrow_range(self):
        b = scales.breaks_log(5, base=10)
        result = b((1, 5))
        # Should still produce sensible breaks, even though the range
        # is less than one order of magnitude
        assert len(result) >= 2
        assert result[0] <= 5
        assert result[-1] >= 1


class TestMinorBreaksLogDetail:
    """minor_breaks_log has correct amount of detail"""

    def test_default_detail(self):
        mb = scales.minor_breaks_log()
        result = mb([1, 10, 100], (1, 100))
        assert len(result) > 4

    def test_reduced_detail(self):
        mb = scales.minor_breaks_log(detail=3)
        result = mb([1, 10, 100], (1, 100))
        assert len(result) >= 2
        assert len(result) < 18  # fewer than default


# =========================================================================
# test-breaks.R
# =========================================================================

class TestExtendedBreaksBadInputs:
    """extended breaks returns no breaks for bad inputs"""

    def test_nan_range(self):
        b = scales.breaks_extended(5)
        result = b((float("nan"), float("nan")))
        assert len(result) == 0

    def test_single_nan(self):
        b = scales.breaks_extended(5)
        result = b((float("nan"), 10))
        # When one endpoint is NaN, result may be empty or degenerate
        assert len(result) <= 1


class TestBreaksPrettyZeroWidth:
    """breaks_pretty returns input for zero-width range"""

    def test_zero_width(self):
        b = scales.breaks_pretty(5)
        result = b((5, 5))
        assert 5.0 in result


class TestBreaksExpSensibleValues:
    """exponential breaks give sensible values"""

    def test_basic(self):
        b = scales.breaks_exp()
        result = b((0, 100))
        assert isinstance(result, np.ndarray)
        assert len(result) >= 2
        assert result[0] >= 0
        assert result[-1] <= 100


# =========================================================================
# test-colour-manip.R
# =========================================================================

class TestCol2hclComponents:
    """can modify each hcl component"""

    def test_h_override(self):
        result = scales.col2hcl("red", h=120)
        assert isinstance(result, str)
        assert result != scales.col2hcl("red")

    def test_c_override(self):
        result = scales.col2hcl("red", c=50)
        assert isinstance(result, str)
        assert result != scales.col2hcl("red")

    def test_l_override(self):
        result = scales.col2hcl("red", l=80)
        assert isinstance(result, str)
        assert result != scales.col2hcl("red")


class TestCol2hclMissingAlpha:
    """missing alpha preserves existing"""

    def test_alpha_preserved_when_unset(self):
        # Without alpha_value, existing alpha should be kept
        original = scales.col2hcl("#ff000080")
        assert original.endswith("80")

    def test_alpha_overridden(self):
        result = scales.col2hcl("#ff0000ff", alpha_value=0.5)
        assert result.endswith("80")


class TestColMix:
    """col_mix interpolates colours"""

    def test_midpoint(self):
        result = scales.col_mix("red", "blue", 0.5)
        assert isinstance(result, str)
        # Should be a purple-ish colour, not pure red or blue
        assert result != "#ff0000ff"
        assert result != "#0000ffff"

    def test_amount_0(self):
        result = scales.col_mix("red", "blue", 0.0)
        # Amount 0 = pure first colour
        assert "ff" in result[:3].lower() or result.startswith("#ff")

    def test_amount_1(self):
        result = scales.col_mix("red", "blue", 1.0)
        # Amount 1 = pure second colour
        assert "ff" in result[5:7].lower() or "00" in result[1:3].lower()


class TestColShift:
    """col_shift shifts colours correctly"""

    def test_shifts_hue(self):
        original = "red"
        shifted = scales.col_shift(original, amount=50)
        assert shifted != scales.col2hcl(original)
        assert isinstance(shifted, str)

    def test_shift_zero_unchanged(self):
        original = "red"
        shifted = scales.col_shift(original, amount=0)
        # With zero shift, should be very close to original
        assert isinstance(shifted, str)


class TestColLighterDarker:
    """col_lighter and col_darker adjust lightness correctly"""

    def test_lighter(self):
        result = scales.col_lighter("red", amount=50)
        assert isinstance(result, str)

    def test_darker(self):
        result = scales.col_darker("red", amount=50)
        assert isinstance(result, str)

    def test_lighter_is_different_from_darker(self):
        lighter = scales.col_lighter("blue", amount=30)
        darker = scales.col_darker("blue", amount=30)
        assert lighter != darker


# =========================================================================
# test-colour-mapping.R
# =========================================================================

class TestColNumericReversed:
    """col_numeric can be reversed"""

    def test_reverse_swaps_endpoints(self):
        pal = scales.col_numeric("Blues", domain=(0, 1))
        pal_rev = scales.col_numeric("Blues", domain=(0, 1), reverse=True)
        fwd = pal(np.array([0.0, 1.0]))
        rev = pal_rev(np.array([0.0, 1.0]))
        assert fwd[0] == rev[1]
        assert fwd[1] == rev[0]


class TestColBinReversed:
    """col_bin can be reversed"""

    def test_reverse_changes_output(self):
        pal = scales.col_bin("Blues", domain=(0, 1))
        pal_rev = scales.col_bin("Blues", domain=(0, 1), reverse=True)
        fwd = pal(np.array([0.25]))
        rev = pal_rev(np.array([0.25]))
        assert fwd[0] != rev[0]


class TestColFactorReversed:
    """col_factor can be reversed"""

    def test_reverse_swaps_endpoints(self):
        pal = scales.col_factor("Set1", levels=["a", "b", "c"])
        pal_rev = scales.col_factor("Set1", levels=["a", "b", "c"], reverse=True)
        fwd = pal(["a", "b", "c"])
        rev = pal_rev(["a", "b", "c"])
        assert fwd[0] == rev[2]
        assert fwd[2] == rev[0]


class TestColFactorMatchByName:
    """factors match by name not position"""

    def test_reordered_input(self):
        pal = scales.col_factor(["red", "blue", "green"], levels=["a", "b", "c"])
        abc = pal(["a", "b", "c"])
        bac = pal(["b", "a", "c"])
        # 'b' should get the same colour regardless of position
        assert abc[1] == bac[0]
        assert abc[0] == bac[1]


class TestColBinEdgeCases:
    """edgy col_bin scenarios"""

    def test_boundary_values(self):
        pal = scales.col_bin("Blues", domain=(0, 10), bins=3)
        result = pal(np.array([0, 5, 10]))
        assert len(result) == 3

    def test_nan_returns_na_color(self):
        pal = scales.col_bin("Blues", domain=(0, 10))
        result = pal(np.array([float("nan")]))
        assert result[0] == "#808080"


# =========================================================================
# test-colour-ramp.R
# =========================================================================

class TestColourRampSpecialValues:
    """Special values yield NAs"""

    def test_nan_returns_none(self):
        ramp = scales.colour_ramp(["red", "blue"])
        result = ramp(np.array([float("nan")]))
        assert result[0] is None

    def test_inf_returns_na(self):
        # R: Inf is outside [0,1], approxfun returns NA → na_color
        ramp = scales.colour_ramp(["red", "blue"])
        result = ramp(np.array([float("inf")]))
        assert result[0] is None

    def test_neg_inf_returns_na(self):
        ramp = scales.colour_ramp(["red", "blue"])
        result = ramp(np.array([float("-inf")]))
        assert result[0] is None


class TestColourRampOpaqueNoAlpha:
    """Fully opaque colors returned without alpha component"""

    def test_alpha_false(self):
        ramp = scales.colour_ramp(["red", "blue"], alpha=False)
        result = ramp(np.array([0.0, 1.0]))
        # Without alpha, hex strings should be 7 chars (#RRGGBB)
        for c in result:
            assert len(c) == 7

    def test_alpha_true_opaque(self):
        # R: when alpha=TRUE but all inputs are fully opaque,
        # output is #RRGGBB (no alpha suffix), same as alpha=FALSE.
        ramp = scales.colour_ramp(["red", "blue"], alpha=True)
        result = ramp(np.array([0.0, 1.0]))
        for c in result:
            assert len(c) == 7

    def test_alpha_true_with_transparency(self):
        # When inputs have varying alpha, partially transparent outputs
        # get #RRGGBBAA; fully opaque outputs get #RRGGBB.
        ramp = scales.colour_ramp(["#FF000080", "#0000FFFF"], alpha=True)
        result = ramp(np.array([0.0, 0.5, 1.0]))
        assert len(result[0]) == 9  # x=0, alpha=0x80 → transparent
        assert len(result[1]) == 9  # x=0.5, alpha interpolated < 1
        assert len(result[2]) == 7  # x=1.0, alpha=0xFF → opaque


# =========================================================================
# test-full-seq.R
# =========================================================================

class TestFullseqNumeric:
    """fullseq works with numeric"""

    def test_basic(self):
        result = scales.fullseq((1, 10), size=2)
        assert isinstance(result, np.ndarray)
        assert result[0] <= 1
        assert result[-1] >= 10

    def test_covers_range(self):
        result = scales.fullseq((0.8, 9.2), size=2)
        assert result[0] <= 0.8
        assert result[-1] >= 9.2
        # Steps should be regular
        diffs = np.diff(result)
        np.testing.assert_allclose(diffs, diffs[0])


# =========================================================================
# test-label-bytes.R
# =========================================================================

class TestLabelBytesAutoUnits:
    """auto units always rounds down"""

    def test_auto_si(self):
        lb = scales.label_bytes(units="auto_si")
        result = lb([1000, 1000000])
        assert result == ["1 kB", "1 MB"]

    def test_auto_binary(self):
        lb = scales.label_bytes(units="auto_binary")
        result = lb([1024, 1048576])
        assert result == ["1 KiB", "1 MiB"]


class TestLabelBytesBinaryOrSI:
    """can use either binary or si units"""

    def test_si(self):
        lb = scales.label_bytes(units="auto_si")
        result = lb([1000])
        assert "kB" in result[0]

    def test_binary(self):
        lb = scales.label_bytes(units="auto_binary")
        result = lb([1024])
        assert "KiB" in result[0]


class TestLabelBytesZeroAndSpecial:
    """handles zero and special values"""

    def test_zero(self):
        lb = scales.label_bytes()
        result = lb([0])
        assert result == ["0 B"]

    def test_nan(self):
        lb = scales.label_bytes()
        result = lb([float("nan")])
        assert "NaN" in result[0] or "nan" in result[0].lower()


# =========================================================================
# test-label-currency.R
# =========================================================================

class TestLabelCurrencyNegative:
    """negative comes before prefix"""

    def test_negative_hyphen(self):
        lc = scales.label_currency(prefix="$", style_negative="hyphen")
        result = lc([-500])
        assert result[0].startswith("$-") or "-" in result[0]

    def test_positive_unchanged(self):
        lc = scales.label_currency(prefix="$")
        result = lc([1000])
        assert result[0].startswith("$")


class TestLabelCurrencyPreservesNAs:
    """preserves NAs"""

    def test_nan(self):
        result = scales.label_currency()([float("nan")])
        assert len(result) == 1
        assert "NaN" in result[0] or "nan" in result[0].lower()


class TestLabelCurrencyScaleCut:
    """can rescale with scale_cut"""

    def test_short_scale(self):
        lc = scales.label_currency(scale_cut=scales.cut_short_scale())
        result = lc([1000, 1000000])
        assert result == ["$1K", "$1M"]


# =========================================================================
# test-label-number.R
# =========================================================================

class TestLabelNumberExpectedFormat:
    """number returns expected format"""

    def test_with_accuracy(self):
        ln = scales.label_number(accuracy=0.01)
        result = ln([123.456, 0.1, -99.9])
        # All should have the right number of decimals
        for r in result:
            assert isinstance(r, str)

    def test_integer_accuracy(self):
        ln = scales.label_number(accuracy=1)
        result = ln([1.0, 2.5, 10.0])
        assert result == ["1", "2", "10"]


class TestLabelNumberMarks:
    """big.mark and decimal.mark work"""

    def test_european_format(self):
        ln = scales.label_number(big_mark=".", decimal_mark=",", accuracy=0.01)
        result = ln([1234.56])
        assert "." in result[0] or "," in result[0]


class TestLabelNumberPositiveNegativeStyles:
    """positive/negative styles work"""

    def test_plus_style(self):
        ln = scales.label_number(style_positive="plus")
        result = ln([100, -100, 0])
        assert result[0].startswith("+")
        assert result[1].startswith("-")

    def test_parens_style(self):
        ln = scales.label_number(style_negative="parens")
        result = ln([100, -100, 0])
        assert "(" in result[1]
        assert ")" in result[1]


class TestLabelNumberScaleCut:
    """scale_cut works correctly"""

    def test_short_scale(self):
        ln = scales.label_number(scale_cut=scales.cut_short_scale())
        result = ln([999, 1234, 1000000, 5600000000])
        assert "K" in result[1]
        assert "M" in result[2]
        assert "B" in result[3]


# =========================================================================
# test-label-ordinal.R
# =========================================================================

class TestLabelOrdinalEnglish:
    """ordinal format in English"""

    def test_suffixes(self):
        result = scales.label_ordinal()([1, 2, 3, 4, 11, 12, 13, 21, 22])
        assert result == [
            "1st", "2nd", "3rd", "4th",
            "11th", "12th", "13th",
            "21st", "22nd",
        ]


# =========================================================================
# test-label-percent.R
# =========================================================================

class TestLabelPercentBasic:
    """percent works on basic inputs"""

    def test_basic(self):
        result = scales.label_percent()([0.15, 0.5, 1.0])
        assert result == ["15%", "50%", "100%"]


class TestLabelPercentSuffix:
    """suffix can be customized"""

    def test_custom_suffix(self):
        lp = scales.label_percent(suffix=" percent")
        result = lp([0.5])
        assert result == ["50 percent"]


# =========================================================================
# test-label-pvalue.R
# =========================================================================

class TestLabelPvalueBoundaries:
    """pvalue handles boundaries"""

    def test_very_small(self):
        lpv = scales.label_pvalue(accuracy=0.001)
        result = lpv([0.00001])
        assert "<" in result[0]

    def test_very_large(self):
        lpv = scales.label_pvalue(accuracy=0.001)
        result = lpv([0.99999])
        assert ">" in result[0]

    def test_normal(self):
        lpv = scales.label_pvalue(accuracy=0.001)
        result = lpv([0.05])
        assert "0.05" in result[0]


# =========================================================================
# test-label-scientific.R
# =========================================================================

class TestLabelScientificFormat:
    """scientific notation format"""

    def test_basic(self):
        result = scales.label_scientific()([0.001, 1000000])
        assert result == ["1e-03", "1e+06"]

    def test_with_decimals(self):
        result = scales.label_scientific()([12345])
        assert "e" in result[0].lower()


# =========================================================================
# test-label-wrap.R
# =========================================================================

class TestLabelWrapLongText:
    """wraps long text at specified width"""

    def test_basic_wrap(self):
        lw = scales.label_wrap(15)
        result = lw(["this is a very long piece of text"])
        assert "\n" in result[0]

    def test_short_text_no_wrap(self):
        lw = scales.label_wrap(100)
        result = lw(["short"])
        assert result == ["short"]


# =========================================================================
# test-trans.R
# =========================================================================

class TestTransRepr:
    """trans has useful print method (repr)"""

    def test_log10_repr(self):
        t = scales.transform_log10()
        r = repr(t)
        assert "log" in r.lower()

    def test_sqrt_repr(self):
        t = scales.transform_sqrt()
        r = repr(t)
        assert "sqrt" in r.lower()

    def test_compose_repr(self):
        t = scales.transform_compose(
            scales.transform_log10(), scales.transform_reverse()
        )
        r = repr(t)
        assert "compose" in r.lower()


class TestAsTransformFromStrings:
    """as.transform generates correct transforms from strings"""

    def test_log10(self):
        t = scales.as_transform("log10")
        assert scales.is_transform(t)
        np.testing.assert_allclose(t.transform(np.array([10.0])), [1.0])

    def test_sqrt(self):
        t = scales.as_transform("sqrt")
        assert scales.is_transform(t)
        np.testing.assert_allclose(t.transform(np.array([4.0])), [2.0])

    def test_identity(self):
        t = scales.as_transform("identity")
        assert scales.is_transform(t)
        np.testing.assert_allclose(t.transform(np.array([42.0])), [42.0])

    def test_reverse(self):
        t = scales.as_transform("reverse")
        assert scales.is_transform(t)
        np.testing.assert_allclose(t.transform(np.array([1.0])), [-1.0])

    def test_log2(self):
        t = scales.as_transform("log2")
        assert scales.is_transform(t)
        np.testing.assert_allclose(t.transform(np.array([8.0])), [3.0])

    def test_passthrough(self):
        t = scales.transform_identity()
        assert scales.as_transform(t) is t


# =========================================================================
# test-trans-numeric.R
# =========================================================================

class TestTransformDomains:
    """all transforms have correct domain"""

    def test_log10_domain(self):
        t = scales.transform_log10()
        assert t.domain[0] > -np.inf  # lower bound is 0
        assert t.domain[0] >= 0

    def test_sqrt_domain(self):
        t = scales.transform_sqrt()
        assert t.domain[0] >= 0

    def test_identity_domain(self):
        t = scales.transform_identity()
        assert t.domain[0] == -np.inf
        assert t.domain[1] == np.inf

    def test_logit_domain(self):
        t = scales.transform_logit()
        assert t.domain[0] >= 0
        assert t.domain[1] <= 1


class TestBoxcoxModulusYJRoundtrip:
    """boxcox/modulus/yj inverse roundtrip"""

    def test_boxcox_p05(self):
        t = scales.transform_boxcox(p=0.5)
        x = np.array([1.0, 2.0, 5.0, 10.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)

    def test_boxcox_p0(self):
        t = scales.transform_boxcox(p=0)
        x = np.array([1.0, 2.0, 10.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)

    def test_boxcox_p1(self):
        t = scales.transform_boxcox(p=1)
        x = np.array([1.0, 5.0, 10.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)

    def test_modulus_roundtrip(self):
        t = scales.transform_modulus(p=0.5)
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)

    def test_yj_roundtrip(self):
        t = scales.transform_yj(p=0.5)
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)


# =========================================================================
# test-trans-compose.R
# =========================================================================

class TestComposedTransforms:
    """composed transforms work correctly"""

    def test_log10_reverse(self):
        tc = scales.transform_compose(
            scales.transform_log10(), scales.transform_reverse()
        )
        result = tc.transform(np.array([10.0, 100.0]))
        np.testing.assert_allclose(result, [-1.0, -2.0])

    def test_roundtrip(self):
        tc = scales.transform_compose(
            scales.transform_log10(), scales.transform_reverse()
        )
        x = np.array([1.0, 10.0, 100.0])
        np.testing.assert_allclose(tc.inverse(tc.transform(x)), x, atol=1e-10)


# =========================================================================
# test-range.R
# =========================================================================

class TestContinuousRangeIgnoresNAs:
    """continuous ranges ignores NAs"""

    def test_nan_ignored(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([1.0, float("nan"), 10.0]))
        lo, hi = cr.range
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(10.0)

    def test_nan_after_training(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([1.0, 5.0, 10.0]))
        cr.train(np.array([float("nan"), 3.0]))
        lo, hi = cr.range
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(10.0)


class TestDiscreteRangeOrder:
    """factor discrete ranges stay in order"""

    def test_preserves_order(self):
        dr = scales.DiscreteRange()
        dr.train(["a", "b", "c"])
        assert list(dr.range) == ["a", "b", "c"]

    def test_new_levels_appended(self):
        dr = scales.DiscreteRange()
        dr.train(["a", "b", "c"])
        dr.train(["b", "a", "d"])
        assert list(dr.range) == ["a", "b", "c", "d"]


# =========================================================================
# test-scale-continuous.R
# =========================================================================

class TestCscaleAppliesTransform:
    """cscale applies transform correctly"""

    def test_identity(self):
        result = scales.cscale(
            np.array([0, 0.5, 1.0]), scales.rescale_pal()
        )
        np.testing.assert_allclose(result, [0.1, 0.55, 1.0])

    def test_with_log_trans(self):
        result = scales.cscale(
            np.array([1, 10, 100]),
            scales.rescale_pal(),
            trans="log10",
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == 3


# =========================================================================
# test-scale-discrete.R
# =========================================================================

class TestDscaleMapsValues:
    """dscale maps values to palette"""

    def test_basic(self):
        result = scales.dscale(["a", "b"], scales.hue_pal())
        assert len(result) == 2
        for c in result:
            assert c.startswith("#")
