"""Coverage tests for scales/labels.py – targeting uncovered lines."""

import warnings
import numpy as np
import pytest
from datetime import datetime, timezone

from scales.labels import (
    # Helpers / internals
    _format_number,
    _apply_style,
    _apply_scale_cut,
    _format_scientific_single,
    _to_datetime,
    _make_tz,
    # Direct formatters
    number,
    comma,
    dollar,
    percent,
    scientific,
    ordinal,
    pvalue,
    format_log,
    # Closure factories
    label_number,
    label_comma,
    label_percent,
    label_dollar,
    label_currency,
    label_scientific,
    label_bytes,
    label_ordinal,
    label_pvalue,
    label_date,
    label_date_short,
    label_time,
    label_timespan,
    label_wrap,
    label_glue,
    label_parse,
    label_math,
    label_log,
    label_number_auto,
    label_number_si,
    label_dictionary,
    compose_label,
    # Ordinal helpers
    ordinal_english,
    ordinal_french,
    ordinal_spanish,
    # Scale-cut helpers
    cut_short_scale,
    cut_long_scale,
    cut_time_scale,
    cut_si,
    # Date utilities
    date_breaks,
    date_format,
    time_format,
    # Legacy aliases
    comma_format,
    dollar_format,
    percent_format,
    scientific_format,
    ordinal_format,
    pvalue_format,
    number_format,
    number_bytes_format,
    number_bytes,
    parse_format,
    math_format,
    wrap_format,
    format_format,
    unit_format,
    number_options,
)


# ---------------------------------------------------------------------------
# _format_number edge cases (lines 113, 137-138)
# ---------------------------------------------------------------------------

class TestFormatNumberEdge:
    def test_negative_inf(self):
        assert _format_number(float("-inf"), 1, None, ".", True) == "-Inf"

    def test_positive_inf(self):
        assert _format_number(float("inf"), 1, None, ".", True) == "Inf"

    def test_nan(self):
        assert _format_number(float("nan"), 1, None, ".", True) == "NaN"

    def test_big_mark_negative_large(self):
        # Lines 136-138: negative number with big_mark
        result = _format_number(-1234567.0, 1, ",", ".", True)
        assert result.startswith("-")
        assert "," in result

    def test_big_mark_small(self):
        # Number < 1000, big_mark should still work
        result = _format_number(999.0, 1, ",", ".", True)
        assert result == "999"


# ---------------------------------------------------------------------------
# _apply_style (lines 170, 176)
# ---------------------------------------------------------------------------

class TestApplyStyle:
    def test_style_positive_space(self):
        result = _apply_style("5", 5.0, "space", "hyphen")
        assert result == " 5"

    def test_style_negative_minus(self):
        result = _apply_style("-5", -5.0, "none", "minus")
        assert "\u2212" in result

    def test_style_negative_parens(self):
        result = _apply_style("-5", -5.0, "none", "parens")
        assert result == "(5)"

    def test_nan_passthrough(self):
        result = _apply_style("NaN", float("nan"), "plus", "parens")
        assert result == "NaN"


# ---------------------------------------------------------------------------
# _apply_scale_cut (line 259-260)
# ---------------------------------------------------------------------------

class TestApplyScaleCut:
    def test_basic_short_scale(self):
        vals = np.array([500, 1500, 1e6, 2e9])
        sc = cut_short_scale()
        scaled, suffixes = _apply_scale_cut(vals, sc)
        assert suffixes[0] == ""
        assert suffixes[1] == "K"
        assert suffixes[2] == "M"
        assert suffixes[3] == "B"


# ---------------------------------------------------------------------------
# cut_long_scale (lines 259-260)
# ---------------------------------------------------------------------------

class TestCutLongScale:
    def test_space_option(self):
        result = cut_long_scale(space=True)
        assert any(" K" in s for _, s in result)

    def test_no_space(self):
        result = cut_long_scale(space=False)
        assert any("K" in s and " K" not in s for _, s in result)


# ---------------------------------------------------------------------------
# cut_time_scale (lines 285-286)
# ---------------------------------------------------------------------------

class TestCutTimeScale:
    def test_basic(self):
        result = cut_time_scale()
        assert len(result) > 0

    def test_with_space(self):
        result = cut_time_scale(space=True)
        assert any(" s" in s for _, s in result)


# ---------------------------------------------------------------------------
# _format_scientific_single edge cases (lines 637-644, 659)
# ---------------------------------------------------------------------------

class TestFormatScientificSingle:
    def test_nan(self):
        assert _format_scientific_single(float("nan"), 3, ".", True) == "NaN"

    def test_neg_inf(self):
        assert _format_scientific_single(float("-inf"), 3, ".", True) == "-Inf"

    def test_pos_inf(self):
        assert _format_scientific_single(float("inf"), 3, ".", True) == "Inf"

    def test_zero_trim(self):
        result = _format_scientific_single(0, 3, ".", True)
        assert result == "0"

    def test_zero_no_trim(self):
        result = _format_scientific_single(0, 3, ".", False)
        assert "e" in result

    def test_decimal_mark_comma(self):
        result = _format_scientific_single(1234.5, 3, ",", True)
        assert "," in result


# ---------------------------------------------------------------------------
# label_bytes edge cases (lines 818-827)
# ---------------------------------------------------------------------------

class TestLabelBytesEdge:
    def test_explicit_si_unit(self):
        fmt = label_bytes(units="kB")
        result = fmt([1000, 2000])
        assert all("kB" in r for r in result)

    def test_explicit_binary_unit(self):
        fmt = label_bytes(units="MiB")
        result = fmt([1048576, 2097152])
        assert all("MiB" in r for r in result)

    def test_unknown_unit(self):
        fmt = label_bytes(units="custom")
        result = fmt([100])
        assert "custom" in result[0]

    def test_auto_binary(self):
        fmt = label_bytes(units="auto_binary")
        result = fmt([1024, 1048576])
        assert "KiB" in result[0]
        assert "MiB" in result[1]

    def test_non_finite(self):
        fmt = label_bytes()
        result = fmt([float("nan"), float("inf"), float("-inf")])
        assert "NaN" in result[0]
        assert "Inf" in result[1]
        assert "-Inf" in result[2]


# ---------------------------------------------------------------------------
# ordinal edge cases (lines 897, 963-964)
# ---------------------------------------------------------------------------

class TestOrdinalEdge:
    def test_non_finite(self):
        result = ordinal([float("nan"), float("inf"), float("-inf")])
        assert "NaN" in result[0]
        assert "Inf" in result[1]
        assert "-Inf" in result[2]

    def test_french_feminin(self):
        fn = ordinal_french(gender="feminin")
        assert fn(1) == "re"

    def test_french_feminin_plural(self):
        fn = ordinal_french(gender="feminin", plural=True)
        assert fn(1) == "res"

    def test_french_masculin_plural(self):
        fn = ordinal_french(gender="masculin", plural=True)
        assert fn(1) == "ers"

    def test_french_not_first(self):
        fn = ordinal_french()
        assert fn(2) == "e"

    def test_french_plural_not_first(self):
        fn = ordinal_french(plural=True)
        assert fn(2) == "es"

    def test_spanish(self):
        fn = ordinal_spanish()
        assert fn(1) == ".\u00ba"

    def test_ordinal_with_big_mark(self):
        result = ordinal([1000], big_mark=",")
        assert "," in result[0]


# ---------------------------------------------------------------------------
# pvalue edge cases (lines 1042, 1052-1053, 1066)
# ---------------------------------------------------------------------------

class TestPvalueEdge:
    def test_greater_than_threshold(self):
        result = pvalue([0.9999], accuracy=0.001)
        assert ">" in result[0]

    def test_add_p_less_than(self):
        result = pvalue([0.0001], accuracy=0.001, add_p=True)
        assert result[0].startswith("p<")

    def test_add_p_greater_than(self):
        result = pvalue([0.9999], accuracy=0.001, add_p=True)
        assert result[0].startswith("p>")

    def test_add_p_normal(self):
        # R uses "p=" (no spaces) as the middle prefix when add_p=TRUE.
        result = pvalue([0.5], accuracy=0.001, add_p=True)
        assert result[0].startswith("p=")

    def test_non_finite(self):
        result = pvalue([float("nan"), float("inf")])
        assert "NaN" in result[0]

    def test_custom_prefix(self):
        result = pvalue([0.0001], accuracy=0.001, prefix=["<<", "~", ">>"])
        assert "<<" in result[0]


# ---------------------------------------------------------------------------
# _to_datetime (lines 1115-1123)
# ---------------------------------------------------------------------------

class TestToDatetime:
    def test_datetime_with_tz(self):
        dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
        result = _to_datetime(dt, timezone.utc)
        assert result is not None
        assert result.year == 2020

    def test_datetime_without_tz(self):
        dt = datetime(2020, 1, 1)
        result = _to_datetime(dt, timezone.utc)
        assert result is not None

    def test_numpy_datetime64(self):
        dt = np.datetime64("2020-01-01T00:00:00")
        result = _to_datetime(dt, timezone.utc)
        assert result is not None
        assert result.year == 2020

    def test_numeric_timestamp(self):
        result = _to_datetime(0.0, timezone.utc)
        assert result is not None
        assert result.year == 1970

    def test_non_finite(self):
        result = _to_datetime(float("nan"), timezone.utc)
        assert result is None

    def test_unsupported_type(self):
        result = _to_datetime("not-a-date", timezone.utc)
        assert result is None


# ---------------------------------------------------------------------------
# _make_tz (lines 1131-1138)
# ---------------------------------------------------------------------------

class TestMakeTz:
    def test_utc(self):
        tz = _make_tz("UTC")
        assert tz == timezone.utc

    def test_offset_positive(self):
        tz = _make_tz("UTC+05:30")
        assert tz is not None

    def test_offset_negative(self):
        tz = _make_tz("UTC-08")
        assert tz is not None

    def test_fallback(self):
        tz = _make_tz("America/New_York")
        # Falls back to UTC
        assert tz == timezone.utc


# ---------------------------------------------------------------------------
# label_date_short (lines 1196-1227)
# ---------------------------------------------------------------------------

class TestLabelDateShort:
    def test_basic(self):
        fmt = label_date_short()
        ts1 = datetime(2020, 1, 15, tzinfo=timezone.utc).timestamp()
        ts2 = datetime(2020, 2, 20, tzinfo=timezone.utc).timestamp()
        ts3 = datetime(2021, 3, 25, tzinfo=timezone.utc).timestamp()
        result = fmt([ts1, ts2, ts3])
        assert len(result) == 3
        # First should include year
        assert "2020" in result[0]

    def test_non_finite(self):
        fmt = label_date_short()
        result = fmt([float("nan")])
        assert result[0] == "NA"


# ---------------------------------------------------------------------------
# label_time (lines 1249-1262)
# ---------------------------------------------------------------------------

class TestLabelTime:
    def test_basic(self):
        fmt = label_time()
        ts = datetime(2020, 1, 1, 12, 30, 45, tzinfo=timezone.utc).timestamp()
        result = fmt([ts])
        assert "12" in result[0]

    def test_na(self):
        fmt = label_time()
        result = fmt([float("nan")])
        assert result[0] == "NA"

    def test_with_format(self):
        fmt = label_time(format="%H:%M")
        ts = datetime(2020, 1, 1, 12, 30, tzinfo=timezone.utc).timestamp()
        result = fmt([ts])
        assert "12:30" in result[0]


# ---------------------------------------------------------------------------
# label_timespan (lines 1285-1339)
# ---------------------------------------------------------------------------

class TestLabelTimespan:
    def test_basic_seconds(self):
        fmt = label_timespan(unit="secs")
        result = fmt([0, 1, 60, 3600, 86400, 604800])
        assert "0s" in result[0]
        assert "s" in result[1]
        assert "m" in result[2]
        assert "h" in result[3]
        assert "d" in result[4]
        assert "w" in result[5]

    def test_space(self):
        fmt = label_timespan(unit="secs", space=True)
        result = fmt([60])
        assert " m" in result[0]

    def test_non_finite(self):
        fmt = label_timespan()
        result = fmt([float("nan"), float("inf")])
        assert "NaN" in result[0]

    def test_sub_second(self):
        # R uses the Unicode Greek mu "\u03bcs" for microseconds when
        # UTF-8 is available; Python matches unconditionally.
        fmt = label_timespan()
        result = fmt([0.001, 0.000001])
        assert "ms" in result[0]
        assert "\u03bcs" in result[1]

    def test_minutes_unit(self):
        fmt = label_timespan(unit="mins")
        result = fmt([1])  # 1 minute = 60 seconds
        assert "m" in result[0] or "s" in result[0]


# ---------------------------------------------------------------------------
# label_date with NA (line 1168)
# ---------------------------------------------------------------------------

class TestLabelDateNA:
    def test_na_value(self):
        fmt = label_date()
        result = fmt([float("nan")])
        assert result[0] == "NA"


# ---------------------------------------------------------------------------
# label_log (lines 1541-1545, 1551)
# ---------------------------------------------------------------------------

class TestLabelLog:
    def test_basic(self):
        fmt = label_log(base=10)
        result = fmt([1, 10, 100, 1000])
        # Mirrors R: log10(1)=0 -> "10^0", log10(10)=1 -> "10^1", ...
        assert result == ["10^0", "10^1", "10^2", "10^3"]

    def test_non_integer_exponent(self):
        fmt = label_log(base=10, digits=3)
        result = fmt([50])  # log10(50) ≈ 1.699
        assert "10^" in result[0]

    def test_non_finite(self):
        fmt = label_log()
        result = fmt([float("nan"), float("inf")])
        assert result[0] == "NaN"

    def test_zero_or_negative_switches_to_signed(self):
        # Per R: any(finite <= 0) triggers signed mode; zeros print as "0".
        fmt = label_log()
        result = fmt([0, -1, 1, 10])
        assert result[0] == "0"
        assert result[1] == "-10^0"
        assert result[2] == "+10^0"
        assert result[3] == "+10^1"


# ---------------------------------------------------------------------------
# format_log (R-parity semantics)
# ---------------------------------------------------------------------------

class TestFormatLog:
    def test_positive_values(self):
        # R: format_log(c(1, 10, 100)) -> c("10^0", "10^1", "10^2")
        result = format_log([1, 10, 100], base=10)
        assert result == ["10^0", "10^1", "10^2"]

    def test_zero_triggers_signed_and_becomes_zero(self):
        # R: any(finite <= 0) -> signed=TRUE; sign(0)=0 -> text "0".
        result = format_log([0, 1, 2, -1], base=10)
        assert result[0] == "0"
        assert result[1] == "+10^0"
        assert result[2].startswith("+10^")
        assert result[3] == "-10^0"

    def test_signed_true(self):
        result = format_log([1, 2], base=10, signed=True)
        assert result[0] == "+10^0"
        assert result[1].startswith("+10^")

    def test_non_integer_base(self):
        result = format_log([2.5], base=2.5)
        assert result[0] == "2.5^1"

    def test_nan_and_inf(self):
        result = format_log([float("nan"), float("inf")])
        assert result[0] == "NaN"

    def test_fractional_exponent(self):
        # log10(sqrt(10)) ≈ 0.5
        result = format_log([10 ** 0.5], base=10)
        assert result[0].startswith("10^0.5")


# ---------------------------------------------------------------------------
# label_number_auto (lines 1575-1589)
# ---------------------------------------------------------------------------

class TestLabelNumberAuto:
    def test_small_values(self):
        fmt = label_number_auto()
        result = fmt([0.0001, 0.0002])
        # Should use scientific notation for very small
        assert "e" in result[0] or "E" in result[0]

    def test_large_values(self):
        fmt = label_number_auto()
        result = fmt([1e9, 2e9])
        assert "e" in result[0] or "E" in result[0]

    def test_normal_values(self):
        fmt = label_number_auto()
        result = fmt([1, 2, 3])
        # Should use normal notation
        assert "e" not in result[0]

    def test_all_nan(self):
        fmt = label_number_auto()
        result = fmt([float("nan")])
        assert "NaN" in result[0]


# ---------------------------------------------------------------------------
# label_number_si (deprecated) (line 1664)
# ---------------------------------------------------------------------------

class TestLabelNumberSi:
    def test_deprecated_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fmt = label_number_si(unit="W")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


# ---------------------------------------------------------------------------
# label_dictionary (line 1674)
# ---------------------------------------------------------------------------

class TestLabelDictionary:
    def test_with_dict(self):
        fmt = label_dictionary(dictionary={1: "one", 2: "two"})
        result = fmt([1, 2])
        assert result == ["one", "two"]

    def test_nomatch(self):
        fmt = label_dictionary(dictionary={1: "one"}, nomatch="???")
        result = fmt([1, 2])
        assert result[1] == "???"

    def test_none_dict(self):
        fmt = label_dictionary()
        result = fmt([1, 2])
        assert result == ["1", "2"]


# ---------------------------------------------------------------------------
# compose_label (line 1745)
# ---------------------------------------------------------------------------

class TestComposeLabel:
    def test_compose_two(self):
        fn = compose_label(label_number(), label_parse())
        result = fn([1, 2])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# unit_format (line 1767-1768)
# ---------------------------------------------------------------------------

class TestUnitFormat:
    def test_basic(self):
        fmt = unit_format(unit="kg", scale=0.001)
        result = fmt([1000, 2000])
        assert "kg" in result[0]


# ---------------------------------------------------------------------------
# date_breaks / date_format / time_format (lines 1776, 1784)
# ---------------------------------------------------------------------------

class TestDateUtilities:
    def test_date_format(self):
        fmt = date_format()
        ts = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        result = fmt([ts])
        assert "2020" in result[0]

    def test_time_format(self):
        fmt = time_format()
        ts = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        result = fmt([ts])
        assert "12" in result[0]


# ---------------------------------------------------------------------------
# number_options (line 1448, 1452)
# ---------------------------------------------------------------------------

class TestNumberOptions:
    def test_basic(self):
        result = number_options()
        assert "decimal_mark" in result

    def test_custom(self):
        result = number_options(decimal_mark=",", big_mark=".")
        # number_options returns the global store after updating
        assert isinstance(result, dict)
        assert "decimal_mark" in result


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------

class TestLegacyAliases:
    def test_comma_format(self):
        assert comma_format is label_comma

    def test_dollar_format(self):
        assert dollar_format is label_dollar

    def test_percent_format(self):
        assert percent_format is label_percent

    def test_scientific_format(self):
        assert scientific_format is label_scientific

    def test_ordinal_format(self):
        assert ordinal_format is label_ordinal

    def test_pvalue_format(self):
        assert pvalue_format is label_pvalue

    def test_number_format(self):
        assert number_format is label_number

    def test_parse_format(self):
        assert parse_format is label_parse

    def test_math_format(self):
        assert math_format is label_math

    def test_wrap_format(self):
        assert wrap_format is label_wrap

    def test_format_format(self):
        assert format_format is label_glue

    def test_number_bytes(self):
        assert number_bytes is label_bytes

    def test_number_bytes_format(self):
        assert number_bytes_format is label_bytes


# ---------------------------------------------------------------------------
# label_math with format_func (line 1448, 1452)
# ---------------------------------------------------------------------------

class TestLabelMath:
    def test_with_format_func(self):
        fmt = label_math(expr="f({x})", format_func=label_number())
        result = fmt([1, 2])
        assert all("f(" in r for r in result)

    def test_without_expr(self):
        fmt = label_math()
        result = fmt([1, 2])
        assert result[0] in ("1", "1.0")


# ---------------------------------------------------------------------------
# label_wrap edge (line 1364)
# ---------------------------------------------------------------------------

class TestLabelWrap:
    def test_string_input(self):
        fmt = label_wrap(width=5)
        result = fmt("hello world")
        assert len(result) == 1
        assert "\n" in result[0]


# ---------------------------------------------------------------------------
# label_glue
# ---------------------------------------------------------------------------

class TestLabelGlue:
    def test_pattern(self):
        fmt = label_glue("{x:.2f} units")
        result = fmt([1.5, 2.3])
        assert result[0] == "1.50 units"
