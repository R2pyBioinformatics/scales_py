"""Tests for scales label functions."""

import numpy as np
import pytest
from datetime import datetime

import scales


# ---------------------------------------------------------------------------
# label_number
# ---------------------------------------------------------------------------

class TestLabelNumber:
    def test_basic(self):
        # Intentional divergence from R: Python keeps no thousands
        # separator by default so labels round-trip through float().
        # Users wanting R's " " space must pass big_mark=" ".
        result = scales.label_number()([1234.5])
        assert result == ["1234"]

    def test_r_style_big_mark(self):
        result = scales.label_number(big_mark=" ")([1234.5])
        assert result == ["1 234"]

    def test_with_accuracy(self):
        result = scales.label_number(accuracy=0.1)([1234.5])
        assert result == ["1234.5"]

    def test_zero(self):
        result = scales.label_number()([0])
        assert result == ["0"]

    def test_empty(self):
        result = scales.label_number()([])
        assert result == []

    def test_nan(self):
        result = scales.label_number()([np.nan])
        assert len(result) == 1
        assert "NaN" in result[0] or "nan" in result[0].lower()

    def test_multiple(self):
        result = scales.label_number()([1, 2, 3])
        assert len(result) == 3


# ---------------------------------------------------------------------------
# label_comma
# ---------------------------------------------------------------------------

class TestLabelComma:
    def test_basic(self):
        result = scales.label_comma()([1234567])
        assert result == ["1,234,567"]

    def test_small_number(self):
        result = scales.label_comma()([999])
        assert result == ["999"]

    def test_zero(self):
        result = scales.label_comma()([0])
        assert result == ["0"]

    def test_multiple(self):
        result = scales.label_comma()([1000, 2000000])
        assert result == ["1,000", "2,000,000"]


# ---------------------------------------------------------------------------
# label_percent
# ---------------------------------------------------------------------------

class TestLabelPercent:
    def test_basic(self):
        result = scales.label_percent()([0.15, 0.5, 1.0])
        assert result == ["15%", "50%", "100%"]

    def test_zero(self):
        result = scales.label_percent()([0.0])
        assert result == ["0%"]

    def test_small_fraction(self):
        result = scales.label_percent()([0.001])
        assert len(result) == 1
        assert "%" in result[0]


# ---------------------------------------------------------------------------
# label_dollar
# ---------------------------------------------------------------------------

class TestLabelDollar:
    def test_basic(self):
        result = scales.label_dollar()([1.5, 1000])
        assert len(result) == 2
        for r in result:
            assert r.startswith("$")

    def test_with_accuracy(self):
        result = scales.label_dollar(accuracy=0.01)([1.5, 1000])
        assert "$1.5" in result[0] or "$1.50" in result[0]
        assert "$1000" in result[1] or "$1,000" in result[1]

    def test_zero(self):
        result = scales.label_dollar()([0])
        assert result == ["$0"]


# ---------------------------------------------------------------------------
# label_scientific
# ---------------------------------------------------------------------------

class TestLabelScientific:
    def test_basic(self):
        result = scales.label_scientific()([0.001, 1000000])
        assert len(result) == 2
        assert "e" in result[0].lower() or "E" in result[0]
        assert "e" in result[1].lower() or "E" in result[1]

    def test_known_values(self):
        result = scales.label_scientific()([0.001, 1000000])
        assert result == ["1e-03", "1e+06"]


# ---------------------------------------------------------------------------
# label_bytes
# ---------------------------------------------------------------------------

class TestLabelBytes:
    def test_basic(self):
        result = scales.label_bytes()([1024, 1048576])
        assert len(result) == 2
        assert "kB" in result[0] or "KB" in result[0]
        assert "MB" in result[1]

    def test_known_values(self):
        result = scales.label_bytes()([1024, 1048576])
        assert result == ["1 kB", "1 MB"]


# ---------------------------------------------------------------------------
# label_ordinal
# ---------------------------------------------------------------------------

class TestLabelOrdinal:
    def test_basic(self):
        result = scales.label_ordinal()([1, 2, 3, 4, 11, 12, 13, 21])
        assert result == ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st"]

    def test_single(self):
        result = scales.label_ordinal()([1])
        assert result == ["1st"]


# ---------------------------------------------------------------------------
# label_pvalue
# ---------------------------------------------------------------------------

class TestLabelPvalue:
    def test_basic(self):
        result = scales.label_pvalue()([0.0001, 0.05, 0.5, 0.999])
        assert len(result) == 4
        assert "<" in result[0]  # very small p-value
        assert "0.05" in result[1]

    def test_known_values(self):
        result = scales.label_pvalue()([0.0001, 0.05, 0.5, 0.999])
        assert result == ["<0.001", "0.0500", "0.5000", "0.9990"]


# ---------------------------------------------------------------------------
# label_date
# ---------------------------------------------------------------------------

class TestLabelDate:
    def test_basic(self):
        dates = [datetime(2023, 1, 15), datetime(2023, 6, 30)]
        result = scales.label_date()(dates)
        assert len(result) == 2
        assert "2023" in result[0]
        assert "2023" in result[1]

    def test_known_format(self):
        dates = [datetime(2023, 1, 15)]
        result = scales.label_date()(dates)
        assert result == ["2023-01-15"]


# ---------------------------------------------------------------------------
# label_wrap
# ---------------------------------------------------------------------------

class TestLabelWrap:
    def test_basic(self):
        result = scales.label_wrap(10)(["this is a long string that should wrap"])
        assert len(result) == 1
        assert "\n" in result[0]

    def test_short_string_no_wrap(self):
        result = scales.label_wrap(100)(["short"])
        assert result == ["short"]


# ---------------------------------------------------------------------------
# label_log
# ---------------------------------------------------------------------------

class TestLabelLog:
    def test_basic(self):
        # Mirrors R's format_log / label_log: 1 -> "10^0", not "1".
        result = scales.label_log(10)([1, 100, 1000])
        assert result == ["10^0", "10^2", "10^3"]

    def test_known_values(self):
        result = scales.label_log(10)([1, 100, 1000])
        assert result == ["10^0", "10^2", "10^3"]


# ---------------------------------------------------------------------------
# label_dictionary
# ---------------------------------------------------------------------------

class TestLabelDictionary:
    def test_basic(self):
        labeller = scales.label_dictionary({"a": "Alpha", "b": "Beta"})
        result = labeller(["a", "b"])
        assert result == ["Alpha", "Beta"]

    def test_missing_key(self):
        labeller = scales.label_dictionary({"a": "Alpha"})
        result = labeller(["a", "c"])
        assert result[0] == "Alpha"
        # Missing key should return key itself or some fallback
        assert len(result) == 2


# ---------------------------------------------------------------------------
# compose_label
# ---------------------------------------------------------------------------

class TestComposeLabel:
    def test_basic(self):
        composed = scales.compose_label(
            scales.label_number(),
            lambda x: [s + "!" for s in x],
        )
        result = composed([1234.5])
        assert result == ["1234!"]

    def test_two_label_functions(self):
        # label_dollar() defaults to R's currency_big_mark=",", so the
        # second formatter gets "1234" and reformats it as "$1,234".
        composed = scales.compose_label(
            scales.label_number(accuracy=1),
            scales.label_dollar(),
        )
        result = composed([1234.5])
        assert result == ["$1,234"]


# ---------------------------------------------------------------------------
# Direct convenience functions
# ---------------------------------------------------------------------------

class TestDirectFunctions:
    def test_number(self):
        result = scales.number([1234.5])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_comma(self):
        result = scales.comma([1234567])
        assert result == ["1,234,567"]

    def test_dollar(self):
        result = scales.dollar([1.5, 1000])
        for r in result:
            assert r.startswith("$")

    def test_percent(self):
        result = scales.percent([0.15, 0.5])
        assert result == ["15%", "50%"]


# ---------------------------------------------------------------------------
# cut_short_scale
# ---------------------------------------------------------------------------

class TestCutShortScale:
    def test_returns_list(self):
        result = scales.cut_short_scale()
        assert isinstance(result, list)

    def test_values(self):
        result = scales.cut_short_scale()
        # Convert to dict for easier checking
        d = dict(result)
        assert d[0] == ""
        assert d[1000.0] == "K"
        assert d[1000000.0] == "M"
        assert d[1000000000.0] == "B"
        assert d[1000000000000.0] == "T"

    def test_length(self):
        result = scales.cut_short_scale()
        assert len(result) == 5


# ---------------------------------------------------------------------------
# cut_si
# ---------------------------------------------------------------------------

class TestCutSI:
    def test_basic(self):
        result = scales.cut_si("g")
        assert isinstance(result, list)
        assert len(result) > 5

    def test_contains_base_unit(self):
        result = scales.cut_si("g")
        d = dict(result)
        assert " g" in d.values()
        assert " kg" in d.values()
        assert " mg" in d.values()


# ---------------------------------------------------------------------------
# ordinal_english / ordinal_french / ordinal_spanish
# ---------------------------------------------------------------------------

class TestOrdinalEnglish:
    def test_returns_callable(self):
        fn = scales.ordinal_english()
        assert callable(fn)

    def test_suffixes(self):
        fn = scales.ordinal_english()
        assert fn(1) == "st"
        assert fn(2) == "nd"
        assert fn(3) == "rd"
        assert fn(4) == "th"
        assert fn(11) == "th"
        assert fn(12) == "th"
        assert fn(13) == "th"
        assert fn(21) == "st"
        assert fn(22) == "nd"
        assert fn(23) == "rd"
        assert fn(100) == "th"
        assert fn(111) == "th"
        assert fn(112) == "th"
        assert fn(113) == "th"


class TestOrdinalFrench:
    def test_returns_callable(self):
        fn = scales.ordinal_french()
        assert callable(fn)

    def test_suffixes(self):
        fn = scales.ordinal_french()
        assert fn(1) == "er"
        assert fn(2) == "e"
        assert fn(10) == "e"


class TestOrdinalSpanish:
    def test_returns_callable(self):
        fn = scales.ordinal_spanish()
        assert callable(fn)

    def test_suffixes(self):
        fn = scales.ordinal_spanish()
        # Spanish ordinals all use .o or similar
        assert fn(1) is not None
        assert fn(2) is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestLabelEdgeCases:
    def test_nan_input_number(self):
        result = scales.label_number()([np.nan, 1.0])
        assert len(result) == 2

    def test_empty_input(self):
        assert scales.label_number()([]) == []
        assert scales.label_comma()([]) == []
        assert scales.label_percent()([]) == []

    def test_zero_input(self):
        assert scales.label_number()([0]) == ["0"]
        assert scales.label_comma()([0]) == ["0"]
        assert scales.label_percent()([0]) == ["0%"]

    def test_negative_numbers(self):
        result = scales.label_number()([-100])
        assert len(result) == 1
        assert "-" in result[0]

    def test_large_numbers_comma(self):
        result = scales.label_comma()([1000000000])
        assert result == ["1,000,000,000"]
