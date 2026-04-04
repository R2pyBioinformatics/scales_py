"""Tests for the 14 items previously at 'implemented' status.

Covers demo_*, number_options, label_parse, label_math, format_format,
label_number_si, and show_col so they can be promoted to 'validated'.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import scales


# ---------------------------------------------------------------------------
# demo_* functions — they call matplotlib, so we test that they are
# callable and accept the documented signatures without error.
# We use a non-interactive backend to avoid popping windows.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mpl_agg():
    """Force non-interactive backend for all tests in this module."""
    import matplotlib
    backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    yield
    matplotlib.use(backend)


class TestDemoContinuous:
    def test_callable(self):
        assert callable(scales.demo_continuous)

    def test_basic_call(self):
        import matplotlib.pyplot as plt
        scales.demo_continuous(np.array([0, 100]))
        plt.close("all")

    def test_with_labels(self):
        import matplotlib.pyplot as plt
        scales.demo_continuous(
            np.array([0, 1000]),
            labels=scales.label_comma(),
        )
        plt.close("all")

    def test_with_breaks(self):
        import matplotlib.pyplot as plt
        scales.demo_continuous(
            np.array([0, 100]),
            breaks=scales.breaks_extended(n=5),
        )
        plt.close("all")


class TestDemoLog10:
    def test_callable(self):
        assert callable(scales.demo_log10)

    def test_basic_call(self):
        import matplotlib.pyplot as plt
        scales.demo_log10(np.array([1, 10000]))
        plt.close("all")


class TestDemoDiscrete:
    def test_callable(self):
        assert callable(scales.demo_discrete)

    def test_basic_call(self):
        import matplotlib.pyplot as plt
        scales.demo_discrete(["a", "b", "c"])
        plt.close("all")


class TestDemoDatetime:
    def test_callable(self):
        assert callable(scales.demo_datetime)

    def test_basic_call(self):
        import matplotlib.pyplot as plt
        from datetime import datetime
        scales.demo_datetime([datetime(2020, 1, 1), datetime(2020, 12, 31)])
        plt.close("all")


class TestDemoTime:
    def test_callable(self):
        assert callable(scales.demo_time)

    def test_basic_call(self):
        import matplotlib.pyplot as plt
        scales.demo_time(np.array([0, 3600]))
        plt.close("all")


class TestDemoTimespan:
    def test_callable(self):
        assert callable(scales.demo_timespan)

    def test_basic_call(self):
        import matplotlib.pyplot as plt
        scales.demo_timespan(np.array([0, 86400]))
        plt.close("all")


# ---------------------------------------------------------------------------
# number_options
# ---------------------------------------------------------------------------

class TestNumberOptions:
    def test_returns_previous(self):
        prev = scales.number_options()
        assert isinstance(prev, dict)
        assert "decimal_mark" in prev

    def test_sets_and_resets(self):
        prev = scales.number_options(decimal_mark=",", big_mark=".")
        from scales.labels import _NUMBER_OPTIONS
        assert _NUMBER_OPTIONS["decimal_mark"] == ","
        assert _NUMBER_OPTIONS["big_mark"] == "."
        # Reset
        scales.number_options()
        assert _NUMBER_OPTIONS["decimal_mark"] == "."

    def test_currency_defaults_inferred(self):
        scales.number_options(decimal_mark=",")
        from scales.labels import _NUMBER_OPTIONS
        assert _NUMBER_OPTIONS["currency_decimal_mark"] == ","
        assert _NUMBER_OPTIONS["currency_big_mark"] == "."
        scales.number_options()  # reset

    def test_all_style_options_accepted(self):
        scales.number_options(
            style_positive="plus",
            style_negative="parens",
        )
        from scales.labels import _NUMBER_OPTIONS
        assert _NUMBER_OPTIONS["style_positive"] == "plus"
        assert _NUMBER_OPTIONS["style_negative"] == "parens"
        scales.number_options()  # reset


# ---------------------------------------------------------------------------
# label_parse
# ---------------------------------------------------------------------------

class TestLabelParse:
    def test_returns_strings(self):
        lp = scales.label_parse()
        result = lp(["alpha", "beta", "gamma"])
        assert result == ["alpha", "beta", "gamma"]

    def test_numeric_input(self):
        lp = scales.label_parse()
        result = lp([1, 2, 3])
        assert all(isinstance(s, str) for s in result)

    def test_empty(self):
        lp = scales.label_parse()
        result = lp([])
        assert result == []


# ---------------------------------------------------------------------------
# label_math / math_format
# ---------------------------------------------------------------------------

class TestLabelMath:
    def test_returns_strings(self):
        lm = scales.label_math()
        result = lm([1, 10, 100])
        assert all(isinstance(s, str) for s in result)

    def test_math_format_alias(self):
        assert scales.math_format is scales.label_math

    def test_parse_format_alias(self):
        assert scales.parse_format is scales.label_parse


# ---------------------------------------------------------------------------
# format_format
# ---------------------------------------------------------------------------

class TestFormatFormat:
    def test_returns_callable(self):
        ff = scales.format_format()
        assert callable(ff)

    def test_formats_values(self):
        ff = scales.format_format()
        result = ff([1.5, 2.5])
        assert all(isinstance(s, str) for s in result)


# ---------------------------------------------------------------------------
# label_number_si (deprecated)
# ---------------------------------------------------------------------------

class TestLabelNumberSi:
    def test_deprecated_warning(self):
        with pytest.warns(DeprecationWarning, match="label_number_si"):
            lsi = scales.label_number_si(unit="g")
            assert callable(lsi)

    def test_formats_correctly(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            lsi = scales.label_number_si(unit="g")
            result = lsi(np.array([0.001, 1, 1000]))
            assert any("mg" in s or "m" in s for s in result)
            assert any("kg" in s or "k" in s for s in result)


# ---------------------------------------------------------------------------
# show_col
# ---------------------------------------------------------------------------

class TestShowCol:
    def test_callable(self):
        assert callable(scales.show_col)

    def test_basic_call(self):
        import matplotlib.pyplot as plt
        scales.show_col(["#FF0000", "#00FF00", "#0000FF"])
        plt.close("all")

    def test_with_labels_false(self):
        import matplotlib.pyplot as plt
        scales.show_col(["red", "blue"], labels=False)
        plt.close("all")
