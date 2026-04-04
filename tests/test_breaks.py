"""Comprehensive tests for scales.breaks module."""

import numpy as np
import pytest

import scales


# ---------------------------------------------------------------------------
# breaks_extended
# ---------------------------------------------------------------------------

class TestBreaksExtended:
    def test_basic_5_breaks(self):
        b = scales.breaks_extended(5)
        result = b((0, 100))
        # Should return approximately 5 nice breaks
        assert len(result) >= 3
        assert len(result) <= 8
        # Should include endpoints or be close
        assert result[0] <= 0
        assert result[-1] >= 100

    def test_returns_array(self):
        b = scales.breaks_extended(5)
        result = b((0, 100))
        assert isinstance(result, np.ndarray)

    def test_nice_numbers(self):
        b = scales.breaks_extended(5)
        result = b((0, 100))
        # Breaks should be round numbers
        for v in result:
            assert v == int(v)

    def test_narrow_range(self):
        b = scales.breaks_extended(5)
        result = b((0, 1))
        assert len(result) >= 2
        assert result[0] <= 0
        assert result[-1] >= 1

    def test_negative_range(self):
        b = scales.breaks_extended(5)
        result = b((-100, 0))
        assert result[0] <= -100
        assert result[-1] >= 0


# ---------------------------------------------------------------------------
# breaks_pretty
# ---------------------------------------------------------------------------

class TestBreaksPretty:
    def test_basic_5_breaks(self):
        b = scales.breaks_pretty(5)
        result = b((0, 10))
        # Should return nice round numbers
        assert len(result) >= 3
        for v in result:
            assert v == int(v)

    def test_expected_values(self):
        b = scales.breaks_pretty(5)
        result = b((0, 10))
        np.testing.assert_allclose(result, [0, 2, 4, 6, 8, 10])

    def test_zero_range_input(self):
        b = scales.breaks_pretty()
        result = b((1, 1))
        assert 1.0 in result

    def test_different_n(self):
        b5 = scales.breaks_pretty(5)
        b10 = scales.breaks_pretty(10)
        r5 = b5((0, 100))
        r10 = b10((0, 100))
        # More breaks requested should give more or equal breaks
        assert len(r10) >= len(r5)


# ---------------------------------------------------------------------------
# breaks_width
# ---------------------------------------------------------------------------

class TestBreaksWidth:
    def test_basic_width_10(self):
        b = scales.breaks_width(10)
        result = b((0, 100))
        expected = np.arange(0, 110, 10, dtype=float)
        np.testing.assert_allclose(result, expected)

    def test_width_25(self):
        b = scales.breaks_width(25)
        result = b((0, 100))
        np.testing.assert_allclose(result, [0, 25, 50, 75, 100])

    def test_with_offset(self):
        b = scales.breaks_width(10, offset=5)
        result = b((0, 100))
        # Breaks should be offset by 5
        expected = np.arange(-5, 110, 10, dtype=float)
        np.testing.assert_allclose(result, expected)

    def test_fractional_width(self):
        b = scales.breaks_width(0.5)
        result = b((0, 2))
        expected = np.arange(0, 2.5, 0.5)
        np.testing.assert_allclose(result, expected)

    def test_non_zero_start(self):
        b = scales.breaks_width(5)
        result = b((10, 30))
        np.testing.assert_allclose(result, [10, 15, 20, 25, 30])


# ---------------------------------------------------------------------------
# breaks_log
# ---------------------------------------------------------------------------

class TestBreaksLog:
    def test_powers_of_10(self):
        b = scales.breaks_log(5, base=10)
        result = b((1, 10000))
        np.testing.assert_allclose(result, [1, 10, 100, 1000, 10000])

    def test_base_2(self):
        b = scales.breaks_log(5, base=2)
        result = b((1, 16))
        # Should contain powers of 2
        for v in result:
            assert v > 0

    def test_returns_array(self):
        b = scales.breaks_log(5, base=10)
        result = b((1, 1000))
        assert isinstance(result, np.ndarray)

    def test_narrow_log_range(self):
        b = scales.breaks_log(5, base=10)
        result = b((1, 10))
        assert len(result) >= 2
        assert 1 in result or result[0] <= 1
        assert 10 in result or result[-1] >= 10


# ---------------------------------------------------------------------------
# breaks_exp
# ---------------------------------------------------------------------------

class TestBreaksExp:
    def test_basic(self):
        b = scales.breaks_exp(5)
        result = b((1, 100))
        assert isinstance(result, np.ndarray)
        assert len(result) >= 2

    def test_small_range(self):
        b = scales.breaks_exp()
        result = b((0, 2))
        assert len(result) >= 2

    def test_shifted_range(self):
        b = scales.breaks_exp()
        result = b((100, 102))
        assert len(result) >= 2
        assert result[0] >= 100
        assert result[-1] <= 102


# ---------------------------------------------------------------------------
# breaks_timespan
# ---------------------------------------------------------------------------

class TestBreaksTimespan:
    def test_secs(self):
        b = scales.breaks_timespan("secs", n=5)
        assert callable(b)

    def test_returns_callable(self):
        b = scales.breaks_timespan("secs")
        assert callable(b)


# ---------------------------------------------------------------------------
# minor_breaks_n
# ---------------------------------------------------------------------------

class TestMinorBreaksN:
    def test_2_between_each_pair(self):
        mb = scales.minor_breaks_n(2)
        result = mb([0, 5, 10], (0, 10))
        # Should return minor breaks between major pairs
        assert len(result) > 0
        # All minor breaks should be within the range
        assert np.all(result >= 0)
        assert np.all(result <= 10)

    def test_1_between_each_pair(self):
        mb = scales.minor_breaks_n(1)
        result = mb([0, 10], (0, 10))
        assert len(result) >= 1

    def test_returns_array(self):
        mb = scales.minor_breaks_n(2)
        result = mb([0, 5, 10], (0, 10))
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# minor_breaks_width
# ---------------------------------------------------------------------------

class TestMinorBreaksWidth:
    def test_fixed_width(self):
        mbw = scales.minor_breaks_width(2.5)
        result = mbw([0, 5, 10], (0, 10))
        np.testing.assert_allclose(result, [0, 2.5, 5, 7.5, 10])

    def test_returns_array(self):
        mbw = scales.minor_breaks_width(1)
        result = mbw([0, 5, 10], (0, 10))
        assert isinstance(result, np.ndarray)

    def test_small_width(self):
        mbw = scales.minor_breaks_width(0.5)
        result = mbw([0, 2], (0, 2))
        assert len(result) >= 4
