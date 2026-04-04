"""Comprehensive tests for scales utility functions."""

import numpy as np
import pytest

import scales


# ---------------------------------------------------------------------------
# zero_range
# ---------------------------------------------------------------------------

class TestZeroRange:
    def test_equal_values(self):
        assert scales.zero_range((1, 1))

    def test_different_values(self):
        assert not scales.zero_range((0, 1))

    def test_nan_returns_truthy(self):
        # NaN signals unknown, conventionally treated as zero range
        assert scales.zero_range((float("nan"), 1))

    def test_both_nan(self):
        assert scales.zero_range((float("nan"), float("nan")))

    def test_both_zero(self):
        assert scales.zero_range((0, 0))

    def test_negative_range(self):
        assert scales.zero_range((-5, -5))
        assert not scales.zero_range((-5, 5))

    def test_very_close_values(self):
        # Within machine epsilon
        assert scales.zero_range((1, 1 + 1e-16))

    def test_not_close_enough(self):
        assert not scales.zero_range((0, 0.001))

    def test_large_equal(self):
        big = 1e100
        assert scales.zero_range((big, big)) is True

    def test_small_equal(self):
        small = 1e-100
        assert scales.zero_range((small, small)) is True

    def test_inf_same(self):
        assert scales.zero_range((float("inf"), float("inf")))
        assert scales.zero_range((float("-inf"), float("-inf")))

    def test_inf_different(self):
        assert not scales.zero_range((1, float("inf")))
        assert not scales.zero_range((float("-inf"), float("inf")))

    def test_tolerance_scaling(self):
        eps = np.finfo(float).eps
        # Within default tolerance
        assert scales.zero_range((1, 1 + eps))
        assert scales.zero_range((1, 1 + 99 * eps))
        # Scaling up or down has no effect (values are rescaled internally)
        assert scales.zero_range((100000 * 1, 100000 * (1 + eps)))
        assert scales.zero_range((0.00001 * 1, 0.00001 * (1 + eps)))


# ---------------------------------------------------------------------------
# expand_range
# ---------------------------------------------------------------------------

class TestExpandRange:
    def test_multiplicative(self):
        lo, hi = scales.expand_range((0, 1), mul=0.05)
        np.testing.assert_allclose((lo, hi), (-0.05, 1.05), atol=1e-10)

    def test_additive(self):
        lo, hi = scales.expand_range((0, 10), add=2)
        np.testing.assert_allclose((lo, hi), (-2, 12), atol=1e-10)

    def test_both_mul_and_add(self):
        lo, hi = scales.expand_range((0, 10), mul=0.1, add=1)
        np.testing.assert_allclose((lo, hi), (-2, 12), atol=1e-10)

    def test_no_expansion(self):
        lo, hi = scales.expand_range((0, 1), mul=0, add=0)
        np.testing.assert_allclose((lo, hi), (0, 1), atol=1e-10)

    def test_zero_width(self):
        lo, hi = scales.expand_range((5, 5), mul=0.05, add=0.6)
        assert lo < 5
        assert hi > 5

    def test_symmetric_range(self):
        lo, hi = scales.expand_range((-5, 5), mul=0.1)
        np.testing.assert_allclose((lo, hi), (-6, 6), atol=1e-10)

    def test_point_range_with_add(self):
        lo, hi = scales.expand_range((1, 1), mul=0, add=0.6)
        # zero-width range: expansion uses add around center
        assert lo < 1 and hi > 1

    def test_wider_range_with_add(self):
        lo, hi = scales.expand_range((1, 9), mul=0, add=2)
        np.testing.assert_allclose((lo, hi), (-1, 11), atol=1e-10)

    def test_mul_with_unit_range(self):
        lo, hi = scales.expand_range((1, 1), mul=1, add=0.6)
        # zero-width range: mul has no effect, add expands symmetrically
        assert lo < 1 and hi > 1


# ---------------------------------------------------------------------------
# round_any
# ---------------------------------------------------------------------------

class TestRoundAny:
    def test_round_to_10(self):
        result = scales.round_any(135, 10)
        assert result == pytest.approx(140.0)

    def test_round_to_100(self):
        result = scales.round_any(135, 100)
        assert result == pytest.approx(100.0)

    def test_round_to_1(self):
        result = scales.round_any(1.7, 1)
        assert result == pytest.approx(2.0)

    def test_round_to_0_5(self):
        result = scales.round_any(1.7, 0.5)
        assert result == pytest.approx(1.5)

    def test_round_zero(self):
        result = scales.round_any(0, 10)
        assert result == pytest.approx(0.0)

    def test_round_negative(self):
        result = scales.round_any(-135, 10)
        assert result == pytest.approx(-140.0)

    def test_round_small_accuracy(self):
        result = scales.round_any(0.123, 0.05)
        assert result == pytest.approx(0.10, abs=0.01)

    def test_exact_multiple(self):
        result = scales.round_any(100, 10)
        assert result == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# fullseq
# ---------------------------------------------------------------------------

class TestFullseq:
    def test_basic_quarter_steps(self):
        result = scales.fullseq((0, 1), 0.25)
        np.testing.assert_allclose(result, [0, 0.25, 0.5, 0.75, 1.0])

    def test_integer_steps(self):
        result = scales.fullseq((0, 5), 1)
        np.testing.assert_allclose(result, [0, 1, 2, 3, 4, 5])

    def test_larger_steps(self):
        result = scales.fullseq((0, 10), 5)
        np.testing.assert_allclose(result, [0, 5, 10])

    def test_non_zero_start(self):
        result = scales.fullseq((2, 4), 0.5)
        np.testing.assert_allclose(result, [2, 2.5, 3, 3.5, 4])

    def test_step_larger_than_range(self):
        result = scales.fullseq((0, 1), 5)
        # Should still return endpoints
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# precision
# ---------------------------------------------------------------------------

class TestPrecision:
    def test_half_step(self):
        result = scales.precision([1, 1.5, 2])
        assert result == pytest.approx(0.1, abs=0.05)

    def test_integers(self):
        result = scales.precision([1, 2, 3])
        assert result >= 0.5

    def test_fine_precision(self):
        result = scales.precision([0.01, 0.02, 0.03])
        assert result <= 0.01

    def test_single_value(self):
        result = scales.precision([5])
        assert result > 0

    def test_mixed_precision(self):
        result = scales.precision([0, 0.5, 1, 1.5])
        assert result == pytest.approx(0.1, abs=0.05)
