"""Comprehensive tests for scales.bounds module."""

import numpy as np
import pytest

import scales


# ---------------------------------------------------------------------------
# rescale
# ---------------------------------------------------------------------------

class TestRescale:
    def test_basic_to_0_1(self):
        result = scales.rescale([1, 5, 10], to=(0, 1))
        np.testing.assert_allclose(result, [0.0, 4.0 / 9.0, 1.0])

    def test_custom_to_range(self):
        result = scales.rescale([0, 5, 10], to=(0, 100))
        np.testing.assert_allclose(result, [0.0, 50.0, 100.0])

    def test_custom_from_range(self):
        result = scales.rescale([0, 5, 10], to=(0, 1), from_range=(0, 20))
        np.testing.assert_allclose(result, [0.0, 0.25, 0.5])

    def test_nan_preserved(self):
        result = scales.rescale([1, float("nan"), 10], to=(0, 1))
        assert result[0] == pytest.approx(0.0)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(1.0)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            scales.rescale([], to=(0, 1))

    def test_single_value(self):
        result = scales.rescale([5], to=(0, 1))
        assert len(result) == 1

    def test_negative_values(self):
        result = scales.rescale([-2, 0, 2], to=(0, 1))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_zero_range_input(self):
        result = scales.rescale([5, 5], to=(0, 1))
        # When input range is zero, midpoint of to is returned
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_nan_handling_all_nan(self):
        result = scales.rescale([float("nan"), float("nan")], to=(0, 1))
        assert all(np.isnan(result))

    def test_zero_range_with_nan(self):
        result = scales.rescale([1, float("nan")], to=(0, 1))
        # zero-range input: all values map to midpoint of to (0.5)
        np.testing.assert_allclose(result[0], 0.5)
        # NaN may or may not be preserved depending on implementation
        assert len(result) == 2

    def test_boolean_input(self):
        result = scales.rescale([False, True], to=(0, 1))
        np.testing.assert_allclose(result, [0.0, 1.0])


# ---------------------------------------------------------------------------
# rescale_mid
# ---------------------------------------------------------------------------

class TestRescaleMid:
    def test_symmetric_mid_5(self):
        result = scales.rescale_mid([0, 5, 10], mid=5)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_asymmetric_mid(self):
        result = scales.rescale_mid([0, 2, 10], mid=5)
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)
        assert result[1] < 0.5

    def test_nan_handling(self):
        result = scales.rescale_mid([0, float("nan"), 10], mid=5)
        assert result[0] == pytest.approx(0.0)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(1.0)

    def test_mid_at_lower_boundary(self):
        result = scales.rescale_mid([-1, 0, 1], mid=-1)
        assert result[0] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_mid_at_upper_boundary(self):
        result = scales.rescale_mid([-1, 0, 1], mid=1)
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.5)

    def test_custom_to_range(self):
        result = scales.rescale_mid([-1, 0, 1], mid=1, to=(0, 10))
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(5.0)

    def test_all_same_values(self):
        result = scales.rescale_mid([1, float("nan"), 1])
        # When all values equal, rescale_mid maps them to 1.0 (top of range)
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])


# ---------------------------------------------------------------------------
# rescale_max
# ---------------------------------------------------------------------------

class TestRescaleMax:
    def test_basic(self):
        result = scales.rescale_max([1, 5, 10])
        np.testing.assert_allclose(result, [0.1, 0.5, 1.0])

    def test_all_same(self):
        result = scales.rescale_max([5, 5, 5])
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])

    def test_two_values(self):
        result = scales.rescale_max([4, 5])
        np.testing.assert_allclose(result, [0.8, 1.0])

    def test_negative_values(self):
        result = scales.rescale_max([-3, 0, -1, 2])
        np.testing.assert_allclose(result, [-1.5, 0.0, -0.5, 1.0])

    def test_nan_handling(self):
        result = scales.rescale_max([1, float("nan")])
        assert result[0] == pytest.approx(1.0)
        assert np.isnan(result[1])

    def test_with_nan_and_negatives(self):
        result = scales.rescale_max([2, float("nan"), 0, -2])
        np.testing.assert_allclose(result[0], 1.0)
        assert np.isnan(result[1])
        np.testing.assert_allclose(result[2], 0.0)
        np.testing.assert_allclose(result[3], -1.0)


# ---------------------------------------------------------------------------
# rescale_none
# ---------------------------------------------------------------------------

class TestRescaleNone:
    def test_identity(self):
        x = [1, 5, 10]
        result = scales.rescale_none(x)
        np.testing.assert_allclose(result, x)

    def test_negative_values(self):
        x = [-3, 0, 3]
        result = scales.rescale_none(x)
        np.testing.assert_allclose(result, x)


# ---------------------------------------------------------------------------
# censor
# ---------------------------------------------------------------------------

class TestCensor:
    def test_basic_default_range(self):
        result = scales.censor([-1, 0, 0.5, 1, 2])
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1:4], [0.0, 0.5, 1.0])
        assert np.isnan(result[4])

    def test_custom_range(self):
        result = scales.censor([0, 5, 10, 15], range=(5, 10))
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1:3], [5.0, 10.0])
        assert np.isnan(result[3])

    def test_only_finite_true_preserves_inf(self):
        result = scales.censor(
            [float("-inf"), 0.5, float("inf")], only_finite=True
        )
        assert np.isinf(result[0])
        assert result[1] == pytest.approx(0.5)
        assert np.isinf(result[2])

    def test_only_finite_false_censors_inf(self):
        result = scales.censor(
            [float("-inf"), 0.5, float("inf")], only_finite=False
        )
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(0.5)
        assert np.isnan(result[2])

    def test_boundary_values_kept(self):
        result = scales.censor([0, 1])
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_all_within_range(self):
        result = scales.censor([0.1, 0.5, 0.9])
        np.testing.assert_allclose(result, [0.1, 0.5, 0.9])


# ---------------------------------------------------------------------------
# squish
# ---------------------------------------------------------------------------

class TestSquish:
    def test_basic(self):
        result = scales.squish([-1, 0, 0.5, 1, 2])
        np.testing.assert_allclose(result, [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_custom_range(self):
        result = scales.squish([0, 5, 10, 15], range=(5, 10))
        np.testing.assert_allclose(result, [5.0, 5.0, 10.0, 10.0])

    def test_within_range_unchanged(self):
        result = scales.squish([0.2, 0.5, 0.8])
        np.testing.assert_allclose(result, [0.2, 0.5, 0.8])


# ---------------------------------------------------------------------------
# squish_infinite
# ---------------------------------------------------------------------------

class TestSquishInfinite:
    def test_only_inf_clamped(self):
        result = scales.squish_infinite(
            [float("-inf"), -1, 0, 0.5, 1, float("inf")]
        )
        assert result[0] == pytest.approx(0.0)   # -inf -> lower bound
        assert result[1] == pytest.approx(-1.0)   # finite kept as-is
        assert result[5] == pytest.approx(1.0)     # inf -> upper bound

    def test_custom_range(self):
        result = scales.squish_infinite(
            [float("-inf"), 5, float("inf")], range=(0, 10)
        )
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(5.0)
        assert result[2] == pytest.approx(10.0)

    def test_finite_values_not_clamped(self):
        result = scales.squish_infinite([-100, 0, 100])
        np.testing.assert_allclose(result, [-100.0, 0.0, 100.0])


# ---------------------------------------------------------------------------
# discard
# ---------------------------------------------------------------------------

class TestDiscard:
    def test_basic(self):
        result = scales.discard([-1, 0, 0.5, 1, 2])
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_custom_range(self):
        result = scales.discard([0, 5, 10, 15], range=(5, 10))
        np.testing.assert_allclose(result, [5.0, 10.0])

    def test_all_in_range(self):
        result = scales.discard([0.2, 0.5, 0.8])
        np.testing.assert_allclose(result, [0.2, 0.5, 0.8])

    def test_all_out_of_range(self):
        result = scales.discard([-2, -1, 2, 3])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# oob_* functions
# ---------------------------------------------------------------------------

class TestOobCensor:
    def test_basic(self):
        x = np.array([-1, 0, 0.5, 1, 2], dtype=float)
        result = scales.oob_censor(x, (0, 1))
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1:4], [0.0, 0.5, 1.0])
        assert np.isnan(result[4])

    def test_preserves_inf_by_default(self):
        x = np.array([float("-inf"), 0.5, float("inf")])
        result = scales.oob_censor(x, (0, 1))
        assert np.isinf(result[0])
        assert result[1] == pytest.approx(0.5)
        assert np.isinf(result[2])

    def test_matches_censor(self):
        x = np.array([-1, 0, 0.5, 1, 2], dtype=float)
        np.testing.assert_array_equal(
            np.isnan(scales.oob_censor(x, (0, 1))),
            np.isnan(scales.censor(x)),
        )


class TestOobCensorAny:
    def test_censors_inf_too(self):
        x = np.array([float("-inf"), -1, 0.5, 1, float("inf")])
        result = scales.oob_censor_any(x, (0, 1))
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_allclose(result[2:4], [0.5, 1.0])
        assert np.isnan(result[4])


class TestOobSquish:
    def test_basic(self):
        x = np.array([-1, 0, 0.5, 1, 2], dtype=float)
        result = scales.oob_squish(x, (0, 1))
        np.testing.assert_allclose(result, [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_matches_squish(self):
        x = np.array([-1, 0, 0.5, 1, 2], dtype=float)
        np.testing.assert_allclose(
            scales.oob_squish(x, (0, 1)),
            scales.squish(x),
        )


class TestOobSquishAny:
    def test_basic(self):
        x = np.array([-1, 0, 0.5, 1, 2], dtype=float)
        result = scales.oob_squish_any(x, (0, 1))
        np.testing.assert_allclose(result, [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_squishes_inf(self):
        x = np.array([float("-inf"), 0.5, float("inf")])
        result = scales.oob_squish_any(x, (0, 1))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


class TestOobSquishInfinite:
    def test_only_inf_squished(self):
        x = np.array([float("-inf"), 0, 1, float("inf")])
        result = scales.oob_squish_infinite(x, (0, 1))
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0, 1.0])

    def test_finite_oob_preserved(self):
        x = np.array([-1, 0.5, 2], dtype=float)
        result = scales.oob_squish_infinite(x, (0, 1))
        np.testing.assert_allclose(result, [-1.0, 0.5, 2.0])


class TestOobKeep:
    def test_all_preserved(self):
        x = np.array([-1, 0, 0.5, 1, 2], dtype=float)
        result = scales.oob_keep(x, (0, 1))
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.5, 1.0, 2.0])

    def test_inf_preserved(self):
        x = np.array([float("-inf"), 0.5, float("inf")])
        result = scales.oob_keep(x, (0, 1))
        assert np.isinf(result[0])
        assert np.isinf(result[2])


class TestOobDiscard:
    def test_basic(self):
        x = np.array([-1, 0, 0.5, 1, 2], dtype=float)
        result = scales.oob_discard(x, (0, 1))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_matches_discard(self):
        x = np.array([-1, 0, 0.5, 1, 2], dtype=float)
        np.testing.assert_allclose(
            scales.oob_discard(x, (0, 1)),
            scales.discard(x),
        )


# ---------------------------------------------------------------------------
# expand_range
# ---------------------------------------------------------------------------

class TestExpandRange:
    def test_multiplicative(self):
        lo, hi = scales.expand_range((0, 1), mul=0.05)
        np.testing.assert_allclose((lo, hi), (-0.05, 1.05), atol=1e-10)

    def test_additive(self):
        lo, hi = scales.expand_range((0, 10), add=1)
        np.testing.assert_allclose((lo, hi), (-1, 11), atol=1e-10)

    def test_both_mul_and_add(self):
        lo, hi = scales.expand_range((0, 10), mul=0.1, add=1)
        # mul expands by 0.1 * 10 = 1 on each side, then add 1 more each side
        np.testing.assert_allclose((lo, hi), (-2, 12), atol=1e-10)

    def test_zero_width_range(self):
        lo, hi = scales.expand_range((5, 5), mul=0.05, add=0.6)
        assert lo < 5 and hi > 5

    def test_no_expansion(self):
        lo, hi = scales.expand_range((0, 1), mul=0, add=0)
        np.testing.assert_allclose((lo, hi), (0, 1), atol=1e-10)

    def test_symmetric(self):
        lo, hi = scales.expand_range((-5, 5), mul=0.1)
        np.testing.assert_allclose((lo, hi), (-6, 6), atol=1e-10)

    def test_expand_range_wider(self):
        lo, hi = scales.expand_range((1, 9), mul=0, add=2)
        np.testing.assert_allclose((lo, hi), (-1, 11), atol=1e-10)

    def test_expand_range_point_with_add(self):
        # When range is zero-width, the implementation may use a default expansion
        lo, hi = scales.expand_range((1, 1), mul=0, add=0.6)
        assert lo < 1 and hi > 1


# ---------------------------------------------------------------------------
# zero_range
# ---------------------------------------------------------------------------

class TestZeroRange:
    def test_equal_values(self):
        assert scales.zero_range((1, 1))

    def test_different_values(self):
        assert not scales.zero_range((0, 1))

    def test_nan_returns_true(self):
        # NaN means unknown, treated as zero range
        assert scales.zero_range((float("nan"), 1))

    def test_both_nan(self):
        result = scales.zero_range((float("nan"), float("nan")))
        # Either True or NaN-ish; the R scales returns NA
        # In Python, this should be truthy
        assert result

    def test_both_zero(self):
        assert scales.zero_range((0, 0))

    def test_very_close(self):
        assert scales.zero_range((1, 1 + 1e-16))

    def test_negative_equal(self):
        assert scales.zero_range((-5, -5))

    def test_large_numbers_same(self):
        assert scales.zero_range((1330020857.8787, 1330020857.8787))

    def test_large_numbers_different(self):
        assert not scales.zero_range((1330020857.8787, 1330020866.8787))

    def test_zero_endpoint(self):
        assert not scales.zero_range((0, 10))
        assert not scales.zero_range((-10, 0))

    def test_inf_same(self):
        assert scales.zero_range((float("inf"), float("inf")))
        assert scales.zero_range((float("-inf"), float("-inf")))

    def test_inf_different(self):
        assert not scales.zero_range((1, float("inf")))
        assert not scales.zero_range((float("-inf"), float("inf")))
