"""Comprehensive tests for scales.range module."""

import numpy as np
import pytest

import scales


# ---------------------------------------------------------------------------
# ContinuousRange
# ---------------------------------------------------------------------------

class TestContinuousRange:
    def test_train_single(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([1, 5]))
        assert cr.range == (1.0, 5.0)

    def test_train_expands(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([1, 5]))
        cr.train(np.array([3, 10]))
        assert cr.range == (1.0, 10.0)

    def test_train_does_not_shrink(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([-1, 45, 10]))
        assert cr.range == (-1.0, 45.0)
        cr.train(np.array([1000]))
        assert cr.range == (-1.0, 1000.0)

    def test_reset_clears(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([1, 5]))
        cr.reset()
        assert cr.range is None

    def test_nan_ignored(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([1, float("nan"), 5]))
        assert cr.range == (1.0, 5.0)

    def test_inf_ignored(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([1, float("inf"), 5]))
        assert cr.range == (1.0, 5.0)

    def test_nan_and_inf_both_ignored(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([1, float("nan"), float("inf"), 5]))
        assert cr.range == (1.0, 5.0)

    def test_neg_inf_ignored(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([float("-inf"), 1, 5]))
        assert cr.range == (1.0, 5.0)

    def test_initial_range_is_none(self):
        cr = scales.ContinuousRange()
        assert cr.range is None

    def test_train_negative_values(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([-10, -1]))
        assert cr.range == (-10.0, -1.0)

    def test_successive_trains(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([0, 5]))
        cr.train(np.array([-5, 3]))
        cr.train(np.array([2, 20]))
        assert cr.range == (-5.0, 20.0)

    def test_single_value(self):
        cr = scales.ContinuousRange()
        cr.train(np.array([7]))
        assert cr.range == (7.0, 7.0)


# ---------------------------------------------------------------------------
# DiscreteRange
# ---------------------------------------------------------------------------

class TestDiscreteRange:
    def test_train_single(self):
        dr = scales.DiscreteRange()
        dr.train(np.array(["a", "b"]))
        result = list(dr.range)
        assert "a" in result
        assert "b" in result

    def test_train_union(self):
        dr = scales.DiscreteRange()
        dr.train(np.array(["a", "b"]))
        dr.train(np.array(["b", "c"]))
        result = list(dr.range)
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert len(result) == 3

    def test_preserves_order(self):
        dr = scales.DiscreteRange()
        dr.train(np.array(["a", "b"]))
        dr.train(np.array(["b", "c"]))
        result = list(dr.range)
        # "a" and "b" should come before "c"
        assert result.index("a") < result.index("c")
        assert result.index("b") < result.index("c")

    def test_reset_clears(self):
        dr = scales.DiscreteRange()
        dr.train(np.array(["a", "b"]))
        dr.reset()
        assert dr.range is None

    def test_initial_range_is_none(self):
        dr = scales.DiscreteRange()
        assert dr.range is None

    def test_no_duplicates(self):
        dr = scales.DiscreteRange()
        dr.train(np.array(["a", "a", "b"]))
        result = list(dr.range)
        assert len(result) == 2

    def test_numeric_discrete(self):
        dr = scales.DiscreteRange()
        dr.train(np.array([1, 2, 3]))
        dr.train(np.array([3, 4]))
        result = list(dr.range)
        assert len(result) == 4

    def test_successive_trains_expand(self):
        dr = scales.DiscreteRange()
        dr.train(np.array(["x"]))
        dr.train(np.array(["y"]))
        dr.train(np.array(["z"]))
        result = list(dr.range)
        assert len(result) == 3
        assert set(result) == {"x", "y", "z"}

    def test_train_with_existing_values(self):
        dr = scales.DiscreteRange()
        dr.train(np.array(["a", "b", "c"]))
        dr.train(np.array(["a", "c"]))  # subset, should not change range
        result = list(dr.range)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Range base class
# ---------------------------------------------------------------------------

class TestRangeBase:
    def test_continuous_is_range(self):
        cr = scales.ContinuousRange()
        assert isinstance(cr, scales.Range)

    def test_discrete_is_range(self):
        dr = scales.DiscreteRange()
        assert isinstance(dr, scales.Range)
