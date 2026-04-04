"""Tests for scales continuous and discrete scale functions."""

import numpy as np
import pytest

import scales


# ---------------------------------------------------------------------------
# cscale (continuous scale)
# ---------------------------------------------------------------------------

class TestCscale:
    def test_basic(self):
        pal = scales.pal_gradient_n(["red", "blue"])
        result = scales.cscale([0, 0.5, 1], pal)
        assert len(result) == 3

    def test_endpoint_colors(self):
        pal = scales.pal_gradient_n(["red", "blue"])
        result = scales.cscale([0, 0.5, 1], pal)
        assert "FF0000" in result[0]
        assert "0000FF" in result[2]

    def test_midpoint_color(self):
        pal = scales.pal_gradient_n(["red", "blue"])
        result = scales.cscale([0, 0.5, 1], pal)
        # Midpoint should be a mix
        assert result[1] != result[0]
        assert result[1] != result[2]

    def test_single_value(self):
        pal = scales.pal_gradient_n(["red", "blue"])
        result = scales.cscale([0.5], pal)
        assert len(result) == 1

    def test_with_rescale_pal(self):
        pal = scales.pal_rescale()
        result = scales.cscale([0, 0.5, 1], pal)
        np.testing.assert_allclose(result[0], 0.1, atol=0.01)
        np.testing.assert_allclose(result[2], 1.0, atol=0.01)

    def test_numpy_array_input(self):
        pal = scales.pal_gradient_n(["red", "blue"])
        result = scales.cscale(np.array([0, 0.5, 1.0]), pal)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# train_continuous
# ---------------------------------------------------------------------------

class TestTrainContinuous:
    def test_basic(self):
        result = scales.train_continuous([1, 5, 10])
        assert result == (1.0, 10.0)

    def test_with_existing_wider(self):
        result = scales.train_continuous([3, 7], existing=(1, 10))
        assert result == (1, 10)

    def test_with_existing_narrower(self):
        result = scales.train_continuous([0, 20], existing=(5, 15))
        assert result[0] <= 0
        assert result[1] >= 20

    def test_single_value(self):
        result = scales.train_continuous([5])
        assert result[0] == 5.0
        assert result[1] == 5.0

    def test_negative_values(self):
        result = scales.train_continuous([-10, -1])
        assert result == (-10.0, -1.0)

    def test_numpy_input(self):
        result = scales.train_continuous(np.array([1, 5, 10]))
        assert result == (1.0, 10.0)

    def test_preserves_existing_range(self):
        # New data within existing range should not shrink it
        result = scales.train_continuous([3, 7], existing=(1, 10))
        assert result[0] <= 1
        assert result[1] >= 10

    def test_nan_ignored(self):
        result = scales.train_continuous([1, np.nan, 10])
        assert result == (1.0, 10.0)


# ---------------------------------------------------------------------------
# dscale (discrete scale)
# ---------------------------------------------------------------------------

class TestDscale:
    def test_basic(self):
        pal = scales.pal_brewer("qual", "Set1")
        result = scales.dscale(["a", "b", "c"], pal)
        assert len(result) == 3

    def test_returns_colors(self):
        pal = scales.pal_brewer("qual", "Set1")
        result = scales.dscale(["a", "b", "c"], pal)
        for c in result:
            assert c.startswith("#")

    def test_single_level(self):
        pal = scales.pal_brewer("qual", "Set1")
        result = scales.dscale(["a"], pal)
        assert len(result) == 1

    def test_many_levels(self):
        pal = scales.pal_brewer("qual", "Set1")
        result = scales.dscale(["a", "b", "c", "d", "e"], pal)
        assert len(result) == 5

    def test_with_manual_pal(self):
        pal = scales.pal_manual(["red", "blue", "green"])
        result = scales.dscale(["x", "y", "z"], pal)
        assert [str(x) for x in result] == ["red", "blue", "green"]


# ---------------------------------------------------------------------------
# train_discrete
# ---------------------------------------------------------------------------

class TestTrainDiscrete:
    def test_basic(self):
        result = scales.train_discrete(["a", "b"])
        assert "a" in result
        assert "b" in result

    def test_with_existing(self):
        result = scales.train_discrete(["a", "b"], existing=["c"])
        assert "a" in result or any(str(x) == "a" for x in result)
        assert "b" in result or any(str(x) == "b" for x in result)
        assert "c" in result or any(str(x) == "c" for x in result)

    def test_preserves_order(self):
        result = scales.train_discrete(["a", "b"], existing=["c"])
        # Existing should come first
        assert str(result[0]) == "c"

    def test_no_duplicates(self):
        result = scales.train_discrete(["a", "b", "a"], existing=["b"])
        str_result = [str(x) for x in result]
        for item in set(str_result):
            count = str_result.count(item)
            assert count == 1, f"Duplicate found: {item}"

    def test_empty_new(self):
        result = scales.train_discrete([], existing=["a", "b"])
        str_result = [str(x) for x in result]
        assert "a" in str_result
        assert "b" in str_result

    def test_empty_existing(self):
        result = scales.train_discrete(["a", "b"])
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# ContinuousRange and DiscreteRange classes
# ---------------------------------------------------------------------------

class TestContinuousRange:
    def test_basic(self):
        rng = scales.ContinuousRange()
        rng.train([1, 5, 3])
        assert rng.range == (1.0, 5.0)

    def test_progressive_training(self):
        rng = scales.ContinuousRange()
        rng.train([1, 5, 3])
        rng.train([0, 4])
        assert rng.range == (0.0, 5.0)

    def test_reset(self):
        rng = scales.ContinuousRange()
        rng.train([1, 5])
        rng.reset()
        rng.train([10, 20])
        assert rng.range == (10.0, 20.0)

    def test_nan_ignored(self):
        rng = scales.ContinuousRange()
        rng.train([1, np.nan, 5])
        assert rng.range == (1.0, 5.0)


class TestDiscreteRange:
    def test_basic(self):
        rng = scales.DiscreteRange()
        rng.train(["b", "a", "c"])
        assert rng.range == ["b", "a", "c"]

    def test_progressive_training(self):
        rng = scales.DiscreteRange()
        rng.train(["b", "a", "c"])
        rng.train(["d", "a"])
        assert rng.range == ["b", "a", "c", "d"]

    def test_reset(self):
        rng = scales.DiscreteRange()
        rng.train(["a", "b"])
        rng.reset()
        rng.train(["x", "y"])
        assert rng.range == ["x", "y"]

    def test_preserves_order(self):
        rng = scales.DiscreteRange()
        rng.train(["c", "a", "b"])
        assert rng.range[0] == "c"
        assert rng.range[1] == "a"
        assert rng.range[2] == "b"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestScaleEdgeCases:
    def test_cscale_boundary_values(self):
        pal = scales.pal_gradient_n(["red", "blue"])
        result = scales.cscale([0, 1], pal)
        assert len(result) == 2
        assert "FF0000" in result[0]
        assert "0000FF" in result[1]

    def test_train_continuous_all_same(self):
        result = scales.train_continuous([5, 5, 5])
        assert result[0] == 5.0
        assert result[1] == 5.0

    def test_dscale_single(self):
        pal = scales.pal_brewer("qual", "Set1")
        result = scales.dscale(["a"], pal)
        assert len(result) == 1
