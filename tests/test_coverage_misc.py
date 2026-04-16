"""Coverage tests for remaining modules: range, scale_continuous, scale_discrete, bounds."""

import numpy as np
import pandas as pd
import pytest

from scales.range import Range, ContinuousRange, DiscreteRange
from scales.scale_continuous import cscale, train_continuous
from scales.scale_discrete import dscale, train_discrete, _is_na, _na_key
from scales.bounds import rescale, rescale_mid, rescale_max, censor, squish, squish_infinite, discard, trim_to_domain
from scales._utils import expand_range


# ===========================================================================
# range.py (lines 44, 80, 139-142, 152, 159, 171)
# ===========================================================================

class TestContinuousRange:
    def test_train_empty(self):
        rng = ContinuousRange()
        rng.train([float("nan")])
        assert rng.range is None

    def test_reset(self):
        rng = ContinuousRange()
        rng.train([1, 5])
        rng.reset()
        assert rng.range is None


class TestDiscreteRange:
    def test_train_with_nan(self):
        rng = DiscreteRange()
        rng.train([1, 2, float("nan")])
        assert rng.range is not None

    def test_train_na_rm(self):
        rng = DiscreteRange()
        rng.train([1, 2, None, float("nan")], na_rm=True)
        # NaN and None should be removed
        assert all(v is not None for v in rng.range)
        for v in rng.range:
            if isinstance(v, float):
                assert not np.isnan(v)

    def test_train_merge(self):
        rng = DiscreteRange()
        rng.train(["a", "b"])
        rng.train(["b", "c"])
        assert "a" in rng.range
        assert "c" in rng.range
        assert rng.range.count("b") == 1

    def test_reset(self):
        rng = DiscreteRange()
        rng.train(["a", "b"])
        rng.reset()
        assert rng.range is None
        assert rng._is_factor is False

    def test_train_merge_with_nan(self):
        rng = DiscreteRange()
        rng.train([1, float("nan")])
        rng.train([2, float("nan")])
        # nan should appear only once
        nan_count = sum(1 for v in rng.range if isinstance(v, float) and np.isnan(v))
        assert nan_count == 1


# ===========================================================================
# scale_continuous.py (lines 80-83, 121-123)
# ===========================================================================

class TestCscale:
    def test_with_transform(self):
        pal = lambda x: x ** 2
        result = cscale([1, 10, 100], pal, trans="log10")
        assert len(result) == 3

    def test_with_na(self):
        pal = lambda x: x
        result = cscale([1, float("nan"), 10], pal, na_value=-1)
        assert result[1] == -1

    def test_string_result_na(self):
        pal = lambda x: np.array(["a"] * len(x))
        result = cscale([1, float("nan"), 10], pal, na_value="NA")
        assert result[1] == "NA"


class TestTrainContinuous:
    def test_empty_no_existing(self):
        with pytest.raises(ValueError):
            train_continuous([float("nan")])

    def test_empty_with_existing(self):
        result = train_continuous([float("nan")], existing=(0.0, 10.0))
        assert result == (0.0, 10.0)

    def test_merge(self):
        result = train_continuous([0, 4], existing=(1.0, 5.0))
        assert result == (0.0, 5.0)


# ===========================================================================
# scale_discrete.py (lines 56-57, 72, 88, 135-137, 149, 173, 183)
# ===========================================================================

class TestDscale:
    def test_basic(self):
        pal = lambda n: list(range(n))
        result = dscale(["a", "b", "a", "c"], pal)
        assert len(result) == 4

    def test_with_na(self):
        pal = lambda n: list(range(n))
        result = dscale(["a", None, "b"], pal, na_value=-1)
        assert result[1] == -1

    def test_empty_levels(self):
        pal = lambda n: list(range(n))
        result = dscale([None, None], pal, na_value=-1)
        assert all(v == -1 for v in result)


class TestTrainDiscrete:
    def test_basic(self):
        result = train_discrete(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_merge(self):
        result = train_discrete(["b", "d"], existing=["a", "b", "c"])
        assert result == ["a", "b", "c", "d"]

    def test_na_rm(self):
        result = train_discrete(["a", None, "b"], na_rm=True)
        assert None not in result

    def test_with_nan(self):
        result = train_discrete(["a", float("nan"), "b"])
        assert len(result) == 3


class TestIsNa:
    def test_none(self):
        assert _is_na(None)

    def test_nan(self):
        assert _is_na(float("nan"))

    def test_string(self):
        assert not _is_na("hello")

    def test_number(self):
        assert not _is_na(5)


class TestNaKey:
    def test_na_values(self):
        assert _na_key(None) is None
        assert _na_key(float("nan")) is None

    def test_normal_value(self):
        assert _na_key("hello") == "hello"


# ===========================================================================
# bounds.py (lines 49, 133, 493-496)
# ===========================================================================

class TestBoundsEdge:
    def test_rescale_mid(self):
        result = rescale_mid([0, 5, 10], to=(0, 1), from_range=(0, 10), mid=5)
        assert len(result) == 3

    def test_rescale_max(self):
        result = rescale_max([0, 5, 10])
        assert len(result) == 3

    def test_datetime_as_numeric(self):
        dt = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")
        result = rescale(dt, to=(0, 1))
        assert len(result) == 2

    def test_all_nan_rescale_mid(self):
        result = rescale_mid([float("nan")], to=(0, 1))
        assert len(result) == 1


class TestTrimToDomain:
    def test_basic(self):
        # Mirrors R's trim_to_domain: returns the transformed range after
        # squishing x into the transform's domain. For log10 with domain
        # (0, inf): squish([-1, 0, 1, 10, 100]) -> [0, 0, 1, 10, 100],
        # then range() -> [0, 100], then log10 -> [-inf, 2]. The -inf
        # is the transformed image of the lower boundary, which the
        # tests above accept as-is per R behaviour.
        from scales.transforms import transform_log10
        t = transform_log10()
        result = trim_to_domain(t, [-1, 0, 1, 10, 100])
        assert result.shape == (2,)
        # Upper end is log10(100) = 2.
        assert result[1] == pytest.approx(2.0)


# ===========================================================================
# Pandas Categorical coverage
# ===========================================================================

class TestPandasCategorical:
    def test_dscale_categorical(self):
        cat = pd.Categorical(["a", "b", "a", "c"], categories=["a", "b", "c"])
        pal = lambda n: list(range(n))
        result = dscale(cat, pal)
        assert len(result) == 4

    def test_train_discrete_categorical(self):
        cat = pd.Categorical(["a", "b", "c"], categories=["a", "b", "c", "d"])
        result = train_discrete(cat, drop=False)
        assert "d" in result

    def test_train_discrete_categorical_drop(self):
        cat = pd.Categorical(["a", "b"], categories=["a", "b", "c"])
        result = train_discrete(cat, drop=True)
        assert "c" not in result

    def test_discrete_range_categorical(self):
        rng = DiscreteRange()
        cat = pd.Categorical(["a", "b"], categories=["a", "b", "c"])
        rng.train(cat)
        assert "c" in rng.range
        assert rng._is_factor is True

    def test_discrete_range_categorical_drop(self):
        rng = DiscreteRange()
        cat = pd.Categorical(["a"], categories=["a", "b"])
        rng.train(cat, drop=True)
        assert "b" not in rng.range


# ===========================================================================
# expand_range (line 123)
# ===========================================================================

class TestExpandRange:
    def test_bad_shape(self):
        with pytest.raises(ValueError):
            expand_range([1, 2, 3])


# ===========================================================================
# Transform derivative coverage
# ===========================================================================

class TestTransformDerivatives:
    def test_log_derivatives(self):
        from scales.transforms import transform_log
        t = transform_log(base=10)
        x = np.array([1.0, 10.0, 100.0])
        d_fwd = t.d_transform(x)
        assert d_fwd is not None
        assert len(d_fwd) == 3
        d_inv = t.d_inverse(np.array([0.0, 1.0, 2.0]))
        assert d_inv is not None

    def test_exp_inverse(self):
        from scales.transforms import transform_exp
        t = transform_exp(base=10)
        y = np.array([10.0, 100.0])
        x_back = t.inverse(y)
        assert np.allclose(x_back, [1.0, 2.0])


# ===========================================================================
# label_timespan sub-nanosecond (lines 1330-1331)
# ===========================================================================

class TestLabelsSubNs:
    def test_sub_nanosecond(self):
        from scales.labels import label_timespan
        fmt = label_timespan()
        result = fmt([1e-12])  # 1 picosecond
        # Should fall through to ns with small value
        assert "ns" in result[0]


# ===========================================================================
# date_breaks (lines 1767-1768)
# ===========================================================================

class TestDateBreaks:
    def test_date_breaks(self):
        from scales.labels import date_breaks
        brk = date_breaks(5)
        result = brk([0, 100])
        assert len(result) > 0


# ===========================================================================
# palettes.py missing lines (401, 554-555, 654-655)
# ===========================================================================

class TestPalettesExtra:
    def test_palette_type_unknown(self):
        from scales.palettes import palette_type
        assert palette_type("not_a_palette") == "unknown"

    def test_viridis_fallback_cmap(self):
        # Test with an option that might trigger fallback
        from scales.palettes import pal_viridis
        pal = pal_viridis(option="D")
        result = pal(3)
        assert len(result) == 3


# ===========================================================================
# colour_mapping.py missing lines (151, 290, 315-316, 456-457)
# ===========================================================================

class TestColourMappingExtra:
    def test_col_bin_n_bins_lt_1(self):
        from scales.colour_mapping import col_bin
        f = col_bin("viridis", bins=[5])  # Only 1 break -> 0 bins
        result = f([3, 7])
        assert all(c == "#808080" for c in result)

    def test_col_bin_right_out_of_range(self):
        from scales.colour_mapping import col_bin
        f = col_bin("viridis", bins=[0, 5, 10], right=True)
        result = f([-1, 15])  # Out of range
        assert result[0] == "#808080"
        assert result[1] == "#808080"

    def test_col_factor_empty_levels(self):
        from scales.colour_mapping import col_factor
        f = col_factor("viridis", domain=[])
        result = f(["a"])
        assert result[0] == "#808080"
