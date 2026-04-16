"""Coverage tests for scales/transforms.py – targeting uncovered lines."""

import numpy as np
import pytest

from scales.transforms import (
    Transform,
    new_transform,
    is_transform,
    as_transform,
    trans_breaks,
    trans_format,
    transform_identity,
    transform_log,
    transform_log10,
    transform_log2,
    transform_log1p,
    transform_exp,
    transform_sqrt,
    transform_reverse,
    transform_reciprocal,
    transform_asinh,
    transform_asn,
    transform_atanh,
    transform_boxcox,
    transform_modulus,
    transform_yj,
    transform_pseudo_log,
    transform_logit,
    transform_probit,
    transform_probability,
    transform_date,
    transform_time,
    transform_timespan,
    transform_compose,
    # Legacy aliases
    trans_new,
    identity_trans,
    log_trans,
    log10_trans,
    log2_trans,
    log1p_trans,
    exp_trans,
    sqrt_trans,
    reverse_trans,
    reciprocal_trans,
    asinh_trans,
    asn_trans,
    atanh_trans,
    boxcox_trans,
    modulus_trans,
    yj_trans,
    pseudo_log_trans,
    logit_trans,
    probit_trans,
    probability_trans,
    date_trans,
    time_trans,
    timespan_trans,
    hms_trans,
    compose_trans,
    is_trans,
    as_trans,
    _pretty_breaks,
    _log_breaks,
    _default_format,
)


# ---------------------------------------------------------------------------
# _pretty_breaks edge cases (lines 103, 108-109)
# ---------------------------------------------------------------------------

class TestPrettyBreaksHelper:
    def test_equal_limits(self):
        brk = _pretty_breaks(5)
        result = brk((5, 5))
        assert len(result) == 1
        assert result[0] == 5

    def test_non_finite(self):
        brk = _pretty_breaks(5)
        result = brk((float("nan"), 10))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _log_breaks edge cases (lines 116-125)
# ---------------------------------------------------------------------------

class TestLogBreaksHelper:
    def test_basic(self):
        brk = _log_breaks(base=10, n=5)
        result = brk((1, 10000))
        assert len(result) > 0

    def test_negative_lo(self):
        brk = _log_breaks(base=10, n=5)
        result = brk((-1, 100))
        assert len(result) > 0

    def test_negative_hi(self):
        brk = _log_breaks(base=10, n=5)
        result = brk((-10, -1))
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _default_format (lines 135)
# ---------------------------------------------------------------------------

class TestDefaultFormat:
    def test_nan(self):
        fmt = _default_format()
        result = fmt(np.array([float("nan"), 1.0, 2.5]))
        assert result[0] == "NA"
        assert result[1] == "1"


# ---------------------------------------------------------------------------
# Transform class (repr)
# ---------------------------------------------------------------------------

class TestTransformRepr:
    def test_repr(self):
        t = transform_identity()
        assert "identity" in repr(t)


# ---------------------------------------------------------------------------
# as_transform edge cases (lines 356-365)
# ---------------------------------------------------------------------------

class TestAsTransform:
    def test_string_lookup(self):
        t = as_transform("log10")
        assert t.name == "log-10"

    def test_string_with_suffix(self):
        t = as_transform("log10_trans")
        assert "log" in t.name

    def test_already_transform(self):
        t = transform_identity()
        assert as_transform(t) is t

    def test_unknown_string(self):
        with pytest.raises(ValueError):
            as_transform("nonexistent_transform")

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            as_transform(42)


# ---------------------------------------------------------------------------
# trans_breaks / trans_format (lines 486, 489)
# ---------------------------------------------------------------------------

class TestTransBreaksFormat:
    def test_trans_breaks(self):
        brk = trans_breaks("log10", n=5)
        result = brk((1, 1000))
        assert len(result) > 0

    def test_trans_format_default(self):
        fmt = trans_format("identity")
        result = fmt(np.array([1.0, 2.0, 3.0]))
        assert len(result) == 3

    def test_trans_format_custom(self):
        custom_fmt = lambda x: [f"{v:.1f}" for v in x]
        fmt = trans_format("identity", format=custom_fmt)
        result = fmt(np.array([1.0, 2.0]))
        assert result == ["1.0", "2.0"]


# ---------------------------------------------------------------------------
# transform_exp with custom base (lines 549-558)
# ---------------------------------------------------------------------------

class TestTransformExp:
    def test_default_base(self):
        t = transform_exp()
        x = np.array([0, 1, 2])
        y = t.transform(x)
        assert np.allclose(y, np.exp(x))

    def test_custom_base(self):
        t = transform_exp(base=10)
        x = np.array([0, 1, 2])
        y = t.transform(x)
        assert np.allclose(y, 10.0 ** x)
        assert "power-10" in t.name


# ---------------------------------------------------------------------------
# transform_asn (lines 646-652)
# ---------------------------------------------------------------------------

class TestTransformAsn:
    def test_forward_inverse(self):
        t = transform_asn()
        x = np.array([0, 0.25, 0.5, 0.75, 1.0])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)


# ---------------------------------------------------------------------------
# transform_atanh (line 668)
# ---------------------------------------------------------------------------

class TestTransformAtanh:
    def test_forward_inverse(self):
        t = transform_atanh()
        x = np.array([-0.5, 0, 0.5])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)

    def test_derivatives(self):
        t = transform_atanh()
        x = np.array([0.0, 0.5])
        d_fwd = t.d_transform(x)
        d_inv_val = t.d_inverse(t.transform(x))
        assert d_fwd is not None
        assert d_inv_val is not None


# ---------------------------------------------------------------------------
# transform_boxcox p=0 and p!=0 (lines 710, 713, 722, 725)
# ---------------------------------------------------------------------------

class TestTransformBoxcox:
    def test_p_zero(self):
        t = transform_boxcox(p=0)
        x = np.array([1, 2, 3])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)
        # Test derivatives
        d_fwd = t.d_transform(x)
        d_inv = t.d_inverse(y)
        assert d_fwd is not None
        assert d_inv is not None

    def test_p_nonzero(self):
        t = transform_boxcox(p=2)
        x = np.array([1, 2, 3])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)
        # Test derivatives
        d_fwd = t.d_transform(x)
        d_inv = t.d_inverse(y)
        assert d_fwd is not None
        assert d_inv is not None

    def test_with_offset(self):
        t = transform_boxcox(p=0.5, offset=1)
        x = np.array([0, 1, 2])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)


# ---------------------------------------------------------------------------
# transform_modulus (lines 761, 764-768)
# ---------------------------------------------------------------------------

class TestTransformModulus:
    def test_p_zero(self):
        t = transform_modulus(p=0)
        x = np.array([-3, -1, 0, 1, 3])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)

    def test_p_nonzero(self):
        t = transform_modulus(p=2)
        x = np.array([-3, -1, 0, 1, 3])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)

    def test_negative_offset(self):
        with pytest.raises(ValueError):
            transform_modulus(p=1, offset=-1)


# ---------------------------------------------------------------------------
# transform_yj (lines 820, 837-839)
# ---------------------------------------------------------------------------

class TestTransformYJ:
    def test_p_zero(self):
        t = transform_yj(p=0)
        x = np.array([0, 1, 2])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)

    def test_p_two(self):
        t = transform_yj(p=2)
        x = np.array([-2, -1, 0, 1, 2])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)

    def test_p_one_with_negatives(self):
        t = transform_yj(p=1)
        x = np.array([-3, -2, -1, 0, 1, 2, 3])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)

    def test_p_half(self):
        t = transform_yj(p=0.5)
        x = np.array([-2, 0, 2])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)


# ---------------------------------------------------------------------------
# transform_probability / transform_probit (lines 917-927, 977-978)
# ---------------------------------------------------------------------------

class TestTransformProbability:
    def test_norm(self):
        t = transform_probability("norm")
        x = np.array([0.1, 0.5, 0.9])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)

    def test_logistic(self):
        t = transform_probability("logistic")
        x = np.array([0.2, 0.5, 0.8])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)


class TestTransformProbit:
    def test_forward_inverse(self):
        t = transform_probit()
        x = np.array([0.1, 0.5, 0.9])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-10)


# ---------------------------------------------------------------------------
# transform_logit derivatives (lines 951, 954-955)
# ---------------------------------------------------------------------------

class TestTransformLogit:
    def test_derivatives(self):
        t = transform_logit()
        x = np.array([0.2, 0.5, 0.8])
        d_fwd = t.d_transform(x)
        assert d_fwd is not None
        assert len(d_fwd) == 3
        d_inv = t.d_inverse(t.transform(x))
        assert d_inv is not None


# ---------------------------------------------------------------------------
# transform_date (lines 1010-1014, 1017-1018, 1021-1022)
# ---------------------------------------------------------------------------

class TestTransformDate:
    def test_datetime64_forward(self):
        t = transform_date()
        dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")
        y = t.transform(dates)
        assert y[1] - y[0] == pytest.approx(1.0)

    def test_numeric_forward(self):
        t = transform_date()
        # Already numeric should pass through
        y = t.transform(np.array([0.0, 1.0]))
        assert np.allclose(y, [0.0, 1.0])

    def test_inverse(self):
        t = transform_date()
        days = np.array([0.0, 1.0])
        result = t.inverse(days)
        assert result.dtype.kind == "M"

    def test_format(self):
        t = transform_date()
        result = t.format_func(np.array([0.0, 1.0]))
        assert len(result) == 2
        assert "1970" in result[0]


# ---------------------------------------------------------------------------
# transform_time (lines 1045-1048, 1051-1052, 1055-1056)
# ---------------------------------------------------------------------------

class TestTransformTime:
    def test_datetime64_forward(self):
        t = transform_time()
        times = np.array(["2020-01-01T00:00:00", "2020-01-01T00:01:00"],
                         dtype="datetime64")
        y = t.transform(times)
        assert y[1] - y[0] == pytest.approx(60.0)

    def test_numeric_forward(self):
        t = transform_time()
        y = t.transform(np.array([0.0, 60.0]))
        assert np.allclose(y, [0.0, 60.0])

    def test_inverse(self):
        t = transform_time()
        result = t.inverse(np.array([0.0, 60.0]))
        assert result.dtype.kind == "m" or result.dtype.kind == "M"

    def test_format(self):
        t = transform_time()
        result = t.format_func(np.array([0.0]))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# transform_timespan (lines 1080-1105)
# ---------------------------------------------------------------------------

class TestTransformTimespan:
    def test_secs(self):
        t = transform_timespan(unit="secs")
        x = np.array([0.0, 60.0, 3600.0])
        y = t.transform(x)
        assert np.allclose(y, x)

    def test_mins(self):
        t = transform_timespan(unit="mins")
        x = np.array([0.0, 1.0])
        y = t.transform(x)
        # 1 minute = 60 seconds, but the scale divides
        assert len(y) == 2

    def test_hours(self):
        t = transform_timespan(unit="hours")
        x = np.array([0.0, 1.0])
        y = t.transform(x)
        assert len(y) == 2

    def test_days(self):
        t = transform_timespan(unit="days")
        x = np.array([0.0, 1.0])
        y = t.transform(x)
        assert len(y) == 2

    def test_weeks(self):
        t = transform_timespan(unit="weeks")
        x = np.array([0.0, 1.0])
        y = t.transform(x)
        assert len(y) == 2

    def test_unknown_unit(self):
        with pytest.raises(ValueError):
            transform_timespan(unit="fortnights")

    def test_timedelta64_forward(self):
        t = transform_timespan(unit="secs")
        td = np.array([0, 60], dtype="timedelta64[s]")
        y = t.transform(td)
        assert np.allclose(y, [0.0, 60.0])

    def test_inverse(self):
        t = transform_timespan(unit="secs")
        result = t.inverse(np.array([0.0, 60.0]))
        assert result.dtype.kind == "m"


# ---------------------------------------------------------------------------
# transform_compose (lines 1141)
# ---------------------------------------------------------------------------

class TestTransformCompose:
    def test_two_transforms(self):
        t = transform_compose("log10", "reverse")
        x = np.array([10, 100, 1000])
        y = t.transform(x)
        x_back = t.inverse(y)
        assert np.allclose(x, x_back, atol=1e-6)
        # R names composed transforms "composition(t1,t2,...)".
        assert t.name.startswith("composition(")

    def test_empty(self):
        with pytest.raises(ValueError):
            transform_compose()


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------

class TestLegacyAliases:
    def test_trans_new(self):
        assert trans_new is new_transform

    def test_identity_trans(self):
        assert identity_trans is transform_identity

    def test_log_trans(self):
        assert log_trans is transform_log

    def test_log10_trans(self):
        assert log10_trans is transform_log10

    def test_log2_trans(self):
        assert log2_trans is transform_log2

    def test_log1p_trans(self):
        assert log1p_trans is transform_log1p

    def test_exp_trans(self):
        assert exp_trans is transform_exp

    def test_sqrt_trans(self):
        assert sqrt_trans is transform_sqrt

    def test_reverse_trans(self):
        assert reverse_trans is transform_reverse

    def test_reciprocal_trans(self):
        assert reciprocal_trans is transform_reciprocal

    def test_asinh_trans(self):
        assert asinh_trans is transform_asinh

    def test_asn_trans(self):
        assert asn_trans is transform_asn

    def test_atanh_trans(self):
        assert atanh_trans is transform_atanh

    def test_boxcox_trans(self):
        assert boxcox_trans is transform_boxcox

    def test_modulus_trans(self):
        assert modulus_trans is transform_modulus

    def test_yj_trans(self):
        assert yj_trans is transform_yj

    def test_pseudo_log_trans(self):
        assert pseudo_log_trans is transform_pseudo_log

    def test_logit_trans(self):
        assert logit_trans is transform_logit

    def test_probit_trans(self):
        assert probit_trans is transform_probit

    def test_probability_trans(self):
        assert probability_trans is transform_probability

    def test_date_trans(self):
        assert date_trans is transform_date

    def test_time_trans(self):
        assert time_trans is transform_time

    def test_timespan_trans(self):
        assert timespan_trans is transform_timespan

    def test_hms_trans(self):
        # Per R, hms_trans is an alias of transform_hms (not timespan).
        from scales.transforms import transform_hms
        assert hms_trans is transform_hms

    def test_compose_trans(self):
        assert compose_trans is transform_compose

    def test_is_trans(self):
        assert is_trans is is_transform

    def test_as_trans(self):
        assert as_trans is as_transform
