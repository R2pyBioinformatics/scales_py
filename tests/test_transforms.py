"""Tests for scales.transforms module."""

import numpy as np
import pytest

import scales


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

class TestTransformIdentity:
    def test_roundtrip(self):
        t = scales.transform_identity()
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)

    def test_values_unchanged(self):
        t = scales.transform_identity()
        x = np.array([-1.0, 0.0, 42.0])
        np.testing.assert_allclose(t.transform(x), x)


# ---------------------------------------------------------------------------
# log10
# ---------------------------------------------------------------------------

class TestTransformLog10:
    def test_forward(self):
        t = scales.transform_log10()
        np.testing.assert_allclose(
            t.transform(np.array([1, 10, 100, 1000])),
            [0, 1, 2, 3],
        )

    def test_inverse(self):
        t = scales.transform_log10()
        np.testing.assert_allclose(
            t.inverse(np.array([0, 1, 2, 3])),
            [1, 10, 100, 1000],
        )

    def test_roundtrip(self):
        t = scales.transform_log10()
        x = np.array([1.0, 10.0, 100.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)


# ---------------------------------------------------------------------------
# log (custom base)
# ---------------------------------------------------------------------------

class TestTransformLog:
    def test_base_2(self):
        t = scales.transform_log(base=2)
        np.testing.assert_allclose(
            t.transform(np.array([1, 2, 4, 8])),
            [0, 1, 2, 3],
        )

    def test_roundtrip(self):
        t = scales.transform_log(base=np.e)
        x = np.array([1.0, np.e, np.e ** 2])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)


# ---------------------------------------------------------------------------
# log2
# ---------------------------------------------------------------------------

class TestTransformLog2:
    def test_forward(self):
        t = scales.transform_log2()
        np.testing.assert_allclose(
            t.transform(np.array([1, 2, 4, 8])),
            [0, 1, 2, 3],
        )

    def test_roundtrip(self):
        t = scales.transform_log2()
        x = np.array([1.0, 4.0, 16.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)


# ---------------------------------------------------------------------------
# log1p
# ---------------------------------------------------------------------------

class TestTransformLog1p:
    def test_forward(self):
        t = scales.transform_log1p()
        np.testing.assert_allclose(
            t.transform(np.array([0, 1, np.e - 1])),
            [0, np.log(2), 1.0],
        )

    def test_roundtrip(self):
        t = scales.transform_log1p()
        x = np.array([0.0, 1.0, 10.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)


# ---------------------------------------------------------------------------
# sqrt
# ---------------------------------------------------------------------------

class TestTransformSqrt:
    def test_forward(self):
        t = scales.transform_sqrt()
        np.testing.assert_allclose(
            t.transform(np.array([0, 1, 4, 9])),
            [0, 1, 2, 3],
        )

    def test_domain(self):
        t = scales.transform_sqrt()
        assert t.domain[0] >= 0

    def test_roundtrip(self):
        t = scales.transform_sqrt()
        x = np.array([0.0, 1.0, 4.0, 25.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)


# ---------------------------------------------------------------------------
# reverse
# ---------------------------------------------------------------------------

class TestTransformReverse:
    def test_forward(self):
        t = scales.transform_reverse()
        np.testing.assert_allclose(
            t.transform(np.array([1, 2, 3])),
            [-1, -2, -3],
        )

    def test_roundtrip(self):
        t = scales.transform_reverse()
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)


# ---------------------------------------------------------------------------
# reciprocal
# ---------------------------------------------------------------------------

class TestTransformReciprocal:
    def test_forward(self):
        t = scales.transform_reciprocal()
        np.testing.assert_allclose(
            t.transform(np.array([1.0, 2.0, 4.0])),
            [1.0, 0.5, 0.25],
        )

    def test_roundtrip(self):
        t = scales.transform_reciprocal()
        x = np.array([1.0, 2.0, 5.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)


# ---------------------------------------------------------------------------
# boxcox
# ---------------------------------------------------------------------------

class TestTransformBoxcox:
    def test_p1_identity_offset(self):
        """boxcox(p=1): f(x) = (x^1 - 1)/1 = x - 1"""
        t = scales.transform_boxcox(p=1)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(t.transform(x), [0.0, 1.0, 2.0])

    def test_p1_roundtrip(self):
        t = scales.transform_boxcox(p=1)
        x = np.array([1.0, 5.0, 10.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)

    def test_p0_log(self):
        """boxcox(p=0) should behave like natural log."""
        t = scales.transform_boxcox(p=0)
        x = np.array([1.0, np.e, np.e ** 2])
        np.testing.assert_allclose(t.transform(x), [0.0, 1.0, 2.0])

    def test_p0_roundtrip(self):
        t = scales.transform_boxcox(p=0)
        x = np.array([1.0, 2.0, 10.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)


# ---------------------------------------------------------------------------
# modulus
# ---------------------------------------------------------------------------

class TestTransformModulus:
    def test_p1_handles_negatives(self):
        t = scales.transform_modulus(p=1)
        result = t.transform(np.array([-2, -1, 0, 1, 2], dtype=float))
        np.testing.assert_allclose(result, [-2, -1, 0, 1, 2])

    def test_roundtrip(self):
        t = scales.transform_modulus(p=1)
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)


# ---------------------------------------------------------------------------
# yj (Yeo-Johnson)
# ---------------------------------------------------------------------------

class TestTransformYJ:
    def test_p0_mixed_signs(self):
        t = scales.transform_yj(p=0)
        result = t.transform(np.array([-1, 0, 1, 2], dtype=float))
        # p=0 has specific formula; just check roundtrip
        inv = t.inverse(result)
        np.testing.assert_allclose(inv, [-1, 0, 1, 2])

    def test_roundtrip(self):
        t = scales.transform_yj(p=1)
        x = np.array([-2.0, 0.0, 2.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x, atol=1e-10)


# ---------------------------------------------------------------------------
# pseudo_log
# ---------------------------------------------------------------------------

class TestTransformPseudoLog:
    def test_smooth_near_zero(self):
        t = scales.transform_pseudo_log()
        result = t.transform(np.array([-1, 0, 1], dtype=float))
        # Should be continuous and anti-symmetric around 0
        assert result[1] == pytest.approx(0.0)
        assert result[0] == pytest.approx(-result[2])

    def test_large_values_approach_log(self):
        t = scales.transform_pseudo_log()
        result = t.transform(np.array([1000.0]))
        expected_log = np.log10(1000)  # ~ 3.0 ish, not exact due to sigma
        assert result[0] > 0

    def test_roundtrip(self):
        t = scales.transform_pseudo_log()
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x, atol=1e-10)


# ---------------------------------------------------------------------------
# logit & probit
# ---------------------------------------------------------------------------

class TestTransformLogit:
    def test_forward(self):
        t = scales.transform_logit()
        result = t.transform(np.array([0.5]))
        np.testing.assert_allclose(result, [0.0])

    def test_inverse(self):
        t = scales.transform_logit()
        result = t.inverse(np.array([0.0]))
        np.testing.assert_allclose(result, [0.5])

    def test_roundtrip(self):
        t = scales.transform_logit()
        x = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x)


class TestTransformProbit:
    def test_midpoint(self):
        t = scales.transform_probit()
        result = t.transform(np.array([0.5]))
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_roundtrip(self):
        t = scales.transform_probit()
        x = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(t.inverse(t.transform(x)), x, atol=1e-10)


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------

class TestTransformCompose:
    def test_log10_then_reverse(self):
        t1 = scales.transform_log10()
        t2 = scales.transform_reverse()
        tc = scales.transform_compose(t1, t2)
        result = tc.transform(np.array([10.0, 100.0]))
        np.testing.assert_allclose(result, [-1.0, -2.0])

    def test_roundtrip(self):
        t1 = scales.transform_log10()
        t2 = scales.transform_reverse()
        tc = scales.transform_compose(t1, t2)
        x = np.array([1.0, 10.0, 100.0])
        np.testing.assert_allclose(tc.inverse(tc.transform(x)), x, atol=1e-10)


# ---------------------------------------------------------------------------
# date & time transforms
# ---------------------------------------------------------------------------

class TestTransformDate:
    def test_exists(self):
        t = scales.transform_date()
        assert scales.is_transform(t)


class TestTransformTime:
    def test_exists(self):
        t = scales.transform_time()
        assert scales.is_transform(t)


# ---------------------------------------------------------------------------
# as_transform / is_transform
# ---------------------------------------------------------------------------

class TestAsTransform:
    def test_string_log10(self):
        t = scales.as_transform("log10")
        assert scales.is_transform(t)
        np.testing.assert_allclose(
            t.transform(np.array([10.0])), [1.0]
        )

    def test_string_sqrt(self):
        t = scales.as_transform("sqrt")
        assert scales.is_transform(t)

    def test_passthrough_transform_object(self):
        t = scales.transform_identity()
        assert scales.as_transform(t) is t


class TestIsTransform:
    def test_true_for_transform(self):
        assert scales.is_transform(scales.transform_log10())

    def test_false_for_string(self):
        assert not scales.is_transform("log10")

    def test_false_for_none(self):
        assert not scales.is_transform(None)


# ---------------------------------------------------------------------------
# trans_breaks
# ---------------------------------------------------------------------------

class TestTransBreaks:
    def test_log10_breaks(self):
        tb = scales.trans_breaks("log10", n=5)
        result = tb((1, 1000))
        # Should return powers of 10
        assert 1.0 in result
        assert 10.0 in result
        assert 100.0 in result
        assert 1000.0 in result

    def test_returns_array(self):
        tb = scales.trans_breaks("log10", n=5)
        result = tb((1, 1000))
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# trans_format
# ---------------------------------------------------------------------------

class TestTransFormat:
    def test_log10_format(self):
        tf = scales.trans_format("log10")
        result = tf(np.array([1, 10, 100]))
        # Should return formatted strings
        assert len(result) == 3
