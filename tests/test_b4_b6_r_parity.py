"""Regression tests for B4 + B6 — R-parity for discrete training & transform derivatives.

B4 — ``train_discrete`` previously
  * had no ``fct`` parameter (R's 3-state ``fct = NA`` factor control);
  * preserved the *insertion* order of plain character / numeric ``new`` instead of
    sorting like R's ``clevels``;
  * sorted a ``Categorical`` ``existing`` instead of respecting its category order.

R semantics (scale-discrete.R:30-97):
  * ``clevels(new)`` sorts non-factor uniques but preserves factor levels;
  * ``fct = NA`` (default) — auto-detect: factor → preserve, else sort;
  * ``fct = TRUE`` — promote a *character existing* to a factor with its current
    order as the level set (only affects existing, **not** new);
  * ``fct = FALSE`` — no effect on existing-as-character (still sorted).

B6 — six transforms were missing the ``d_transform`` / ``d_inverse`` derivatives
that R sets on the transformer object: ``asn``, ``exp``, ``modulus``,
``yeo_johnson``, ``pseudo_log``, ``probit``.  Without them, downstream code that
walks the transform chain (e.g. ``trans_new`` consumers, density / area
geoms in ggplot2-python) silently lost rescaling fidelity for these scales.
Closed-form derivatives are now provided, cross-validated against R to the
machine-precision shown below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# B4 — train_discrete fct parameter + clevels parity
# ---------------------------------------------------------------------------

class TestB4_TrainDiscreteFct:
    """fct three-state matches R exactly across 8 R-validated combinations."""

    def test_char_new_is_sorted_regardless_of_fct(self):
        # R: train_discrete(c("c","a","b")) → "a b c"
        #    train_discrete(c("c","a","b"), fct=TRUE) → "a b c"
        #    train_discrete(c("c","a","b"), fct=FALSE) → "a b c"
        # ``fct`` only modulates *existing*; ``clevels(new)`` always sorts.
        from scales import train_discrete
        assert list(train_discrete(["c", "a", "b"])) == ["a", "b", "c"]
        assert list(train_discrete(["c", "a", "b"], fct=True)) == ["a", "b", "c"]
        assert list(train_discrete(["c", "a", "b"], fct=False)) == ["a", "b", "c"]

    def test_factor_new_preserves_level_order(self):
        # R: train_discrete(factor(c("c","a","b"), levels=c("c","a","b"))) → "c a b"
        from scales import train_discrete
        fac = pd.Categorical(["c", "a", "b"], categories=["c", "a", "b"])
        assert list(train_discrete(fac)) == ["c", "a", "b"]

    def test_merge_factor_existing_preserves_factor_order(self):
        # R: train_discrete("d", existing=factor(...,levels=c("c","a","b"))) → "c a b d"
        # Was the second-pass bug: Py was sorting Categorical existing → "a b c d".
        from scales import train_discrete
        fac = pd.Categorical(["c", "a", "b"], categories=["c", "a", "b"])
        assert list(train_discrete(["d"], existing=fac)) == ["c", "a", "b", "d"]

    def test_merge_char_existing_sorts_existing(self):
        # R: train_discrete("d", existing=c("a","b","c")) → "a b c d"
        from scales import train_discrete
        assert list(train_discrete(["d"], existing=["a", "b", "c"])) == [
            "a", "b", "c", "d"
        ]

    def test_fct_true_promotes_char_existing_to_factor(self):
        # R: train_discrete("d", existing=c("c","a","b"), fct=TRUE) → "c a b d"
        # ``fct=TRUE`` treats list-as-given as the factor's level order.
        from scales import train_discrete
        assert list(
            train_discrete(["d"], existing=["c", "a", "b"], fct=True)
        ) == ["c", "a", "b", "d"]

    def test_fct_false_on_factor_existing_still_clevels(self):
        # R: train_discrete("d", existing=factor(c("c","a","b"), levels=...),
        #                   fct=FALSE) → "c a b d"
        # ``fct=FALSE`` does NOT downgrade a real factor — clevels still wins.
        from scales import train_discrete
        fac = pd.Categorical(["c", "a", "b"], categories=["c", "a", "b"])
        assert list(train_discrete(["d"], existing=fac, fct=False)) == [
            "c", "a", "b", "d"
        ]

    def test_none_new_returns_existing(self):
        # R: train_discrete(NULL, existing=c("a","b")) → c("a","b")
        from scales import train_discrete
        assert list(train_discrete(None, existing=["a", "b"])) == ["a", "b"]

    def test_na_placement_last(self):
        # R: train_discrete(c("a", NA, "b")) → "a b NA" (NA sorted last)
        from scales import train_discrete
        for inp in [["a", None, "b"], ["b", None, "a"], [None, "b", "a"]]:
            out = train_discrete(inp)
            assert [v for v in out if v is not None] == ["a", "b"]
            assert out[-1] is None  # NA placed last

    def test_na_rm_drops_nas(self):
        from scales import train_discrete
        out = train_discrete(["a", None, "b"], na_rm=True)
        assert None not in out
        assert list(out) == ["a", "b"]

    def test_merge_na_in_existing_placement(self):
        # R: train_discrete("d", existing=c("a", NA, "b")) → "a b d NA"
        # Final union goes through ``sort(range, na.last = TRUE)`` so
        # NA always ends up last regardless of where it sat in existing.
        from scales import train_discrete
        out = train_discrete(["d"], existing=["a", None, "b"])
        assert [v for v in out if v is not None] == ["a", "b", "d"]
        assert out[-1] is None  # NA placed last after sort


# ---------------------------------------------------------------------------
# B6 — Transform derivatives (d_transform + d_inverse) cross-validated to R
# ---------------------------------------------------------------------------

class TestB6_AsnDerivatives:
    """``asin(sqrt(x))`` — d/dx = 1/sqrt(x - x²); d⁻¹ = sin(y)·cos(y) = sin(2y)/2."""

    def test_d_transform_matches_R(self):
        from scales.transforms import transform_asn
        # R: transform_asn()$d_transform(c(0.1, 0.5, 0.9))
        #    → 3.333333 2 3.333333
        t = transform_asn()
        np.testing.assert_allclose(
            t.d_transform(np.array([0.1, 0.5, 0.9])),
            [3.333333, 2.0, 3.333333],
            rtol=1e-6,
        )

    def test_d_inverse_matches_R(self):
        from scales.transforms import transform_asn
        t = transform_asn()
        # R: transform_asn()$d_inverse(transform_asn()$transform(c(0.1, 0.5, 0.9)))
        #    → 0.3 0.5 0.3
        y = t.transform(np.array([0.1, 0.5, 0.9]))
        np.testing.assert_allclose(
            t.d_inverse(y), [0.3, 0.5, 0.3], rtol=1e-6
        )


class TestB6_ExpDerivatives:
    """``base^x`` — d/dx = ln(base)·base^x; d⁻¹ = 1/(y·ln(base))."""

    def test_d_transform_base2(self):
        from scales.transforms import transform_exp
        # R: transform_exp(2)$d_transform(c(1,2,3))
        #    → 1.386294 2.772589 5.545177
        t = transform_exp(2)
        np.testing.assert_allclose(
            t.d_transform(np.array([1.0, 2.0, 3.0])),
            [1.386294, 2.772589, 5.545177],
            rtol=1e-6,
        )

    def test_d_inverse_base2(self):
        from scales.transforms import transform_exp
        t = transform_exp(2)
        # R reference: 0.7213475 0.3606738 0.1803369
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            t.d_inverse(t.transform(x)),
            [0.7213475, 0.3606738, 0.1803369],
            rtol=1e-6,
        )


class TestB6_ModulusDerivatives:
    """Modulus transform — 2-branch on |p|<eps to match R's ``abs(p)<1e-7``."""

    def test_p_half_d_transform_matches_R(self):
        from scales.transforms import transform_modulus
        # R: transform_modulus(0.5)$d_transform(c(1,2,3))
        #    → 0.7071068 0.5773503 0.5
        t = transform_modulus(0.5)
        np.testing.assert_allclose(
            t.d_transform(np.array([1.0, 2.0, 3.0])),
            [0.7071068, 0.5773503, 0.5],
            rtol=1e-6,
        )

    def test_p_zero_falls_back_to_log_branch(self):
        from scales.transforms import transform_modulus
        # R: transform_modulus(0)$d_transform(c(1,2,3))
        #    → 0.5 0.3333333 0.25  (= 1/(|x|+1))
        t = transform_modulus(0)
        np.testing.assert_allclose(
            t.d_transform(np.array([1.0, 2.0, 3.0])),
            [0.5, 0.3333333, 0.25],
            rtol=1e-6,
        )


class TestB6_YeoJohnsonDerivatives:
    """Yeo-Johnson — 4-piecewise (x≥0 vs x<0) × (p==0 vs general)."""

    def test_yj_p_half(self):
        from scales.transforms import transform_yj
        # R: transform_yj(0.5)$d_transform(c(-1, 0, 1, 2))
        #    → 1.414214 1 0.7071068 0.5773503
        t = transform_yj(0.5)
        np.testing.assert_allclose(
            t.d_transform(np.array([-1.0, 0.0, 1.0, 2.0])),
            [1.414214, 1.0, 0.7071068, 0.5773503],
            rtol=1e-6,
        )

    def test_yj_at_zero_evaluates_to_one(self):
        """R limit: yj derivative at x=0 is exactly 1 for any p (continuity)."""
        from scales.transforms import transform_yj
        for p in [-1.0, 0.0, 0.5, 1.0, 2.0]:
            t = transform_yj(p)
            d = float(t.d_transform(np.array([0.0]))[0])
            assert abs(d - 1.0) < 1e-12, f"yj({p}) at 0 = {d}, expected 1.0"

    def test_yj_p_zero(self):
        from scales.transforms import transform_yj
        # R: transform_yj(0)$d_transform(c(-1, 0, 1))
        #    → 2 1 0.5
        t = transform_yj(0)
        np.testing.assert_allclose(
            t.d_transform(np.array([-1.0, 0.0, 1.0])),
            [2.0, 1.0, 0.5],
            rtol=1e-6,
        )

    def test_yj_p_two(self):
        from scales.transforms import transform_yj
        # R: transform_yj(2)$d_transform(c(-1, 0, 1))
        #    → 0.5 1 2
        t = transform_yj(2)
        np.testing.assert_allclose(
            t.d_transform(np.array([-1.0, 0.0, 1.0])),
            [0.5, 1.0, 2.0],
            rtol=1e-6,
        )


class TestB6_PseudoLogDerivatives:
    """``asinh(x / (2σ)) / log(base)`` — d/dx = 1 / (log(base)·√(x²+4σ²))."""

    def test_d_transform_matches_R(self):
        from scales.transforms import transform_pseudo_log
        # R: transform_pseudo_log()$d_transform(c(-1, 0, 1, 100))
        #    → 0.4472136 0.5 0.4472136 0.009998
        t = transform_pseudo_log()
        np.testing.assert_allclose(
            t.d_transform(np.array([-1.0, 0.0, 1.0, 100.0])),
            [0.4472136, 0.5, 0.4472136, 0.009998],
            rtol=1e-4,
        )


class TestB6_ProbitDerivatives:
    """Probit = transform_probability("norm") — derivative = 1/φ(Φ⁻¹(x))."""

    def test_d_transform_matches_R(self):
        from scales.transforms import transform_probit
        # R: transform_probit()$d_transform(c(0.1, 0.5, 0.9))
        #    → 5.69805986 2.50662827 5.69805986
        t = transform_probit()
        np.testing.assert_allclose(
            t.d_transform(np.array([0.1, 0.5, 0.9])),
            [5.69805986, 2.50662827, 5.69805986],
            rtol=1e-6,
        )

    def test_d_inverse_matches_R(self):
        from scales.transforms import transform_probit
        t = transform_probit()
        y = t.transform(np.array([0.1, 0.5, 0.9]))
        # R: 0.17549833 0.39894228 0.17549833
        np.testing.assert_allclose(
            t.d_inverse(y),
            [0.17549833, 0.39894228, 0.17549833],
            rtol=1e-6,
        )
