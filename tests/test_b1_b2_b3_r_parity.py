"""Regression tests for B1 + B2 + B3 — R-parity small fixes.

B1 — ``label_scientific(...)(0)`` returns ``"0e+00"`` in R, regardless
of ``digits`` or ``trim``. Python previously returned bare ``"0"``
(or ``"0.00e+00"`` with ``trim=False``), so a 0 tick on a scientific
axis printed differently from its neighbours.

B2 — ``log_breaks(...)((1e-5, 1e-4))`` raised ``ValueError`` because
numpy refuses ``int ** np.array([-5, -4, ...])`` (negative integer
powers require a float base). R returns ``[1e-05, 3e-05, 1e-04, 3e-04]``
without complaint. Coerced ``base = float(base)`` at the public and
internal entry points so any sub-unary range works.

B3 — ``train_continuous([])`` raised ``ValueError`` in Python while R's
``scale-continuous.R:44-47`` returns the *existing* range (or ``NULL``)
unchanged. The function now returns ``existing`` on empty / ``None``
input, matching the silent-no-op behaviour of ``ContinuousRange.train``
and aligning with R.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


class TestB1_LabelScientificZero:
    @pytest.mark.parametrize("digits", [1, 3, 5])
    @pytest.mark.parametrize("trim", [True, False])
    def test_zero_always_0e00(self, digits, trim):
        """R: label_scientific(digits=d, trim=t)(0) → "0e+00" for any d, t."""
        from scales import label_scientific
        out = label_scientific(digits=digits, trim=trim)(0)
        assert out == ["0e+00"]

    def test_nonzero_unaffected(self):
        from scales import label_scientific
        # 1.23e+05 etc. still uses normal scientific formatting.
        # Default trim=True strips trailing zeros — matches R:
        #   R: label_scientific()(123456) → "1.23e+05"
        #   R: label_scientific()(-1e-7) → "-1e-07"
        assert label_scientific(digits=3)(123456) == ["1.23e+05"]
        assert label_scientific()(-1e-7) == ["-1e-07"]

    def test_mixed_vector_with_zero(self):
        from scales import label_scientific
        # R: label_scientific()(c(0, 1e3, 1e-3)) → c("0e+00", "1e+03", "1e-03")
        # (default trim=TRUE drops trailing-zero coeff to bare "1")
        out = label_scientific()([0, 1e3, 1e-3])
        assert out == ["0e+00", "1e+03", "1e-03"]


class TestB2_LogBreaksNegativePowers:
    @pytest.mark.parametrize("rng", [
        (1e-5, 1e-4),
        (1e-10, 1e-8),
        (1e-100, 1e-50),
    ])
    def test_subunary_ranges_no_crash(self, rng):
        """R produces breaks for these ranges; Py previously crashed."""
        from scales import log_breaks
        out = log_breaks()(rng)
        assert len(out) > 0
        assert np.all(np.isfinite(out))

    def test_decade_aligned_ranges_match_R(self):
        """Decade-aligned ranges: powers-of-base directly. Verified
        byte-equivalent to R."""
        from scales import log_breaks
        # rtol 1e-10 absorbs numpy's int**float epsilon (1e-5 ↔
        # 9.999999999999999e-06) without masking real divergence
        np.testing.assert_allclose(log_breaks()((1, 1000)),
                                    [1.0, 10.0, 100.0, 1000.0], rtol=1e-10)
        np.testing.assert_allclose(log_breaks()((1, 1e6)),
                                    [1.0, 100.0, 10000.0, 1e6], rtol=1e-10)
        # Negative-power decade alignment — the original B2 crash
        # trigger. Verifies (a) no crash and (b) R-faithful count of 3.
        np.testing.assert_allclose(log_breaks()((1e-5, 1)),
                                    [1e-5, 1e-3, 1e-1], rtol=1e-10)
        np.testing.assert_allclose(log_breaks()((0.01, 100)),
                                    [0.01, 0.1, 1.0, 10.0, 100.0], rtol=1e-10)

    def test_explicit_float_base(self):
        """``base=2.0`` already worked; ``base=2`` (int) should now
        also work without crashing on negative powers."""
        from scales import log_breaks
        out = log_breaks(base=2)((0.125, 8))
        assert len(out) > 0

    def test_positive_range_unaffected(self):
        from scales import log_breaks
        out = log_breaks()((1, 1000))
        assert list(out) == [1.0, 10.0, 100.0, 1000.0]


class TestB3_TrainContinuousEmpty:
    def test_empty_with_no_existing_returns_None(self):
        """R: train_continuous(NULL) → NULL."""
        from scales import train_continuous
        assert train_continuous([]) is None
        assert train_continuous(None) is None

    def test_empty_with_existing_returns_existing(self):
        """R: train_continuous(NULL, c(0, 10)) → c(0, 10)."""
        from scales import train_continuous
        assert train_continuous([], existing=(0.0, 10.0)) == (0.0, 10.0)
        assert train_continuous(None, existing=(-5.0, 5.0)) == (-5.0, 5.0)

    def test_all_nonfinite_returns_existing(self):
        """NaN/Inf-only data is equivalent to empty after the filter."""
        from scales import train_continuous
        assert train_continuous([math.nan, math.inf, -math.inf]) is None
        assert train_continuous(
            [math.nan], existing=(1.0, 2.0)
        ) == (1.0, 2.0)

    def test_normal_path_unaffected(self):
        from scales import train_continuous
        assert train_continuous([1, 5, 3]) == (1.0, 5.0)
        assert train_continuous([0, 4], existing=(1.0, 5.0)) == (0.0, 5.0)

    def test_consistency_with_continuous_range_class(self):
        """``ContinuousRange.train([])`` already silent on empty —
        ensure the function-level helper matches."""
        from scales import ContinuousRange, train_continuous
        r = ContinuousRange()
        r.train([])
        assert r.range is None
        assert train_continuous([]) == r.range  # both None
