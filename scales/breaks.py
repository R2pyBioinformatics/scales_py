"""
Break generators for continuous scales.

Python port of the R scales package break generators
(https://github.com/r-lib/scales). Corresponds to:
  - R/breaks.R
  - R/breaks-retired.R

All public break generators are *closure factories*: they return a callable
that accepts ``(x, n=None)`` and returns a :class:`numpy.ndarray` of break
positions.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "breaks_extended",
    "breaks_pretty",
    "breaks_width",
    "breaks_timespan",
    "breaks_exp",
    "cbreaks",
    # Legacy aliases
    "extended_breaks",
    "pretty_breaks",
]


# ---------------------------------------------------------------------------
# Extended Wilkinson algorithm helpers
# ---------------------------------------------------------------------------

def _simplicity(q_idx: int, n_Q: int, j: int, lmin: float, lmax: float,
                lstep: float) -> float:
    """Simplicity score for a candidate labelling."""
    # Whether the label sequence includes zero
    v = 1.0 if ((lmin <= 0 <= lmax) or
                 (lmin >= 0 >= lmax)) else 0.0
    return 1.0 - (q_idx / (n_Q - 1.0)) - j + v if n_Q > 1 else 1.0 - j + v


def _simplicity_max(q_idx: int, n_Q: int, j: int) -> float:
    """Upper bound on simplicity (best-case v=1)."""
    return 1.0 - (q_idx / (n_Q - 1.0)) - j + 1.0 if n_Q > 1 else 2.0 - j


def _coverage(dmin: float, dmax: float, lmin: float, lmax: float) -> float:
    """Coverage score – how well the labels cover the data range.

    R (labeling::.coverage):
        1 - 0.5 * ((dmax - lmax)^2 + (dmin - lmin)^2) / (0.1*(dmax-dmin))^2

    Note the denominator factor is ``0.1`` — **not** ``0.5`` — which
    makes the penalty for label overshoot far more severe.  With 0.5,
    candidates that extend well beyond the data range appear almost
    as good as tight ones, causing the algorithm to prefer
    ``[0,10,20,30,40]`` over ``[10,15,...,35]`` for data in ``[9,35]``.
    """
    data_range = dmax - dmin
    if data_range < 1e-100:
        return 1.0
    tenth = 0.1 * data_range
    return (1.0
            - 0.5 * ((dmax - lmax) ** 2 + (dmin - lmin) ** 2)
            / (tenth ** 2))


def _coverage_max(dmin: float, dmax: float, span: float) -> float:
    """Upper bound on coverage for a given label span.

    R (labeling::.coverage.max) uses the same ``0.1 * range``
    denominator as :func:`_coverage`.
    """
    data_range = dmax - dmin
    if data_range < 1e-100:
        return 1.0
    if span >= data_range:
        tenth = 0.1 * data_range
        return 1.0 - 0.5 * ((span - data_range) ** 2) / (tenth ** 2)
    return 1.0


def _density(k: int, m: int, dmin: float, dmax: float,
             lmin: float, lmax: float) -> float:
    """Density score – penalty for too many or too few ticks."""
    r = (k - 1.0) / (lmax - lmin) if lmax != lmin else 1.0
    rt = (m - 1.0) / (max(lmax, dmax) - min(lmin, dmin))
    if rt == 0:
        return 1.0
    ratio = r / rt
    return 2.0 - max(ratio, 1.0 / ratio)


def _density_max(k: int, m: int) -> float:
    """Upper bound on density."""
    if k >= m:
        return 2.0 - (k - 1.0) / (m - 1.0) if m > 1 else 1.0
    return 1.0


def _legibility() -> float:
    """Legibility score (constant; formatting quality is not assessed here)."""
    return 1.0


def _extended(
    dmin: float,
    dmax: float,
    n: int = 5,
    Q: Sequence[float] = (1, 5, 2, 2.5, 4, 3),
    only_loose: bool = False,
    w: Tuple[float, float, float, float] = (0.25, 0.2, 0.5, 0.05),
) -> np.ndarray:
    """
    Wilkinson's extended algorithm for nice axis breaks.

    Parameters
    ----------
    dmin : float
        Data minimum.
    dmax : float
        Data maximum.
    n : int
        Desired number of breaks.
    Q : sequence of float
        Preference-ordered list of nice step multiples.
    only_loose : bool
        If ``True``, the returned breaks are guaranteed to enclose
        ``[dmin, dmax]``.
    w : tuple of float
        Weights for (simplicity, coverage, density, legibility).

    Returns
    -------
    numpy.ndarray
        Optimal break positions.
    """
    if dmax - dmin < 1e-10:
        return np.array([dmin])

    n_Q = len(Q)
    best_score = -2.0
    best: Optional[np.ndarray] = None

    j = 1
    while j < 50:
        for q_idx, q in enumerate(Q):
            sm = _simplicity_max(q_idx, n_Q, j)
            if (w[0] * sm + w[1] + w[2] + w[3]) < best_score:
                # Outer loop can't beat best; done with j
                j = 50  # break outer
                break

            for k in range(2, 50):
                dm = _density_max(k, n)
                if (w[0] * sm + w[1] + w[2] * dm + w[3]) < best_score:
                    break

                delta = (dmax - dmin) / (k + 1) / j / q
                base = 1.0 if delta == 0 else 10.0 ** math.floor(math.log10(delta))

                for r_mul in (1, 2, 5, 10, 20, 50, 100):
                    step = j * q * r_mul * base
                    if step < 1e-100:
                        continue

                    lmin_start = int(math.floor(dmax / step)) - (k - 1)
                    lmin_end = int(math.ceil(dmin / step))

                    for i in range(lmin_start, lmin_end + 1):
                        lmin = i * step
                        lmax = lmin + step * (k - 1)

                        if only_loose:
                            if lmin > dmin or lmax < dmax:
                                continue

                        s = _simplicity(q_idx, n_Q, j, lmin, lmax, step)
                        c = _coverage(dmin, dmax, lmin, lmax)
                        d = _density(k, n, dmin, dmax, lmin, lmax)
                        leg = _legibility()

                        score = (w[0] * s + w[1] * c
                                 + w[2] * d + w[3] * leg)

                        if score > best_score:
                            best_score = score
                            best = np.arange(lmin, lmax + step * 0.5, step)
                            # Trim to exact k labels
                            best = best[:k]
        j += 1

    if best is None:
        # Fallback: linspace
        return np.linspace(dmin, dmax, n)

    # Clean up floating-point dust
    best = np.round(best, decimals=10)
    # Remove trailing zeros artifact
    mask = np.abs(best) < 1e-14
    best[mask] = 0.0
    return best


# ---------------------------------------------------------------------------
# Pretty breaks (R's pretty() algorithm)
# ---------------------------------------------------------------------------

def _pretty(dmin: float, dmax: float, n: int = 5) -> np.ndarray:
    """
    R-style ``pretty()`` for axis breaks.

    Attempt to find a "nice" step size covering ``[dmin, dmax]`` with
    approximately *n* intervals.

    Parameters
    ----------
    dmin : float
        Data minimum.
    dmax : float
        Data maximum.
    n : int
        Desired number of intervals (not ticks).

    Returns
    -------
    numpy.ndarray
        Break positions.
    """
    if not np.isfinite(dmin) or not np.isfinite(dmax):
        return np.array([dmin, dmax])
    if dmax - dmin < 1e-10:
        return np.array([dmin])

    # R's pretty algorithm
    h = 1.5  # high
    h5 = 0.5 + 1.5 * h  # =2.75

    dx = dmax - dmin
    cell = max(abs(dmin), abs(dmax))
    # Rough cell size
    if h5 >= 1.5 * h + 0.5:
        U = 1 + (1.0 / (1 + h))
    else:
        U = 1 + (1.5 / (1 + h5))

    # Initial cell size estimate
    cell = dx / n
    if cell < 20 * 1e-07 * max(abs(dmin), abs(dmax)):
        cell = 20 * 1e-07 * max(abs(dmin), abs(dmax))

    base = 10 ** math.floor(math.log10(cell))
    unit = cell / base

    if unit < 1.5:
        step = 1.0
    elif unit < 2.5:
        step = 2.0
    elif unit < 4.0:
        step = 2.5
    elif unit < 7.5:
        step = 5.0
    else:
        step = 10.0

    step *= base
    lo = step * math.floor(dmin / step)
    hi = step * math.ceil(dmax / step)

    # Nudge to include boundaries
    if lo > dmin:
        lo -= step
    if hi < dmax:
        hi += step

    result = np.arange(lo, hi + step * 0.5, step)
    # Clean up floating-point dust
    result = np.round(result, decimals=10)
    mask = np.abs(result) < 1e-14
    result[mask] = 0.0
    return result


# ---------------------------------------------------------------------------
# Public break generators
# ---------------------------------------------------------------------------

def breaks_extended(
    n: int = 5,
    *,
    Q: Sequence[float] = (1, 5, 2, 2.5, 4, 3),
    only_loose: bool = False,
) -> Callable[[ArrayLike, Optional[int]], np.ndarray]:
    """
    Create a break function using Wilkinson's extended algorithm.

    Parameters
    ----------
    n : int, optional
        Desired number of breaks (default 5).
    Q : sequence of float, optional
        Preference-ordered list of "nice" step multiples
        (default ``(1, 5, 2, 2.5, 4, 3)``).
    only_loose : bool, optional
        If ``True``, the returned breaks are guaranteed to enclose the
        data range (default ``False``).

    Returns
    -------
    callable
        A function ``(x, n=None) -> numpy.ndarray`` that computes break
        positions for data *x*.

    Examples
    --------
    >>> brk = breaks_extended(n=5)
    >>> brk([1.3, 9.8])
    array([ 0.,  2.,  4.,  6.,  8., 10.])
    """

    def breaks_fn(x: ArrayLike, n_: Optional[int] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.array([])
        dmin, dmax = float(x.min()), float(x.max())
        k = n_ if n_ is not None else n
        return _extended(dmin, dmax, n=k, Q=Q, only_loose=only_loose)

    return breaks_fn


def breaks_pretty(n: int = 5) -> Callable[[ArrayLike, Optional[int]], np.ndarray]:
    """
    Create a break function using R's ``pretty()`` algorithm.

    Parameters
    ----------
    n : int, optional
        Desired number of breaks (default 5).

    Returns
    -------
    callable
        A function ``(x, n=None) -> numpy.ndarray`` that computes break
        positions for data *x*.

    Examples
    --------
    >>> brk = breaks_pretty(n=5)
    >>> brk([0.5, 9.3])
    array([ 0.,  2.,  4.,  6.,  8., 10.])
    """

    def breaks_fn(x: ArrayLike, n_: Optional[int] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.array([])
        dmin, dmax = float(x.min()), float(x.max())
        k = n_ if n_ is not None else n
        return _pretty(dmin, dmax, n=k)

    return breaks_fn


def breaks_width(
    width: float,
    offset: float = 0,
) -> Callable[[ArrayLike, Optional[int]], np.ndarray]:
    """
    Create a break function with fixed-width intervals.

    Parameters
    ----------
    width : float
        Distance between consecutive breaks.
    offset : float, optional
        Shift all breaks by this amount (default 0).

    Returns
    -------
    callable
        A function ``(x, n=None) -> numpy.ndarray`` that computes break
        positions for data *x*.

    Examples
    --------
    >>> brk = breaks_width(width=0.5)
    >>> brk([0.1, 2.4])
    array([0. , 0.5, 1. , 1.5, 2. , 2.5])
    """
    if width <= 0:
        raise ValueError("`width` must be positive")

    def breaks_fn(x: ArrayLike, n_: Optional[int] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.array([])
        dmin, dmax = float(x.min()), float(x.max())

        # Shift by offset, compute grid, shift back
        lo = math.floor((dmin - offset) / width) * width + offset
        hi = math.ceil((dmax - offset) / width) * width + offset

        result = np.arange(lo, hi + width * 0.5, width)
        # Clean up floating-point dust
        result = np.round(result, decimals=10)
        return result

    return breaks_fn


_TIMESPAN_UNITS = {
    "secs": 1,
    "mins": 60,
    "hours": 3600,
    "days": 86400,
    "weeks": 604800,
}


def breaks_timespan(
    unit: str = "secs",
    n: int = 5,
) -> Callable[[ArrayLike, Optional[int]], np.ndarray]:
    """
    Create a break function for timespan (duration) data.

    The data are assumed to be in seconds; breaks are placed at multiples
    of the chosen *unit*.

    Parameters
    ----------
    unit : str, optional
        One of ``"secs"``, ``"mins"``, ``"hours"``, ``"days"``,
        ``"weeks"`` (default ``"secs"``).
    n : int, optional
        Desired number of breaks (default 5).

    Returns
    -------
    callable
        A function ``(x, n=None) -> numpy.ndarray`` that computes break
        positions for data *x*.

    Raises
    ------
    ValueError
        If *unit* is not one of the recognised values.

    Examples
    --------
    >>> brk = breaks_timespan(unit="mins", n=5)
    >>> brk([0, 7200])
    array([   0.,   60.,  120., ...])
    """
    if unit not in _TIMESPAN_UNITS:
        raise ValueError(
            f"Unknown unit {unit!r}. Choose from: "
            f"{', '.join(_TIMESPAN_UNITS)}"
        )
    multiplier = _TIMESPAN_UNITS[unit]

    def breaks_fn(x: ArrayLike, n_: Optional[int] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.array([])
        dmin, dmax = float(x.min()), float(x.max())
        k = n_ if n_ is not None else n

        # Scale to unit, compute pretty breaks, scale back
        scaled_min = dmin / multiplier
        scaled_max = dmax / multiplier
        brks = _pretty(scaled_min, scaled_max, n=k)
        return brks * multiplier

    return breaks_fn


def breaks_exp(
    n: int = 5,
) -> Callable[[ArrayLike, Optional[int]], np.ndarray]:
    """
    Create a break function suitable for exponential transformations.

    For data spanning several orders of magnitude the breaks are placed
    at ``0`` plus the last ``n - 1`` integer powers of 10.  For data with
    a smaller range, falls back to :func:`breaks_extended`.

    Parameters
    ----------
    n : int, optional
        Desired number of breaks (default 5).

    Returns
    -------
    callable
        A function ``(x, n=None) -> numpy.ndarray`` that computes break
        positions for data *x*.

    Examples
    --------
    >>> brk = breaks_exp(n=4)
    >>> brk([0.01, 1000])
    array([0.e+00, 1.e+01, 1.e+02, 1.e+03])
    """

    def breaks_fn(x: ArrayLike, n_: Optional[int] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.array([])
        dmin, dmax = float(x.min()), float(x.max())
        k = n_ if n_ is not None else n

        # If the range is large (multiple orders of magnitude), use
        # powers of 10.
        if dmax > 0 and dmin >= 0:
            log_max = math.log10(max(dmax, 1e-100))
            log_min = math.log10(max(dmin, 1e-100))
            order_span = log_max - log_min
        else:
            order_span = 0

        if order_span >= 2:
            # Use powers of 10
            max_power = int(math.ceil(log_max))
            # Take the last (k-1) integer powers, plus 0
            powers = list(range(max(0, max_power - k + 1), max_power + 1))
            brks = [0.0] + [10.0 ** p for p in powers]
            # Trim to k breaks
            brks = brks[-(k):]
            return np.array(brks)

        # Fall back to extended breaks for small ranges
        return _extended(dmin, dmax, n=k)

    return breaks_fn


def cbreaks(
    x: ArrayLike,
    breaks_fun: Optional[Callable] = None,
    labels_fun: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    Comprehensive breaks (deprecated).

    .. deprecated:: 0.1.0
        Use the specific break generators directly instead.

    Parameters
    ----------
    x : array-like
        Data range (length-2 vector ``[min, max]``).
    breaks_fun : callable, optional
        Break function.  Defaults to :func:`breaks_extended` ``()``.
    labels_fun : callable, optional
        Label function.  If ``None``, labels are the string
        representation of breaks.

    Returns
    -------
    dict
        Dictionary with keys ``"breaks"`` and ``"labels"``.
    """
    warnings.warn(
        "cbreaks() is deprecated. Use the specific break generators directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    if breaks_fun is None:
        breaks_fun = breaks_extended()

    brks = breaks_fun(x)

    if labels_fun is not None:
        labels = labels_fun(brks)
    else:
        labels = [str(b) for b in brks]

    return {"breaks": brks, "labels": labels}


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------

#: Legacy alias for :func:`breaks_extended`.
extended_breaks = breaks_extended

#: Legacy alias for :func:`breaks_pretty`.
pretty_breaks = breaks_pretty
