"""
Break generators for log-scaled axes.

Python port of ``R/breaks-log.R`` from the R *scales* package
(https://github.com/r-lib/scales).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "breaks_log",
    "minor_breaks_log",
]


def breaks_log(
    n: int = 5,
    base: float = 10,
) -> Callable[[ArrayLike], np.ndarray]:
    """Create a break generator for log-scaled axes.

    Returns a function that accepts a numeric range and produces
    "nice" log-spaced breaks (integer powers of *base* that span the
    data range, potentially with in-between values when the range is
    narrow).

    Parameters
    ----------
    n : int, optional
        Target number of breaks (default 5).
    base : float, optional
        Logarithm base (default 10).

    Returns
    -------
    callable
        A function ``(x) -> numpy.ndarray`` of break positions.

    Examples
    --------
    >>> brk = breaks_log(n=5, base=10)
    >>> brk([1, 10000])
    array([1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04])
    """

    def _breaks(x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        if len(x) == 0:
            return np.array([], dtype=float)

        rng = (float(np.min(x)), float(np.max(x)))

        # Integer powers of base that span the range
        log_rng = (
            math.floor(math.log(rng[0], base)),
            math.ceil(math.log(rng[1], base)),
        )

        # Generate candidate breaks as powers of base
        powers = np.arange(log_rng[0], log_rng[1] + 1)
        breaks = base ** powers

        # If we have too few breaks, fill in between powers
        if len(breaks) < n:
            breaks = _fill_log_breaks(log_rng, base, n)

        # If we have too many, thin them out
        if len(breaks) > 2 * n:
            # Keep every k-th break to get close to n
            k = max(1, len(breaks) // n)
            breaks = breaks[::k]

        return breaks

    return _breaks


def _fill_log_breaks(
    log_rng: tuple[int, int],
    base: float,
    n: int,
) -> np.ndarray:
    """Generate denser log-spaced breaks when integer powers are too few.

    Fills in sub-decade values (e.g. 2, 5 for base 10) between
    integer powers of *base*.
    """
    # For base 10, nice sub-decade multipliers
    if base == 10:
        multipliers_options = [
            [1],
            [1, 3],
            [1, 2, 5],
            [1, 2, 3, 5],
            [1, 2, 3, 5, 7],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
        ]
    elif base == 2:
        multipliers_options = [
            [1],
            [1, 1.5],
        ]
    else:
        multipliers_options = [
            [1],
            [1, base / 2],
            [1, base / 3, 2 * base / 3],
        ]

    # Try each set of multipliers, pick the one closest to n
    best = None
    best_diff = float("inf")
    for mults in multipliers_options:
        breaks = []
        for p in range(log_rng[0], log_rng[1] + 1):
            for m in mults:
                breaks.append(m * base**p)
        diff = abs(len(breaks) - n)
        if diff < best_diff:
            best_diff = diff
            best = np.array(sorted(set(breaks)))
    return best if best is not None else np.array([base**log_rng[0]])


def minor_breaks_log(
    detail: Optional[int] = None,
    smallest: Optional[float] = None,
) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    """Create a minor-break generator for log-scaled axes.

    For base-10 axes the standard convention is to place minor ticks at
    2, 3, 4, ..., 9 between each pair of major powers of 10.

    Parameters
    ----------
    detail : int or None, optional
        Controls how many minor ticks to place between major breaks.

        * ``None`` (default) or ``10`` -- all sub-decade values
          ``2, 3, ..., 9``.
        * ``5`` -- even sub-decade values ``2, 4, 6, 8``.
        * ``1`` -- no extra minor ticks.

        More precisely, ``detail`` evenly spaced multipliers in
        ``[1, base)`` are used.
    smallest : float or None, optional
        Smallest absolute value for which minor breaks are generated.
        Minor breaks below this threshold are dropped.  Useful for
        avoiding clutter near zero on log axes.

    Returns
    -------
    callable
        A function ``(major_breaks, limits, n) -> numpy.ndarray`` of
        minor-break positions (in data space, *not* log space).

    Examples
    --------
    >>> minors = minor_breaks_log()
    >>> majors = np.array([1, 10, 100])
    >>> minors(majors, np.array([1, 100]), 5)
    """

    def _minor_breaks(
        major: np.ndarray,
        limits: np.ndarray,
        n: int = 5,
    ) -> np.ndarray:
        major = np.asarray(major, dtype=float)
        limits = np.asarray(limits, dtype=float)

        if len(major) < 2:
            return np.array([], dtype=float)

        # Detect the base from the ratio of consecutive major breaks
        major_sorted = np.sort(major[major > 0])
        if len(major_sorted) < 2:
            return np.array([], dtype=float)

        base = major_sorted[1] / major_sorted[0]
        # Round base to nearest integer if close
        base_int = round(base)
        if abs(base - base_int) < 0.01:
            base = float(base_int)

        # Determine multipliers
        effective_detail = detail if detail is not None else int(base)
        if effective_detail <= 1:
            return np.array([], dtype=float)

        multipliers = np.linspace(1, base, effective_detail + 1)[:-1]
        # Drop the 1 (that's the major break itself)
        multipliers = multipliers[1:]

        # Generate minor breaks
        log_base = math.log(base)
        lo = math.floor(math.log(max(limits[0], 1e-100), base)) - 1
        hi = math.ceil(math.log(max(limits[1], 1e-100), base)) + 1

        minors: list[float] = []
        for p in range(lo, hi + 1):
            for m in multipliers:
                val = m * base**p
                if limits[0] <= val <= limits[1]:
                    minors.append(val)

        result = np.array(sorted(set(minors)), dtype=float)

        # Apply smallest threshold
        if smallest is not None:
            result = result[np.abs(result) >= smallest]

        return result

    return _minor_breaks
