"""
Minor-break generators for linear (non-log) scales.

Python port of ``R/minor_breaks.R`` from the R *scales* package
(https://github.com/r-lib/scales).

All public generators are closure factories: they return a callable with
signature ``(major_breaks, limits, n) -> numpy.ndarray``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "minor_breaks_n",
    "minor_breaks_width",
    "regular_minor_breaks",
]


def minor_breaks_n(
    n: int = 2,
) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    """Place *n* minor breaks between each pair of major breaks.

    Parameters
    ----------
    n : int, optional
        Number of minor breaks between consecutive major breaks
        (default 2).

    Returns
    -------
    callable
        A function ``(major_breaks, limits, n_minor) -> numpy.ndarray``
        of minor-break positions.

    Examples
    --------
    >>> fn = minor_breaks_n(n=4)
    >>> fn(np.array([0, 10, 20]), np.array([0, 20]), 4)
    array([ 2.,  4.,  6.,  8., 12., 14., 16., 18.])
    """

    def _minor_breaks(
        major: np.ndarray,
        limits: np.ndarray,
        n_minor: int = n,
    ) -> np.ndarray:
        major = np.sort(np.asarray(major, dtype=float))
        limits = np.asarray(limits, dtype=float)

        if len(major) < 2:
            return np.array([], dtype=float)

        minors: list[float] = []
        for i in range(len(major) - 1):
            lo = major[i]
            hi = major[i + 1]
            step = (hi - lo) / (n_minor + 1)
            for j in range(1, n_minor + 1):
                val = lo + j * step
                if limits[0] <= val <= limits[1]:
                    minors.append(val)

        return np.array(sorted(set(minors)), dtype=float)

    return _minor_breaks


def minor_breaks_width(
    width: float,
    offset: float = 0,
) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    """Place minor breaks at fixed-width intervals.

    Parameters
    ----------
    width : float
        Spacing between consecutive minor breaks.
    offset : float, optional
        Offset from zero for the break grid (default 0).  The grid is
        placed at ``... , offset - width, offset, offset + width, ...``
        and then clipped to the axis limits.

    Returns
    -------
    callable
        A function ``(major_breaks, limits, n) -> numpy.ndarray``
        of minor-break positions.

    Examples
    --------
    >>> fn = minor_breaks_width(2.5)
    >>> fn(np.array([0, 10, 20]), np.array([0, 20]), 5)
    array([ 0. ,  2.5,  5. ,  7.5, 10. , 12.5, 15. , 17.5, 20. ])
    """

    def _minor_breaks(
        major: np.ndarray,
        limits: np.ndarray,
        n: int = 5,
    ) -> np.ndarray:
        limits = np.asarray(limits, dtype=float)
        lo, hi = float(limits[0]), float(limits[1])

        # Build grid aligned to offset
        start = np.floor((lo - offset) / width) * width + offset
        stop = np.ceil((hi - offset) / width) * width + offset
        breaks = np.arange(start, stop + width / 2, width)

        # Clip to limits
        breaks = breaks[(breaks >= lo - 1e-10) & (breaks <= hi + 1e-10)]

        return breaks

    return _minor_breaks


def regular_minor_breaks(
    reverse: bool = False,
) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    """Default minor-break placement: ``n - 1`` evenly spaced between majors.

    This is the standard minor-break strategy used by ggplot2.  Between
    each pair of consecutive major breaks, ``n - 1`` minor breaks are
    inserted at equal spacing.

    Parameters
    ----------
    reverse : bool, optional
        If ``True``, the limits are internally reversed before
        computing breaks and the result is reversed back.  Useful for
        reversed continuous scales (default ``False``).

    Returns
    -------
    callable
        A function ``(major_breaks, limits, n) -> numpy.ndarray``
        of minor-break positions.

    Examples
    --------
    >>> fn = regular_minor_breaks()
    >>> fn(np.array([0, 5, 10]), np.array([0, 10]), 2)
    array([ 2.5,  7.5])
    """

    def _minor_breaks(
        major: np.ndarray,
        limits: np.ndarray,
        n: int = 2,
    ) -> np.ndarray:
        major = np.asarray(major, dtype=float)
        limits = np.asarray(limits, dtype=float)

        if reverse:
            major = -major
            limits = -limits[::-1]

        major = np.sort(major)

        if len(major) < 2 or n < 1:
            return np.array([], dtype=float)

        # n-1 minor breaks between each pair of majors
        n_minor = n - 1
        if n_minor < 1:
            return np.array([], dtype=float)

        minors: list[float] = []
        # Extend beyond the major range by one step on each side
        step = major[1] - major[0]
        extended = np.concatenate(
            [[major[0] - step], major, [major[-1] + step]]
        )

        for i in range(len(extended) - 1):
            lo = extended[i]
            hi = extended[i + 1]
            interval_step = (hi - lo) / n
            for j in range(1, n):
                val = lo + j * interval_step
                if limits[0] <= val <= limits[1]:
                    minors.append(val)

        result = np.array(sorted(set(minors)), dtype=float)

        if reverse:
            result = -result[::-1]

        return result

    return _minor_breaks
