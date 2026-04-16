"""
Bounds, rescaling, and out-of-bounds handling utilities.

Python port of ``scales::bounds.R`` from the R *scales* package
(https://github.com/r-lib/scales).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from ._utils import zero_range

__all__ = [
    "rescale",
    "rescale_mid",
    "rescale_max",
    "rescale_none",
    "censor",
    "squish",
    "squish_infinite",
    "discard",
    "oob_censor",
    "oob_censor_any",
    "oob_squish",
    "oob_squish_any",
    "oob_squish_infinite",
    "oob_keep",
    "oob_discard",
    "trim_to_domain",
    "trans_range",  # R alias: trans_range <- trim_to_domain
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_RangeLike = Tuple[float, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_numeric(x: np.ndarray) -> np.ndarray:
    """Convert datetime64 arrays to float64 (nanoseconds since epoch).

    Non-datetime arrays are returned as float64 without modification.
    """
    if np.issubdtype(x.dtype, np.datetime64):
        return x.astype("datetime64[ns]").astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _ensure_array(x: Union[np.ndarray, list, float]) -> np.ndarray:
    """Coerce *x* to a NumPy array."""
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Rescaling
# ---------------------------------------------------------------------------

def rescale(
    x: Union[np.ndarray, list, float],
    to: _RangeLike = (0, 1),
    from_range: Optional[_RangeLike] = None,
) -> np.ndarray:
    """Linearly rescale a numeric vector to a new range.

    Parameters
    ----------
    x : array_like
        Numeric values to rescale.
    to : tuple of float, optional
        Output range ``(min, max)``.  Default ``(0, 1)``.
    from_range : tuple of float or None, optional
        Input range ``(min, max)``.  When *None* (default) the range is
        computed from ``x`` (ignoring NaN).

    Returns
    -------
    np.ndarray
        Rescaled values.
    """
    x = _ensure_array(x)
    x_num = _as_numeric(x)

    if from_range is None:
        if x_num.size > 0 and np.all(np.isnan(x_num)):
            return np.full_like(x_num, np.nan)
        from_range = (np.nanmin(x_num), np.nanmax(x_num))

    from_min, from_max = float(from_range[0]), float(from_range[1])
    to_min, to_max = float(to[0]), float(to[1])

    if from_min == from_max:
        return np.full_like(x_num, (to_min + to_max) / 2.0)

    return (x_num - from_min) / (from_max - from_min) * (to_max - to_min) + to_min


def rescale_mid(
    x: Union[np.ndarray, list, float],
    to: _RangeLike = (0, 1),
    from_range: Optional[_RangeLike] = None,
    mid: float = 0,
) -> np.ndarray:
    """Rescale numeric vector to new range with a specified midpoint.

    The *mid* value is mapped to the mean of *to*.

    Parameters
    ----------
    x : array_like
        Numeric values to rescale.
    to : tuple of float, optional
        Output range ``(min, max)``.  Default ``(0, 1)``.
    from_range : tuple of float or None, optional
        Input range ``(min, max)``.  Defaults to ``(min(x), max(x))``.
    mid : float, optional
        Value in the input domain that should be mapped to the midpoint
        of *to*.  Default ``0``.

    Returns
    -------
    np.ndarray
        Rescaled values.
    """
    x = _ensure_array(x)
    x_num = _as_numeric(x)

    if from_range is None:
        if x_num.size > 0 and np.all(np.isnan(x_num)):
            return np.full_like(x_num, np.nan)
        from_range = (np.nanmin(x_num), np.nanmax(x_num))

    from_min, from_max = float(from_range[0]), float(from_range[1])
    to_min, to_max = float(to[0]), float(to[1])

    # Mirrors R's rescale_mid.numeric:
    #   if (zero_range(from) || zero_range(to)) return mean(to) for non-NA
    #   extent <- 2 * max(abs(from - mid))
    #   (x - mid) / extent * diff(to) + mean(to)
    to_mean = (to_min + to_max) / 2.0
    if zero_range((from_min, from_max)) or zero_range((to_min, to_max)):
        result = np.where(np.isnan(x_num), np.nan, to_mean)
        return result

    extent = 2.0 * max(abs(from_min - mid), abs(from_max - mid))
    return (x_num - mid) / extent * (to_max - to_min) + to_mean


def rescale_max(
    x: Union[np.ndarray, list, float],
    to: _RangeLike = (0, 1),
    from_range: Optional[_RangeLike] = None,
) -> np.ndarray:
    """Rescale numeric vector relative to its maximum.

    Parameters
    ----------
    x : array_like
        Numeric values to rescale.
    to : tuple of float, optional
        Output range ``(min, max)``.  Default ``(0, 1)``.
    from_range : tuple of float or None, optional
        Input range ``(min, max)``.  Defaults to ``(0, max(x))``.

    Returns
    -------
    np.ndarray
        Rescaled values.
    """
    x = _ensure_array(x)
    x_num = _as_numeric(x)

    if from_range is None:
        from_range = (0.0, np.nanmax(x_num))

    return x_num / float(from_range[1]) * float(to[1])


def rescale_none(
    x: Union[np.ndarray, list, float],
    to: Optional[_RangeLike] = None,
    from_range: Optional[_RangeLike] = None,
) -> np.ndarray:
    """Identity rescaler — returns *x* unchanged.

    Parameters
    ----------
    x : array_like
        Values.
    to : ignored
    from_range : ignored

    Returns
    -------
    np.ndarray
    """
    return _ensure_array(x)


# ---------------------------------------------------------------------------
# Censoring / squishing / discarding
# ---------------------------------------------------------------------------

def censor(
    x: Union[np.ndarray, list, float],
    range: _RangeLike = (0, 1),
    only_finite: bool = True,
) -> np.ndarray:
    """Replace values outside *range* with ``np.nan``.

    Parameters
    ----------
    x : array_like
        Numeric values.
    range : tuple of float, optional
        Allowed ``(min, max)`` range.  Default ``(0, 1)``.
    only_finite : bool, optional
        If *True* (default), infinite values are **not** censored.

    Returns
    -------
    np.ndarray
        Array with out-of-range values replaced by ``np.nan``.
    """
    x = np.array(_ensure_array(x), dtype=np.float64)
    lo, hi = float(range[0]), float(range[1])

    if only_finite:
        finite = np.isfinite(x)
        oob = finite & ((x < lo) | (x > hi))
    else:
        oob = (x < lo) | (x > hi)

    x[oob] = np.nan
    return x


def squish(
    x: Union[np.ndarray, list, float],
    range: _RangeLike = (0, 1),
    only_finite: bool = True,
) -> np.ndarray:
    """Clamp (squish) values outside *range* to the nearest boundary.

    Parameters
    ----------
    x : array_like
        Numeric values.
    range : tuple of float, optional
        Allowed ``(min, max)`` range.  Default ``(0, 1)``.
    only_finite : bool, optional
        If *True* (default), infinite values are **not** squished.

    Returns
    -------
    np.ndarray
        Array with out-of-range values replaced by the closest boundary.
    """
    x = np.array(_ensure_array(x), dtype=np.float64)
    lo, hi = float(range[0]), float(range[1])

    if only_finite:
        finite = np.isfinite(x)
        x = np.where(finite & (x < lo), lo, x)
        x = np.where(finite & (x > hi), hi, x)
    else:
        x = np.clip(x, lo, hi)
    return x


def squish_infinite(
    x: Union[np.ndarray, list, float],
    range: _RangeLike = (0, 1),
) -> np.ndarray:
    """Replace infinite values with the corresponding boundary of *range*.

    Finite values (including ``np.nan``) are left untouched.

    Parameters
    ----------
    x : array_like
        Numeric values.
    range : tuple of float, optional
        ``(min, max)`` range used as replacement values.  Default ``(0, 1)``.

    Returns
    -------
    np.ndarray
    """
    x = np.array(_ensure_array(x), dtype=np.float64)
    lo, hi = float(range[0]), float(range[1])
    x[x == -np.inf] = lo
    x[x == np.inf] = hi
    return x


def discard(
    x: Union[np.ndarray, list, float],
    range: _RangeLike = (0, 1),
) -> np.ndarray:
    """Remove values outside *range* (returns a shorter array).

    Parameters
    ----------
    x : array_like
        Numeric values.
    range : tuple of float, optional
        Allowed ``(min, max)`` range.  Default ``(0, 1)``.

    Returns
    -------
    np.ndarray
        Array containing only in-range values.
    """
    x = np.array(_ensure_array(x), dtype=np.float64)
    lo, hi = float(range[0]), float(range[1])
    mask = (x >= lo) & (x <= hi)
    return x[mask]


# ---------------------------------------------------------------------------
# OOB handler functions
# ---------------------------------------------------------------------------
# In the R package, oob_* functions are direct functions with signature
# ``(x, range)``.  We mirror that here.

def oob_censor(
    x: Union[np.ndarray, list, float],
    range: _RangeLike = (0, 1),
) -> np.ndarray:
    """Censor out-of-bounds values (replace with ``np.nan``).

    Infinite values are **not** censored (``only_finite=True``).

    Parameters
    ----------
    x : array_like
    range : tuple of float

    Returns
    -------
    np.ndarray
    """
    return censor(x, range=range, only_finite=True)


def oob_censor_any(
    x: Union[np.ndarray, list, float],
    range: _RangeLike = (0, 1),
) -> np.ndarray:
    """Censor **all** out-of-bounds values, including infinite.

    Parameters
    ----------
    x : array_like
    range : tuple of float

    Returns
    -------
    np.ndarray
    """
    return censor(x, range=range, only_finite=False)


def oob_squish(
    x: Union[np.ndarray, list, float],
    range: _RangeLike = (0, 1),
) -> np.ndarray:
    """Squish (clamp) out-of-bounds values to range limits.

    Infinite values are **not** squished (``only_finite=True``).

    Parameters
    ----------
    x : array_like
    range : tuple of float

    Returns
    -------
    np.ndarray
    """
    return squish(x, range=range, only_finite=True)


def oob_squish_any(
    x: Union[np.ndarray, list, float],
    range: _RangeLike = (0, 1),
) -> np.ndarray:
    """Squish **all** out-of-bounds values, including infinite.

    Parameters
    ----------
    x : array_like
    range : tuple of float

    Returns
    -------
    np.ndarray
    """
    return squish(x, range=range, only_finite=False)


def oob_squish_infinite(
    x: Union[np.ndarray, list, float],
    range: _RangeLike = (0, 1),
) -> np.ndarray:
    """Only squish infinite values to range limits.

    Finite out-of-bounds values are left untouched.

    Parameters
    ----------
    x : array_like
    range : tuple of float

    Returns
    -------
    np.ndarray
    """
    return squish_infinite(x, range=range)


def oob_keep(
    x: Union[np.ndarray, list, float],
    range: _RangeLike = (0, 1),
) -> np.ndarray:
    """Keep all values unchanged (no out-of-bounds modification).

    Parameters
    ----------
    x : array_like
    range : ignored

    Returns
    -------
    np.ndarray
    """
    return np.array(_ensure_array(x), dtype=np.float64)


def oob_discard(
    x: Union[np.ndarray, list, float],
    range: _RangeLike = (0, 1),
) -> np.ndarray:
    """Discard (remove) out-of-bounds values.

    Parameters
    ----------
    x : array_like
    range : tuple of float

    Returns
    -------
    np.ndarray
        Shorter array with out-of-bounds values removed.
    """
    return discard(x, range=range)


# ---------------------------------------------------------------------------
# Transform domain utilities
# ---------------------------------------------------------------------------

def trim_to_domain(
    transform: object,
    x: Union[np.ndarray, list, float],
) -> np.ndarray:
    """Compute the **transformed range** of *x*, clipped to the transform's domain.

    Mirrors R's ``trim_to_domain``:
    ``range(transform$transform(range(squish(x, transform$domain), na.rm = TRUE)))``.

    The return shape is **always a length-2 array**: the (min, max) of the
    transformed, domain-squished data.

    Parameters
    ----------
    transform : object
        A transform object exposing ``transform(x)`` and a ``domain``
        ``(min, max)`` tuple.
    x : array_like
        Values to summarise.

    Returns
    -------
    np.ndarray
        Length-2 array ``[trans_min, trans_max]``.
    """
    x_arr = np.array(_ensure_array(x), dtype=np.float64)
    # Accept both string-named transforms and transform objects. If the
    # object exposes a `domain` attribute, use it; otherwise assume the
    # whole real line.
    domain = getattr(transform, "domain", (-np.inf, np.inf))
    squished = squish(x_arr, range=domain, only_finite=True)
    finite = squished[np.isfinite(squished)]
    if finite.size == 0:
        return np.array([np.nan, np.nan], dtype=float)
    raw_range = np.array([finite.min(), finite.max()], dtype=float)
    transformed = np.asarray(transform.transform(raw_range), dtype=float)
    if transformed.size == 0 or np.all(~np.isfinite(transformed)):
        return np.array([np.nan, np.nan], dtype=float)
    return np.array([float(np.nanmin(transformed)), float(np.nanmax(transformed))])


# R alias: trans_range <- trim_to_domain
trans_range = trim_to_domain
