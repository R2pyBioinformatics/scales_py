"""
Utility functions for the scales package.

Python port of utility functions from the R scales package
(https://github.com/r-lib/scales). Corresponds primarily to:
  - R/utils.R
  - R/range.R
  - R/full-seq.R
  - R/round.R
"""

from __future__ import annotations

import sys
from datetime import timedelta
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "zero_range",
    "expand_range",
    "rescale_common",
    "recycle_common",
    "fullseq",
    "round_any",
    "offset_by",
    "precision",
    "demo_continuous",
    "demo_log10",
    "demo_discrete",
    "demo_datetime",
    "demo_time",
    "demo_timespan",
]


def zero_range(
    x: ArrayLike, tol: float = 1000 * sys.float_info.epsilon
) -> bool:
    """
    Check if a range has (effectively) zero extent.

    Parameters
    ----------
    x : array-like of length 2
        A ``[min, max]`` pair describing a range.
    tol : float, optional
        Tolerance for comparison, by default ``1000 * sys.float_info.epsilon``.
        The range is considered zero when
        ``abs(max - min) < tol * abs(mean(min, max))``.

    Returns
    -------
    bool
        ``True`` if the range is effectively zero, ``False`` otherwise.

    Notes
    -----
    * If either element is ``None`` / ``np.nan``, returns ``True``
      (consistent with R's ``NA`` handling).
    * If both elements are infinite with the same sign, returns ``False``.
    * If elements are infinite with different signs, returns ``False``.
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (2,):
        raise ValueError("x must have exactly 2 elements")

    # NA / NaN handling – mirror R: any NA → NA, which we map to True
    if np.any(np.isnan(x)):
        return True

    # Both the same (including both +Inf or both -Inf)
    if x[0] == x[1]:
        return True

    # Mixed infinities → definitely not zero range
    if np.any(np.isinf(x)):
        return False

    diff = np.abs(x[1] - x[0])
    mean = np.abs(x[0] + x[1]) / 2.0

    if mean == 0.0:
        # Avoid 0/0; compare diff to tol directly
        return diff < tol

    return (diff / mean) < tol


def expand_range(
    range: ArrayLike,
    mul: float = 0,
    add: float = 0,
    zero_width: float = 1,
) -> tuple[float, float]:
    """
    Expand a numeric range by multiplicative and additive amounts.

    Parameters
    ----------
    range : array-like of length 2
        ``[min, max]`` of the data range.
    mul : float, optional
        Multiplicative expansion factor (default 0). The range is expanded
        outward by ``mul * (max - min)`` on each side.
    add : float, optional
        Additive expansion amount (default 0). Added to each side of the
        range after multiplicative expansion.
    zero_width : float, optional
        If the range has zero width (see :func:`zero_range`), expand by
        this amount instead (default 1). Half is subtracted from the
        minimum and half is added to the maximum.

    Returns
    -------
    tuple of float
        ``(new_min, new_max)`` after expansion.
    """
    range = np.asarray(range, dtype=float)
    if range.shape != (2,):
        raise ValueError("range must have exactly 2 elements")

    if zero_range(range):
        center = range[0]
        return (center - zero_width / 2, center + zero_width / 2)

    extent = range[1] - range[0]
    return (
        range[0] - extent * mul - add,
        range[1] + extent * mul + add,
    )


def rescale_common(
    x: ArrayLike,
    to: tuple[float, float],
    from_range: tuple[float, float],
) -> np.ndarray:
    """
    Linearly rescale *x* from ``from_range`` into ``to``.

    Parameters
    ----------
    x : array-like
        Numeric values to rescale.
    to : tuple of float
        ``(new_min, new_max)`` target range.
    from_range : tuple of float
        ``(old_min, old_max)`` source range.

    Returns
    -------
    numpy.ndarray
        Rescaled values.
    """
    x = np.asarray(x, dtype=float)
    from_range = np.asarray(from_range, dtype=float)
    to = np.asarray(to, dtype=float)

    if zero_range(from_range):
        return np.full_like(x, (to[0] + to[1]) / 2.0)

    return (x - from_range[0]) / (from_range[1] - from_range[0]) * (
        to[1] - to[0]
    ) + to[0]


def recycle_common(
    *args: ArrayLike, size: Optional[int] = None
) -> list[np.ndarray]:
    """
    Recycle arrays to a common length.

    Each argument must be either length 1 (scalar) or length ``size``.
    Scalars are broadcast to length ``size``. If *size* is ``None`` it is
    inferred as the length of the longest non-scalar argument.

    Parameters
    ----------
    *args : array-like
        One or more arrays to recycle.
    size : int, optional
        Target length. Inferred from the arguments when ``None``.

    Returns
    -------
    list of numpy.ndarray
        Recycled arrays, all of length *size*.

    Raises
    ------
    ValueError
        If arguments have incompatible lengths (not 1 and not *size*).
    """
    arrays = [np.atleast_1d(np.asarray(a)) for a in args]

    if size is None:
        lengths = [len(a) for a in arrays]
        non_scalar = [l for l in lengths if l != 1]
        if not non_scalar:
            size = 1
        else:
            size = max(non_scalar)

    result: list[np.ndarray] = []
    for i, a in enumerate(arrays):
        n = len(a)
        if n == size:
            result.append(a)
        elif n == 1:
            result.append(np.repeat(a, size))
        else:
            raise ValueError(
                f"Argument {i} has length {n}, which is not 1 or {size}"
            )

    return result


def fullseq(
    range: ArrayLike,
    size: float,
    pad: bool = False,
) -> np.ndarray:
    """
    Generate a sequence of fixed-size intervals that covers *range*.

    Parameters
    ----------
    range : array-like of length 2
        ``[min, max]`` of the data range.
    size : float
        Step size for the sequence.
    pad : bool, optional
        If ``True``, extend the sequence by one *size* on each side
        (default ``False``).

    Returns
    -------
    numpy.ndarray
        A regular numeric sequence from (at most) ``min`` to (at least)
        ``max``, with spacing *size*.
    """
    range = np.asarray(range, dtype=float)
    if range.shape != (2,):
        raise ValueError("range must have exactly 2 elements")

    if not np.isfinite(size) or size <= 0:
        raise ValueError("size must be a positive finite number")

    lo = np.floor(range[0] / size) * size
    hi = np.ceil(range[1] / size) * size

    if pad:
        lo -= size
        hi += size

    # Use round_any to avoid floating-point fuzz at boundaries
    return np.arange(lo, hi + size / 2, size)


def round_any(
    x: ArrayLike,
    accuracy: float,
    f: Callable = np.round,
) -> np.ndarray:
    """
    Round values to the nearest multiple of *accuracy*.

    Parameters
    ----------
    x : array-like
        Numeric values to round.
    accuracy : float
        Rounding unit; values are rounded to the nearest multiple of
        this number.
    f : callable, optional
        Rounding function, by default :func:`numpy.round`. Other useful
        choices include :func:`numpy.floor` and :func:`numpy.ceil`.

    Returns
    -------
    numpy.ndarray
        Rounded values.
    """
    x = np.asarray(x, dtype=float)
    return f(x / accuracy) * accuracy


def offset_by(
    x: Union[float, np.datetime64, Any],
    size: Union[float, timedelta, np.timedelta64, Any],
) -> Any:
    """
    Offset a value by *size*.

    For plain numerics, this is simple addition. For datetime-like objects
    the *size* should be an appropriate timedelta.

    Parameters
    ----------
    x : float or datetime-like
        Starting value.
    size : float, timedelta, or numpy.timedelta64
        Amount to offset by.

    Returns
    -------
    float or datetime-like
        ``x + size``.
    """
    return x + size


def precision(x: ArrayLike) -> float:
    """
    Detect the precision of a numeric vector.

    The precision is the smallest power of 10 that captures the spacing
    between unique, finite, non-NaN values.

    Parameters
    ----------
    x : array-like
        Numeric values.

    Returns
    -------
    float
        Precision as a power of 10 (e.g. ``0.01`` for data with two
        decimal places of resolution).

    Notes
    -----
    If *x* has fewer than 2 unique finite values, returns ``1``.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.unique(x)

    if len(x) <= 1:
        return 1.0

    diffs = np.diff(np.sort(x))
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 1.0

    smallest = np.min(diffs)

    if smallest == 0:
        return 1.0

    # Smallest power of 10 <= smallest diff.
    # Round the log10 to avoid floating-point fuzz (e.g. log10(0.1) ≈ -1.0000000000000004).
    log_val = np.log10(smallest)
    rounded = np.round(log_val)
    if np.abs(log_val - rounded) < 1e-6:
        log_val = rounded
    return float(10 ** np.floor(log_val))


# ---------------------------------------------------------------------------
# Demo functions (R source: utils.R)
# ---------------------------------------------------------------------------


def demo_continuous(
    x: ArrayLike,
    *,
    labels: object = None,
    breaks: object = None,
    trans: object = None,
    **kwargs: object,
) -> None:
    """Show a continuous scale demo using matplotlib.

    Creates a simple plot that demonstrates how breaks and labels
    render for the supplied data range.  The R version delegates to
    ggplot2; this Python version uses matplotlib.

    Parameters
    ----------
    x : array-like
        Data range (used as x-limits).
    labels : callable or None
        Label formatter (e.g. ``label_comma()``).
    breaks : callable or None
        Break generator (e.g. ``breaks_extended(n=5)``).
    trans : Transform or None
        Transformation object.
    **kwargs
        Ignored (accepted for forward-compatibility).
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))

    fig, ax = plt.subplots(figsize=(6, 1))

    # Generate breaks
    if breaks is not None:
        ticks = np.asarray(breaks(x))
    else:
        ticks = np.linspace(xmin, xmax, 6)

    # Generate labels
    if labels is not None:
        tick_labels = labels(ticks)
    else:
        tick_labels = [str(v) for v in ticks]

    ax.set_xlim(xmin, xmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks([])
    ax.set_title("Continuous scale demo")
    plt.tight_layout()
    plt.show()


def demo_log10(
    x: ArrayLike,
    *,
    labels: object = None,
    breaks: object = None,
    **kwargs: object,
) -> None:
    """Show a log-10 scale demo using matplotlib.

    Parameters
    ----------
    x : array-like
        Data range.
    labels : callable or None
        Label formatter.
    breaks : callable or None
        Break generator.
    **kwargs
        Ignored.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))

    fig, ax = plt.subplots(figsize=(6, 1))
    ax.set_xscale("log")
    ax.set_xlim(max(xmin, 1e-10), xmax)

    if breaks is not None:
        ticks = np.asarray(breaks(x))
        ax.set_xticks(ticks)
    if labels is not None:
        ax.set_xticklabels(labels(ax.get_xticks()))

    ax.set_yticks([])
    ax.set_title("Log-10 scale demo")
    plt.tight_layout()
    plt.show()


def demo_discrete(
    x: ArrayLike,
    *,
    labels: object = None,
    **kwargs: object,
) -> None:
    """Show a discrete scale demo using matplotlib.

    Parameters
    ----------
    x : array-like
        Categorical values.
    labels : callable or None
        Label formatter.
    **kwargs
        Ignored.
    """
    import matplotlib.pyplot as plt

    x = list(x) if not isinstance(x, (list, np.ndarray)) else list(x)
    positions = list(range(len(x)))

    fig, ax = plt.subplots(figsize=(6, 1))
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_xticks(positions)

    if labels is not None:
        tick_labels = labels(x)
    else:
        tick_labels = [str(v) for v in x]

    ax.set_xticklabels(tick_labels)
    ax.set_yticks([])
    ax.set_title("Discrete scale demo")
    plt.tight_layout()
    plt.show()


def demo_datetime(
    x: ArrayLike,
    *,
    labels: object = None,
    breaks: object = None,
    **kwargs: object,
) -> None:
    """Show a datetime scale demo using matplotlib.

    Parameters
    ----------
    x : array-like
        Datetime values.
    labels : callable or None
        Label formatter.
    breaks : callable or None
        Break generator.
    **kwargs
        Ignored.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 2), layout="constrained")
    ax.set_xlim(min(x), max(x))
    fig.autofmt_xdate()
    ax.set_yticks([])
    ax.set_title("Datetime scale demo")
    plt.show()


def demo_time(
    x: ArrayLike,
    *,
    labels: object = None,
    breaks: object = None,
    **kwargs: object,
) -> None:
    """Show a time scale demo using matplotlib.

    Parameters
    ----------
    x : array-like
        Time/numeric values representing seconds.
    labels : callable or None
        Label formatter.
    breaks : callable or None
        Break generator.
    **kwargs
        Ignored.
    """
    demo_continuous(x, labels=labels, breaks=breaks, **kwargs)


def demo_timespan(
    x: ArrayLike,
    *,
    labels: object = None,
    breaks: object = None,
    **kwargs: object,
) -> None:
    """Show a timespan scale demo using matplotlib.

    Parameters
    ----------
    x : array-like
        Timespan data (numeric seconds).
    labels : callable or None
        Label formatter.
    breaks : callable or None
        Break generator.
    **kwargs
        Ignored.
    """
    demo_continuous(x, labels=labels, breaks=breaks, **kwargs)
