"""
Continuous-scale helpers: apply and train continuous scales.

Python port of ``R/scale-continuous.R`` from the R *scales* package
(https://github.com/r-lib/scales).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .bounds import censor, rescale
from .transforms import Transform, as_transform

__all__ = [
    "cscale",
    "train_continuous",
]


def cscale(
    x: ArrayLike,
    palette: Callable[[np.ndarray], np.ndarray],
    na_value: Any = np.nan,
    trans: Optional[Union[Transform, str]] = None,
    oob: Callable[[np.ndarray], np.ndarray] = censor,
) -> np.ndarray:
    """Apply a continuous scale to numeric data.

    Mirrors R's ``cscale`` + ``map_continuous``: transforms *x*, rescales
    to ``[0, 1]``, applies *oob* (censor by default) to that rescaled
    result, then passes it through *palette*.  NaNs (including those
    introduced by *oob*) are replaced with *na_value*.

    Parameters
    ----------
    x : array_like
        Numeric values in data coordinates.
    palette : callable
        A continuous palette function that maps values in ``[0, 1]`` to
        output values (e.g. colours or sizes).
    na_value : any, optional
        Value used for ``NaN`` entries in *x* (default ``np.nan``).
    trans : Transform or str, optional
        If given, *x* is first transformed before rescaling.  May be a
        :class:`~scales.transforms.Transform` object or a string name
        recognised by :func:`~scales.transforms.as_transform`.
    oob : callable, optional
        Out-of-bounds handler applied to the rescaled ``[0, 1]`` values
        before the palette.  Default is :func:`~scales.bounds.censor`,
        which replaces values outside ``[0, 1]`` with ``NaN`` — matching
        R's ``map_continuous(oob = censor)``.  Use
        :func:`~scales.bounds.squish` to clamp instead.

    Returns
    -------
    numpy.ndarray
        Palette-mapped values, same length as *x*.

    Examples
    --------
    >>> from scales.palettes import pal_seq_gradient
    >>> cscale([1, 5, 10], pal_seq_gradient("white", "blue"))
    """
    x = np.asarray(x, dtype=float)

    # 1. Optionally transform
    if trans is not None:
        if isinstance(trans, str):
            trans = as_transform(trans)
        x = trans.transform(x)

    # 2. Identify NAs *before* rescaling
    na_mask = ~np.isfinite(x)

    # 3. Rescale to [0, 1] using the finite range of x
    scaled = rescale(x, to=(0.0, 1.0))

    # 4. Apply OOB handler (default: censor → NaN). After this, any value
    #    outside [0, 1] that the user asked to censor becomes NaN.
    scaled = np.asarray(oob(scaled), dtype=float)
    na_mask = na_mask | ~np.isfinite(scaled)

    # 5. Apply palette
    result = palette(scaled)
    result = np.asarray(result)

    # 6. Replace NAs
    if np.any(na_mask):
        if result.dtype.kind in ("U", "S", "O"):
            # String / object array
            result = result.astype(object)
        result[na_mask] = na_value

    return result


def train_continuous(
    new: ArrayLike,
    existing: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Train (update) a continuous range with new data.

    Combines the range of *new* with an *existing* ``(min, max)`` range
    to produce an updated range that spans both.

    Parameters
    ----------
    new : array_like
        New numeric observations.  Non-finite values are ignored.
    existing : tuple of float or None, optional
        Previously computed ``(min, max)`` range.  ``None`` indicates
        no prior range.

    Returns
    -------
    tuple of float
        Updated ``(min, max)`` range.

    Examples
    --------
    >>> train_continuous([1, 5, 3])
    (1.0, 5.0)
    >>> train_continuous([0, 4], existing=(1.0, 5.0))
    (0.0, 5.0)
    """
    new = np.asarray(new, dtype=float)
    new = new[np.isfinite(new)]

    if len(new) == 0:
        if existing is None:
            raise ValueError("Cannot train on empty data with no existing range.")
        return existing

    new_range = (float(np.min(new)), float(np.max(new)))

    if existing is None:
        return new_range

    return (
        min(existing[0], new_range[0]),
        max(existing[1], new_range[1]),
    )
