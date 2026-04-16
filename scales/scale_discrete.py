"""
Discrete-scale helpers: apply and train discrete scales.

Python port of ``R/scale-discrete.R`` from the R *scales* package
(https://github.com/r-lib/scales).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "dscale",
    "train_discrete",
]


def dscale(
    x: ArrayLike,
    palette: Callable[[int], Any],
    na_value: Any = None,
) -> np.ndarray:
    """Apply a discrete scale to categorical data.

    Maps each unique level of *x* to a palette output, then broadcasts
    back to the full length of *x*.

    Parameters
    ----------
    x : array_like
        Discrete (categorical) values.  May be strings, integers, or a
        :class:`pandas.Categorical`.
    palette : callable
        A discrete palette function that takes an integer *n* (number
        of levels) and returns a sequence of *n* output values.
    na_value : any, optional
        Value used for ``None`` / ``NaN`` entries in *x* (default
        ``None``).

    Returns
    -------
    numpy.ndarray
        Palette-mapped values, same length as *x*.

    Examples
    --------
    >>> from scales.palettes import pal_brewer
    >>> dscale(["a", "b", "a", "c"], pal_brewer())
    """
    # Determine levels (ordered unique values)
    if hasattr(x, "categories"):
        # pandas Categorical
        levels = list(x.categories)
        x_arr = np.asarray(x)
    else:
        x_arr = np.asarray(x)
        # Preserve first-appearance order
        seen: set = set()
        levels: list = []
        for val in x_arr.flat:
            key = _na_key(val)
            if key not in seen:
                seen.add(key)
                if not _is_na(val):
                    levels.append(val)

    n = len(levels)
    if n == 0:
        return np.full(x_arr.shape, na_value, dtype=object)

    # Get palette colours / values for n levels
    pal_values = palette(n)
    pal_values = np.asarray(pal_values)

    # Build lookup: level -> palette value
    lookup: dict = {}
    for i, lev in enumerate(levels):
        lookup[lev] = pal_values[i] if i < len(pal_values) else na_value

    # Map x through the lookup
    result = np.empty(x_arr.shape, dtype=pal_values.dtype if len(pal_values) > 0 else object)
    for idx in np.ndindex(x_arr.shape):
        val = x_arr[idx]
        if _is_na(val):
            result[idx] = na_value
        else:
            result[idx] = lookup.get(val, na_value)

    return result


def train_discrete(
    new: Union[ArrayLike, Sequence],
    existing: Optional[List] = None,
    drop: bool = False,
    na_rm: bool = False,
) -> list:
    """Train (update) a discrete range with new data.

    Combines the unique levels of *new* with an *existing* level list
    to produce an updated set of levels (preserving order of first
    appearance).

    Parameters
    ----------
    new : array_like or sequence
        New discrete observations.
    existing : list or None, optional
        Previously computed list of levels.  ``None`` indicates no
        prior levels.
    drop : bool, optional
        If ``True`` and *new* is a :class:`pandas.Categorical`,
        unused categories are dropped (default ``False``).
    na_rm : bool, optional
        If ``True``, ``None`` / ``NaN`` values are removed from the
        result (default ``False``).

    Returns
    -------
    list
        Updated list of unique levels.

    Examples
    --------
    >>> train_discrete(["a", "b", "c"])
    ['a', 'b', 'c']
    >>> train_discrete(["b", "d"], existing=["a", "b", "c"])
    ['a', 'b', 'c', 'd']
    """
    # Extract levels from new data.
    # R semantics: non-factor input is `sort(unique(...))`; Categorical
    # (factor) input preserves its defined order.
    existing_is_factor = hasattr(existing, "categories")
    new_is_factor = hasattr(new, "categories")

    if new_is_factor:
        if drop:
            new = new.remove_unused_categories()
        new_levels = list(new.categories)
    else:
        arr = np.asarray(new)
        seen: set = set()
        uniq: list = []
        for val in arr.flat:
            key = _na_key(val)
            if key not in seen:
                seen.add(key)
                uniq.append(val)
        new_levels = uniq

    if na_rm:
        new_levels = [v for v in new_levels if not _is_na(v)]

    if existing is None:
        if new_is_factor:
            return new_levels
        # Non-factor: sort alphabetically per R's clevels().
        return sorted(new_levels, key=lambda v: (v is None, str(v)))

    existing_keys = {_na_key(v) for v in existing}
    merged = list(existing)
    for v in new_levels:
        key = _na_key(v)
        if key not in existing_keys:
            existing_keys.add(key)
            merged.append(v)

    # When neither side is a factor, R re-sorts the union.
    if not (existing_is_factor or new_is_factor):
        merged = sorted(merged, key=lambda v: (v is None, str(v)))

    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_na(val: Any) -> bool:
    """Check if a value is NA-like (None or NaN)."""
    if val is None:
        return True
    try:
        return np.isnan(val)
    except (TypeError, ValueError):
        return False


def _na_key(val: Any) -> Any:
    """Return a hashable key, mapping all NA variants to ``None``."""
    if _is_na(val):
        return None
    return val
