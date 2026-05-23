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
    new: Union[ArrayLike, Sequence, None],
    existing: Optional[List] = None,
    drop: bool = False,
    na_rm: bool = False,
    fct: Optional[bool] = None,
) -> Optional[list]:
    """Train (update) a discrete range with new data.

    Combines the unique levels of *new* with an *existing* level list
    to produce an updated set of levels.  R parity
    (``scale-discrete.R:30-97``): the ``fct`` argument is a three-state
    control on whether the union is treated as factor-ordered or
    sorted-character.

    Parameters
    ----------
    new : array_like, sequence, or None
        New discrete observations.  ``None`` short-circuits and
        returns *existing* unchanged (R: ``if (is.null(new)) existing``).
    existing : list or None, optional
        Previously computed list of levels.  ``None`` indicates no
        prior levels.
    drop : bool, optional
        If ``True`` and *new* is a :class:`pandas.Categorical`,
        unused categories are dropped (default ``False``).
    na_rm : bool, optional
        If ``True``, ``None`` / ``NaN`` values are removed from the
        result (default ``False``).
    fct : bool or None, optional
        Three-state factor-control matching R's ``fct = NA`` default.

        * ``None`` (default, R's ``NA``) — detect from input types
          (Categorical / pandas factor → preserve order, plain
          character / numeric → sort).
        * ``True`` — force factor semantics: preserve combined order
          even when neither side is a Categorical.
        * ``False`` — force non-factor: sort the union alphabetically
          even if one side was a Categorical.

    Returns
    -------
    list or None
        Updated list of unique levels, or ``None`` if *new* is ``None``
        and *existing* is also ``None``.

    Examples
    --------
    >>> train_discrete(["a", "b", "c"])
    ['a', 'b', 'c']
    >>> train_discrete(["b", "d"], existing=["a", "b", "c"])
    ['a', 'b', 'c', 'd']
    >>> train_discrete(None, existing=["a", "b"]) == ["a", "b"]
    True
    >>> train_discrete(["c", "a", "b"], fct=True)  # preserve insert order
    ['c', 'a', 'b']
    """
    # R scale-discrete.R:38-40 — NULL new returns existing unchanged.
    if new is None:
        return existing

    # R scale-discrete.R:54-97 ``discrete_range``. Faithful port:
    #
    #   new_is_factor <- is.factor(new)                            # line 56
    #   old_is_factor <- is.factor(old) || isTRUE(fct)             # line 57
    #   new          <- clevels(new, drop, na.rm)                  # line 58
    #   if (is.null(old)) return(new)
    #
    #   if (old_is_factor && !is.factor(old)) old <- factor(old,old)
    #   if (!is.character(old))               old <- clevels(old, na.rm)
    #   else                                  old <- sort(old, na.last=...)
    #
    #   # Richer side becomes primary
    #   if (new_is_factor && !old_is_factor) { swap(old, new); swap(flags) }
    #
    #   new_levels <- setdiff(new, old)
    #   if (length(new_levels) == 0) return(old)
    #   range <- c(old, new_levels)
    #   if (old_is_factor) return(range)
    #   sort(range, na.last = ...)
    new_is_factor = hasattr(new, "categories")
    old_is_factor = hasattr(existing, "categories") or (fct is True)
    new_levels = _clevels(new, drop=drop, na_rm=na_rm)

    if existing is None:
        return new_levels

    # Normalise ``old`` into a level list.  Dispatch order matters —
    # Categoricals trip ``_is_character_only`` too (their elements *are*
    # strings), so a real factor must be caught first.
    if hasattr(existing, "categories"):
        # Real factor — ``clevels`` preserves category order regardless
        # of ``fct``.  ``fct=False`` does NOT downgrade a real factor.
        old_levels = _clevels(existing, na_rm=na_rm)
    elif old_is_factor:
        # ``fct=TRUE`` on a plain list — R's ``factor(old, old)`` step
        # treats list-as-given as the factor's level order.
        old_levels = list(existing)
        if na_rm:
            old_levels = [v for v in old_levels if not _is_na(v)]
    elif _is_character_only(existing):
        # Plain character existing — R line 71: ``sort(old, na.last=...)``.
        old_levels = sorted(existing, key=lambda v: (v is None, str(v)))
        if na_rm:
            old_levels = [v for v in old_levels if not _is_na(v)]
    else:
        # Numeric / mixed non-character non-factor — clevels (sort+unique).
        old_levels = _clevels(existing, na_rm=na_rm)

    # R lines 79-86 — "if new is more rich than old it becomes the primary".
    # When ``new`` is a factor but ``old`` is not, the factor's level order
    # wins.  This is implemented as a swap of (old, new) and their flags.
    if new_is_factor and not old_is_factor:
        old_levels, new_levels = new_levels, old_levels
        old_is_factor, new_is_factor = True, False

    # R line 88: ``new_levels <- setdiff(new, old)``.
    old_keys = {_na_key(v) for v in old_levels}
    appended = [v for v in new_levels if _na_key(v) not in old_keys]
    if len(appended) == 0:
        return old_levels

    range_ = list(old_levels) + appended

    # R line 95: ``if (old_is_factor) return(range)`` — factor side wins,
    # no resort.  Otherwise the union is sorted alphabetically.
    if old_is_factor:
        return range_
    range_ = sorted(range_, key=lambda v: (v is None, str(v)))
    if na_rm:
        range_ = [v for v in range_ if not _is_na(v)]
    return range_


def _clevels(
    x: Union[ArrayLike, Sequence],
    drop: bool = False,
    na_rm: bool = False,
) -> list:
    """Port of R ``clevels()`` — factor levels, or sorted-uniques for
    non-factor input.  R reference: scale-discrete.R:99-116."""
    if hasattr(x, "categories"):
        if drop:
            x = x.remove_unused_categories()
        levs = list(x.categories)
        # R: ``if (!na.rm && any(is.na(x))) levs <- c(levs, NA)``
        if not na_rm:
            arr = np.asarray(x)
            if any(_is_na(v) for v in arr.flat):
                levs.append(None)
        return levs
    arr = np.asarray(x)
    seen: set = set()
    uniq: list = []
    for val in arr.flat:
        key = _na_key(val)
        if key not in seen:
            seen.add(key)
            uniq.append(val)
    if na_rm:
        uniq = [v for v in uniq if not _is_na(v)]
    return sorted(uniq, key=lambda v: (v is None, str(v)))


def _is_character_only(seq: Sequence) -> bool:
    """True iff every non-NA element is a string (R: ``is.character``)."""
    for v in seq:
        if _is_na(v):
            continue
        if not isinstance(v, str):
            return False
    return True


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
