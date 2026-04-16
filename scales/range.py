"""
Mutable range classes for accumulating scale domains.

Python port of ``R/range.R`` from the R *scales* package
(https://github.com/r-lib/scales).  The R source uses R6 classes;
here we use plain Python classes with the same ``train`` / ``reset``
interface.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "Range",
    "ContinuousRange",
    "DiscreteRange",
]


class Range:
    """Base range class.

    Attributes
    ----------
    range : object or None
        The accumulated range.  ``None`` until the first ``train()``
        call.
    """

    def __init__(self) -> None:
        self.range: object = None

    def train(self, x: ArrayLike, **kwargs) -> None:  # pragma: no cover
        """Update the range with new data (implemented by subclasses)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the range to its initial (empty) state."""
        self.range = None


class ContinuousRange(Range):
    """Mutable continuous range that accumulates via :meth:`train`.

    An R6-style object that progressively builds a numeric ``(min, max)``
    range across multiple ``train()`` calls.

    Examples
    --------
    >>> rng = ContinuousRange()
    >>> rng.train([1, 5, 3])
    >>> rng.range
    (1.0, 5.0)
    >>> rng.train([0, 4])
    >>> rng.range
    (0.0, 5.0)
    """

    def __init__(self) -> None:
        super().__init__()
        self.range: Optional[tuple[float, float]] = None

    def train(self, x: ArrayLike) -> None:
        """Update the range with new numeric data.

        Parameters
        ----------
        x : array_like
            Numeric values.  Non-finite values (``NaN``, ``Inf``) are
            silently dropped before the range is updated.
        """
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return
        new_range = (float(np.min(x)), float(np.max(x)))
        if self.range is None:
            self.range = new_range
        else:
            self.range = (
                min(self.range[0], new_range[0]),
                max(self.range[1], new_range[1]),
            )

    def reset(self) -> None:
        """Reset to an empty range."""
        self.range = None


class DiscreteRange(Range):
    """Mutable discrete range (ordered set of unique levels).

    Mirrors R ``scales::discrete_range`` / ``clevels``
    (scales/R/scale-discrete.R:55-116):

    * If the input is a pandas Categorical (R factor), its ``categories``
      order is preserved.
    * Otherwise, levels are **sorted alphabetically** (R ``sort(unique(x))``).
    * When combined with an existing range, a factor input keeps its
      order; a non-factor combination is re-sorted.

    Examples
    --------
    >>> rng = DiscreteRange()
    >>> rng.train(["b", "a", "c"])
    >>> rng.range
    ['a', 'b', 'c']
    >>> rng.train(["d", "a"])
    >>> rng.range
    ['a', 'b', 'c', 'd']
    """

    def __init__(self) -> None:
        super().__init__()
        self.range: Optional[list] = None
        self._is_factor: bool = False

    def train(
        self,
        x: Union[ArrayLike, Sequence, "pd.Categorical"],
        drop: bool = False,
        na_rm: bool = False,
    ) -> None:
        """Update the range with new discrete data.

        Parameters
        ----------
        x : array_like or pandas.Categorical
            Discrete values.  If *x* is a :class:`pandas.Categorical`
            its categories are used (respecting order).  Otherwise,
            the unique values are sorted alphabetically to match R's
            ``sort(unique(x))`` behaviour in ``clevels()``.
        drop : bool, optional
            If ``True`` and *x* is categorical, unused categories are
            dropped before training (default ``False``).
        na_rm : bool, optional
            If ``True``, ``None`` / ``NaN`` values are removed before
            training (default ``False``).
        """
        new_is_factor = hasattr(x, "categories")
        # Handle pandas Categoricals — factor-style, preserve order.
        if new_is_factor:
            if drop:
                x = x.remove_unused_categories()
            levels = list(x.categories)
        else:
            x = np.asarray(x)
            # R's clevels for non-factor: sort(unique(x))
            seen: set = set()
            uniq: list = []
            for val in x.flat:
                key = val
                if isinstance(val, float) and np.isnan(val):
                    key = None
                if key not in seen:
                    seen.add(key)
                    uniq.append(val)
            # Sort alphabetically (R default).  Keep NaN separate —
            # sorted() will raise on mixed None/str, so we strip first
            # and re-append.
            non_na = [v for v in uniq if not (v is None or
                        (isinstance(v, float) and np.isnan(v)))]
            na_tail = [v for v in uniq if v is None or
                        (isinstance(v, float) and np.isnan(v))]
            try:
                non_na = sorted(non_na)
            except TypeError:
                # Mixed incomparable types — keep insertion order
                pass
            levels = non_na + na_tail

        # Optionally strip NaN / None
        if na_rm:
            levels = [
                v
                for v in levels
                if not (v is None or (isinstance(v, float) and np.isnan(v)))
            ]

        if self.range is None:
            # First batch — remember whether it was a factor.
            self.range = levels
            self._is_factor = new_is_factor
        else:
            # Combine with existing range.  R discrete_range
            # (scale-discrete.R:82-96): union of old ∪ new_levels.
            # Keep factor order if either side was a factor; else
            # re-sort alphabetically.
            existing_set = set()
            for v in self.range:
                if isinstance(v, float) and np.isnan(v):
                    existing_set.add(None)
                else:
                    existing_set.add(v)
            combined = list(self.range)
            for v in levels:
                key = None if (isinstance(v, float) and np.isnan(v)) else v
                if key not in existing_set:
                    existing_set.add(key)
                    combined.append(v)

            if self._is_factor or new_is_factor:
                self.range = combined
                self._is_factor = True
            else:
                non_na = [v for v in combined if not (v is None or
                            (isinstance(v, float) and np.isnan(v)))]
                na_tail = [v for v in combined if v is None or
                            (isinstance(v, float) and np.isnan(v))]
                try:
                    non_na = sorted(non_na)
                except TypeError:
                    pass
                self.range = non_na + na_tail

    def reset(self) -> None:
        """Reset to an empty range."""
        self.range = None
        self._is_factor = False
