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
import pandas as pd
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

    Accumulates factor / categorical levels across multiple ``train()``
    calls, preserving order of first appearance.

    Examples
    --------
    >>> rng = DiscreteRange()
    >>> rng.train(["b", "a", "c"])
    >>> rng.range
    ['b', 'a', 'c']
    >>> rng.train(["d", "a"])
    >>> rng.range
    ['b', 'a', 'c', 'd']
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
            its categories are used (respecting order).
        drop : bool, optional
            If ``True`` and *x* is categorical, unused categories are
            dropped before training (default ``False``).
        na_rm : bool, optional
            If ``True``, ``None`` / ``NaN`` values are removed before
            training (default ``False``).
        """
        # Handle pandas Categoricals
        if hasattr(x, "categories"):
            self._is_factor = True
            if drop:
                x = x.remove_unused_categories()
            levels = list(x.categories)
        else:
            x = np.asarray(x)
            # Build unique list preserving first-appearance order
            seen: set = set()
            levels: list = []
            for val in x.flat:
                key = val
                # Treat np.nan consistently
                if isinstance(val, float) and np.isnan(val):
                    key = None
                if key not in seen:
                    seen.add(key)
                    levels.append(val)

        # Optionally strip NaN / None
        if na_rm:
            levels = [
                v
                for v in levels
                if not (v is None or (isinstance(v, float) and np.isnan(v)))
            ]

        if self.range is None:
            self.range = levels
        else:
            existing_set = set()
            for v in self.range:
                if isinstance(v, float) and np.isnan(v):
                    existing_set.add(None)
                else:
                    existing_set.add(v)
            new_levels = list(self.range)
            for v in levels:
                key = None if (isinstance(v, float) and np.isnan(v)) else v
                if key not in existing_set:
                    existing_set.add(key)
                    new_levels.append(v)
            self.range = new_levels

    def reset(self) -> None:
        """Reset to an empty range."""
        self.range = None
        self._is_factor = False
