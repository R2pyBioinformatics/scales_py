"""
Break generators for log-scaled axes.

Faithful Python port of ``R/breaks-log.R``:

* ``breaks_log`` replicates R's algorithm: look for integer powers of
  ``base``; when too few, densify via ``log_sub_breaks`` (Wilkinson-style
  greedy candidate selection), falling back to ``extended_breaks``.
* ``minor_breaks_log`` honours R's ``detail in {1, 5, 10}`` contract,
  handles negatives by mirroring ticks across zero, and exposes the
  selected detail on the result as a ``.detail`` attribute.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "breaks_log",
    "log_breaks",  # R alias: log_breaks <- breaks_log
    "minor_breaks_log",
]


# ---------------------------------------------------------------------------
# breaks_log
# ---------------------------------------------------------------------------

def _log_sub_breaks(
    rng: tuple[float, float], n: int = 5, base: float = 10
) -> np.ndarray:
    """Port of R's internal ``log_sub_breaks``.

    Greedily picks candidate integers in ``(1, base)`` that, added to
    the current ``steps`` set, maximise the minimum log-spacing gap.
    Repeats until there are at least ``n - 2`` relevant breaks or the
    candidate pool is exhausted.  Falls back to
    :func:`scales.breaks.breaks_extended` when no admissible set is
    found.
    """
    lo = math.floor(rng[0])
    hi = math.ceil(rng[1])
    if base <= 2:
        return np.array(
            [base ** p for p in range(int(lo), int(hi) + 1)], dtype=float
        )

    steps: list[float] = [1.0]

    def _delta(x: float) -> float:
        candidate = sorted({x, *steps, base})
        log_vals = [math.log(v, base) for v in candidate]
        diffs = [log_vals[i + 1] - log_vals[i] for i in range(len(log_vals) - 1)]
        return min(diffs)

    candidate_pool = [c for c in range(2, int(base))]

    breaks = np.array([], dtype=float)
    relevant_count = 0
    powers = np.arange(int(lo), int(hi) + 1)

    while candidate_pool:
        # Pick the candidate that maximises the minimum log spacing.
        deltas = [_delta(c) for c in candidate_pool]
        best_idx = int(np.argmax(deltas))
        steps.append(float(candidate_pool.pop(best_idx)))

        # Regenerate break set: outer product of base^powers and steps.
        breaks = np.sort(
            np.asarray([b * s for b in base ** powers for s in steps], dtype=float)
        )
        mask = (base ** rng[0] <= breaks) & (breaks <= base ** rng[1])
        relevant_count = int(np.sum(mask))
        if relevant_count >= (n - 2):
            break

    if relevant_count >= (n - 2):
        # Include one extra break on each side when available, mirroring
        # R's `lower_end` / `upper_end` logic.
        inside = np.where((base ** rng[0] <= breaks) & (breaks <= base ** rng[1]))[0]
        if inside.size == 0:
            return np.array([], dtype=float)
        lower_end = max(int(inside.min()) - 1, 0)
        upper_end = min(int(inside.max()) + 1, len(breaks) - 1)
        return breaks[lower_end:upper_end + 1]

    # Fallback: Wilkinson extended on the original (exp'd) range.
    from .breaks import breaks_extended
    return breaks_extended(n=n)(np.asarray([base ** rng[0], base ** rng[1]]))


def breaks_log(
    n: int = 5,
    base: float = 10,
) -> Callable[[ArrayLike], np.ndarray]:
    """Create a break generator for log-scaled axes.

    Faithful port of R's ``breaks_log``:

    1. Take ``range(x)`` ignoring non-finite.  If not finite → empty.
    2. Compute integer-power candidates ``base^seq(floor(lo), ceil(hi))``.
    3. If fewer than ``n - 2`` fall inside the data range, shrink the
       step ``by`` from ``floor((max - min) / n) + 1`` down to 1; when
       that's still not enough, delegate to ``log_sub_breaks``.

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
    """

    def _breaks(x: ArrayLike) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        finite = x_arr[np.isfinite(x_arr) & (x_arr > 0)]
        if finite.size == 0:
            return np.array([], dtype=float)

        raw_rng = (float(np.min(finite)), float(np.max(finite)))
        rng = (math.log(raw_rng[0], base), math.log(raw_rng[1], base))
        lo = math.floor(rng[0])
        hi = math.ceil(rng[1])

        if hi == lo:
            return np.array([base ** lo], dtype=float)

        by = math.floor((hi - lo) / n) + 1
        while by >= 1:
            powers = np.arange(lo, hi + by, by)
            breaks = base ** powers
            mask = (base ** rng[0] <= breaks) & (breaks <= base ** rng[1])
            if int(np.sum(mask)) >= (n - 2):
                return breaks
            by -= 1

        return _log_sub_breaks(rng, n=n, base=base)

    return _breaks


log_breaks = breaks_log


# ---------------------------------------------------------------------------
# minor_breaks_log
# ---------------------------------------------------------------------------

def minor_breaks_log(
    detail: Optional[int] = None,
    smallest: Optional[float] = None,
) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    """Create a minor-break generator for log-10 axes.

    Faithful port of R's ``minor_breaks_log``:

    * ``detail`` must be one of ``{1, 5, 10}`` if given.
      - ``10`` → tens ladder only (``10^k``).
      - ``5``  → tens plus mid-decade ``5 * 10^k``.
      - ``1``  → tens plus fives plus ones (``1..9 * 10^k``).
    * When ``detail`` is ``None``, the function auto-selects based on
      the number of decades covered: ``> 15`` → 10, ``> 8`` → 5, else 1.
    * When any data value is ``<= 0`` (so a signed-log scale such as
      ``asinh`` is in use), the ladder is reflected about zero and a
      ``0`` tick is added.

    Parameters
    ----------
    detail : int, optional
        One of 1, 5, or 10.
    smallest : float, optional
        Smallest absolute value to include when negatives are present.
        Defaults to ``min(1, max(|x|)) * 0.1``.

    Returns
    -------
    callable
        Function ``(x, ...) -> numpy.ndarray``.  The returned array
        carries a ``.detail`` attribute (per-tick 10/5/1) mirroring R.
    """
    if detail is not None and detail not in (1, 5, 10):
        raise ValueError("detail must be one of 1, 5, or 10")
    if smallest is not None:
        if not np.isfinite(smallest) or smallest <= 0 or smallest < 1e-100:
            raise ValueError(
                "smallest must be a finite positive number >= 1e-100"
            )

    def _minor_breaks(x: np.ndarray, *args, **kwargs) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        has_negatives = bool(np.any(x_arr <= 0))

        if has_negatives:
            large = float(np.nanmax(np.abs(x_arr)))
            small = smallest if smallest is not None else min(1.0, large) * 0.1
            # Work with (small*10, large) as the effective positive range.
            x_used = np.sort(np.array([small * 10.0, large]))
        else:
            x_used = x_arr[x_arr > 0]
            small = None

        start = int(math.floor(math.log10(np.min(x_used)))) - 1
        end = int(math.ceil(math.log10(np.max(x_used)))) + 1

        if detail is None:
            # R: findInterval(abs(end - start), c(8, 15), left.open=TRUE) + 1
            span = abs(end - start)
            if span > 15:
                chosen = 10
            elif span > 8:
                chosen = 5
            else:
                chosen = 1
        else:
            chosen = detail

        ladder = np.array(
            [10.0 ** p for p in range(start, end + 1)], dtype=float
        )

        tens = np.array([], dtype=float)
        fives = np.array([], dtype=float)
        ones = np.array([], dtype=float)
        if chosen in (10, 5, 1):
            tens = ladder
        if chosen in (5, 1):
            fives = 5.0 * ladder
        if chosen == 1:
            ones = np.array(
                [m * p for p in ladder for m in range(1, 10)], dtype=float
            )
            ones = np.setdiff1d(ones, np.concatenate([tens, fives]))

        if has_negatives:
            tens = tens[tens >= small]
            tens = np.concatenate([tens, -tens, np.array([0.0])])
            fives = fives[fives >= small]
            fives = np.concatenate([fives, -fives])
            ones = ones[ones >= small]
            ones = np.concatenate([ones, -ones])

        ticks = np.concatenate([tens, fives, ones])
        # R attaches detail codes (10/5/1) per tick via attr().
        # numpy arrays don't take arbitrary attributes on the stock
        # dtype, but subclassing works. We attach via np.ndarray view.
        detail_codes = np.concatenate([
            np.full(len(tens), 10, dtype=int),
            np.full(len(fives), 5, dtype=int),
            np.full(len(ones), 1, dtype=int),
        ])
        result = ticks.view(np.ndarray)
        try:
            result = result.copy()
            result.detail = detail_codes  # type: ignore[attr-defined]
        except AttributeError:
            pass
        return result

    return _minor_breaks
