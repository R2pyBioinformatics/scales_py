"""
Colour mapping functions for the scales package.

Python port of R/colour-mapping.R from the R scales package
(https://github.com/r-lib/scales).  Provides factory functions that return
callables mapping data values to hex colour strings:

- :func:`col_numeric` -- continuous linear interpolation
- :func:`col_bin` -- binned (stepped) colour mapping
- :func:`col_quantile` -- quantile-based binning
- :func:`col_factor` -- categorical / factor mapping
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from numpy.typing import ArrayLike

from .colour_ramp import colour_ramp

__all__ = [
    "col_numeric",
    "col_bin",
    "col_quantile",
    "col_factor",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_palette_func(
    pal: Union[str, Sequence[str]],
    n: Optional[int] = None,
) -> Callable[[ArrayLike], List[str]]:
    """
    Convert a palette specification to a callable ramp function.

    Parameters
    ----------
    pal : str or sequence of str
        Either the name of a matplotlib colourmap (e.g. ``"Blues"``,
        ``"viridis"``) or an explicit list of colour strings.
    n : int, optional
        Number of discrete colours to sample when *pal* is a colourmap name.
        Not used when *pal* is already a list of colours.

    Returns
    -------
    callable
        A function that maps an array of floats in [0, 1] to hex strings.
    """
    if isinstance(pal, str):
        cmap = matplotlib.colormaps.get_cmap(pal)

        def _cmap_ramp(x: ArrayLike) -> List[str]:
            x = np.asarray(x, dtype=float)
            result: List[str] = []
            for val in x.flat:
                if np.isnan(val):
                    result.append(None)  # type: ignore[arg-type]
                else:
                    rgba = cmap(np.clip(val, 0.0, 1.0))
                    result.append(mcolors.to_hex(rgba, keep_alpha=True))
            return result

        return _cmap_ramp
    else:
        return colour_ramp(list(pal))


def _safe_palette_func(
    pal: Union[str, Sequence[str]],
    na_color: str,
    n: Optional[int] = None,
) -> Callable[[ArrayLike], List[str]]:
    """
    Wrap a palette function with NA handling.

    Missing / out-of-domain values are replaced by *na_color*.

    Parameters
    ----------
    pal : str or sequence of str
        Palette specification (see :func:`_to_palette_func`).
    na_color : str
        Hex colour for missing values.
    n : int, optional
        Forwarded to :func:`_to_palette_func`.

    Returns
    -------
    callable
        Palette function with NA safety.
    """
    ramp = _to_palette_func(pal, n=n)

    def _safe(x: ArrayLike) -> List[str]:
        result = ramp(x)
        return [na_color if c is None else c for c in result]

    return _safe


def _rescale(
    x: np.ndarray,
    domain_min: float,
    domain_max: float,
) -> np.ndarray:
    """Linearly rescale *x* from *[domain_min, domain_max]* to [0, 1]."""
    rng = domain_max - domain_min
    if rng == 0:
        return np.where(np.isnan(x), np.nan, 0.5)
    return (x - domain_min) / rng


def _pretty_breaks(domain_min: float, domain_max: float, n: int) -> np.ndarray:
    """
    Compute *n* + 1 "pretty" breakpoints spanning the domain.

    Uses numpy's ``linspace`` rounded to a clean step size that resembles
    R's ``pretty()`` heuristic.
    """
    raw_step = (domain_max - domain_min) / n
    if raw_step == 0:
        return np.array([domain_min, domain_max])

    magnitude = 10 ** np.floor(np.log10(raw_step))
    residual = raw_step / magnitude
    if residual <= 1.5:
        nice_step = 1.0 * magnitude
    elif residual <= 3.0:
        nice_step = 2.0 * magnitude
    elif residual <= 7.0:
        nice_step = 5.0 * magnitude
    else:
        nice_step = 10.0 * magnitude

    lo = np.floor(domain_min / nice_step) * nice_step
    hi = np.ceil(domain_max / nice_step) * nice_step
    return np.arange(lo, hi + nice_step * 0.5, nice_step)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def col_numeric(
    palette: Union[str, Sequence[str]],
    domain: Optional[Tuple[float, float]] = None,
    na_color: str = "#808080",
    reverse: bool = False,
) -> Callable[[ArrayLike], List[str]]:
    """
    Map continuous numeric values to colours via linear interpolation.

    Parameters
    ----------
    palette : str or sequence of str
        Colourmap name (e.g. ``"Blues"``, ``"viridis"``) or a list of
        colour strings defining the ramp.
    domain : tuple of (float, float), optional
        ``(min, max)`` of the data domain.  If *None*, the domain is
        inferred from the first call.
    na_color : str, default "#808080"
        Colour for missing / ``NaN`` values.
    reverse : bool, default False
        Reverse the palette direction.

    Returns
    -------
    callable
        ``f(x)`` mapping numeric array to a list of hex colour strings.

    Examples
    --------
    >>> f = col_numeric(["white", "red"], domain=(0, 100))
    >>> f([0, 50, 100])  # doctest: +SKIP
    ['#ffffffff', '#ff8080ff', '#ff0000ff']
    """
    ramp = _safe_palette_func(palette, na_color)

    # Mutable state for auto-domain
    state: Dict[str, Any] = {
        "domain": domain,
    }

    def _map(x: ArrayLike) -> List[str]:
        x = np.asarray(x, dtype=float)

        if state["domain"] is None:
            finite = x[np.isfinite(x)]
            if len(finite) == 0:
                return [na_color] * x.size
            state["domain"] = (float(finite.min()), float(finite.max()))

        lo, hi = state["domain"]
        scaled = _rescale(x, lo, hi)
        if reverse:
            scaled = 1.0 - scaled
        # Clamp to [0, 1]; NaN stays NaN
        scaled = np.where(np.isnan(scaled), np.nan, np.clip(scaled, 0, 1))
        return ramp(scaled)

    return _map


def col_bin(
    palette: Union[str, Sequence[str]],
    domain: Optional[Tuple[float, float]] = None,
    bins: Union[int, Sequence[float]] = 7,
    pretty: bool = True,
    na_color: str = "#808080",
    reverse: bool = False,
    right: bool = False,
) -> Callable[[ArrayLike], List[str]]:
    """
    Map continuous data to colours through binning.

    Parameters
    ----------
    palette : str or sequence of str
        Palette specification.
    domain : tuple of (float, float), optional
        Data domain.  Required if *bins* is an integer and *pretty* is False.
    bins : int or sequence of float, default 7
        Number of bins or explicit breakpoints.
    pretty : bool, default True
        Use "pretty" breakpoints when *bins* is an integer.
    na_color : str, default "#808080"
        Colour for missing values.
    reverse : bool, default False
        Reverse palette direction.
    right : bool, default False
        If *True*, bins are right-closed ``(a, b]``; otherwise left-closed
        ``[a, b)``.

    Returns
    -------
    callable
        ``f(x)`` mapping numeric array to a list of hex colour strings.
    """
    state: Dict[str, Any] = {
        "breaks": None,
        "ramp": None,
    }

    def _ensure_breaks(x: np.ndarray) -> np.ndarray:
        if state["breaks"] is not None:
            return state["breaks"]

        if not isinstance(bins, int):
            state["breaks"] = np.sort(np.asarray(bins, dtype=float))
            return state["breaks"]

        # Need domain
        dom = domain
        if dom is None:
            finite = x[np.isfinite(x)]
            if len(finite) == 0:
                state["breaks"] = np.array([0.0, 1.0])
                return state["breaks"]
            dom = (float(finite.min()), float(finite.max()))

        if pretty:
            state["breaks"] = _pretty_breaks(dom[0], dom[1], bins)
        else:
            state["breaks"] = np.linspace(dom[0], dom[1], bins + 1)
        return state["breaks"]

    def _map(x: ArrayLike) -> List[str]:
        x = np.asarray(x, dtype=float)
        breaks = _ensure_breaks(x)
        n_bins = len(breaks) - 1
        if n_bins < 1:
            return [na_color] * x.size

        if state["ramp"] is None:
            state["ramp"] = _safe_palette_func(palette, na_color)

        ramp = state["ramp"]

        # Digitize: find bin index for each value
        if right:
            indices = np.digitize(x, breaks, right=True)
        else:
            indices = np.digitize(x, breaks, right=False)

        # Map bin indices to [0, 1] palette positions
        result: List[str] = []
        for i, val in enumerate(x.flat):
            if np.isnan(val):
                result.append(na_color)
                continue

            idx = int(indices.flat[i])
            # digitize with right=False: bin 0 means below first break,
            # bin len(breaks) means above last break.
            if right:
                if idx < 1 or idx > n_bins:
                    result.append(na_color)
                    continue
                bin_idx = idx - 1
            else:
                if idx < 1 or idx > n_bins:
                    result.append(na_color)
                    continue
                bin_idx = idx - 1

            # Centre of the bin mapped to [0, 1]
            frac = (bin_idx + 0.5) / n_bins
            if reverse:
                frac = 1.0 - frac
            colors = ramp(np.array([frac]))
            result.append(colors[0])

        return result

    return _map


def col_quantile(
    palette: Union[str, Sequence[str]],
    domain: Optional[ArrayLike] = None,
    n: int = 4,
    probs: Optional[Sequence[float]] = None,
    na_color: str = "#808080",
    reverse: bool = False,
    right: bool = False,
) -> Callable[[ArrayLike], List[str]]:
    """
    Map quantile-based bins to colours.

    Parameters
    ----------
    palette : str or sequence of str
        Palette specification.
    domain : array-like, optional
        Reference data from which quantiles are computed.  If *None*,
        quantiles are computed on the first call.
    n : int, default 4
        Number of quantile bins (ignored if *probs* is given).
    probs : sequence of float, optional
        Explicit quantile probabilities (e.g. ``[0, 0.25, 0.5, 0.75, 1]``).
    na_color : str, default "#808080"
        Colour for missing values.
    reverse : bool, default False
        Reverse palette direction.
    right : bool, default False
        Bin closure direction.

    Returns
    -------
    callable
        ``f(x)`` mapping numeric array to a list of hex colour strings.
    """
    if probs is None:
        probs_arr = np.linspace(0, 1, n + 1)
    else:
        probs_arr = np.asarray(probs, dtype=float)

    state: Dict[str, Any] = {"breaks": None}

    if domain is not None:
        domain_arr = np.asarray(domain, dtype=float)
        finite = domain_arr[np.isfinite(domain_arr)]
        if len(finite) > 0:
            state["breaks"] = np.unique(np.quantile(finite, probs_arr))

    def _map(x: ArrayLike) -> List[str]:
        x = np.asarray(x, dtype=float)

        if state["breaks"] is None:
            finite = x[np.isfinite(x)]
            if len(finite) == 0:
                return [na_color] * x.size
            state["breaks"] = np.unique(np.quantile(finite, probs_arr))

        # Delegate to col_bin with the computed breaks
        mapper = col_bin(
            palette,
            bins=state["breaks"],
            na_color=na_color,
            reverse=reverse,
            right=right,
        )
        return mapper(x)

    return _map


def col_factor(
    palette: Union[str, Sequence[str]],
    domain: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[str]] = None,
    ordered: bool = False,
    na_color: str = "#808080",
    reverse: bool = False,
) -> Callable[[Union[ArrayLike, Sequence[str]]], List[str]]:
    """
    Map categorical (factor) values to colours.

    Parameters
    ----------
    palette : str or sequence of str
        Palette specification.  When a list of colours, the number of
        colours should ideally match the number of levels.
    domain : sequence of str, optional
        Valid category labels.  If *None*, inferred on first call.
    levels : sequence of str, optional
        Synonym for *domain* (mirrors R's ``levels`` argument).
    ordered : bool, default False
        If *True*, treat categories as ordered and interpolate; otherwise
        assign evenly spaced colours.
    na_color : str, default "#808080"
        Colour for missing / unknown levels.
    reverse : bool, default False
        Reverse palette direction.

    Returns
    -------
    callable
        ``f(x)`` mapping an array of category labels to a list of hex
        colour strings.
    """
    lvls = list(levels) if levels is not None else (
        list(domain) if domain is not None else None
    )

    state: Dict[str, Any] = {"levels": lvls, "colors": None}

    def _ensure_colors(x_levels: List[str]) -> Dict[str, str]:
        if state["colors"] is not None:
            return state["colors"]

        all_levels = state["levels"] if state["levels"] is not None else x_levels
        if reverse:
            all_levels = list(reversed(all_levels))

        n = len(all_levels)
        if n == 0:
            state["colors"] = {}
            return state["colors"]

        ramp = _safe_palette_func(palette, na_color)

        if n == 1:
            positions = np.array([0.5])
        else:
            positions = np.linspace(0, 1, n)

        hex_colors = ramp(positions)
        state["colors"] = dict(zip(all_levels, hex_colors))
        state["levels"] = all_levels
        return state["colors"]

    def _map(x: Union[ArrayLike, Sequence[str]]) -> List[str]:
        if isinstance(x, np.ndarray):
            labels = x.astype(str).tolist()
        else:
            labels = [str(v) for v in x]

        if state["levels"] is None:
            # Discover levels from unique values, preserving first-seen order
            seen: Dict[str, None] = {}
            for lab in labels:
                if lab not in seen:
                    seen[lab] = None
            state["levels"] = list(seen.keys())

        color_map = _ensure_colors(labels)
        return [color_map.get(lab, na_color) for lab in labels]

    return _map
