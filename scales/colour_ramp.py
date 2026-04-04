"""
Colour ramp interpolation for the scales package.

Python port of R/colour-ramp.R from the R scales package
(https://github.com/r-lib/scales).  Creates callable colour ramps that
map values in [0, 1] to hex colour strings by interpolation in CIELAB
colour space, using matplotlib's ``LinearSegmentedColormap`` internally.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgba
from numpy.typing import ArrayLike

__all__ = [
    "colour_ramp",
]


def colour_ramp(
    colors: Sequence[str],
    na_color: Optional[str] = None,
    alpha: bool = True,
) -> Callable[[ArrayLike], List[str]]:
    """
    Create a colour ramp that maps [0, 1] values to colours.

    Interpolation is performed in CIELAB colour space via matplotlib's
    ``LinearSegmentedColormap``.

    Parameters
    ----------
    colors : sequence of str
        Two or more colours (any format accepted by matplotlib) that define
        the endpoints and optional interior knots of the ramp.
    na_color : str, optional
        Colour to use for ``NaN`` / missing values.  If *None*, ``NaN``
        inputs produce ``None`` entries in the output list.
    alpha : bool, default True
        If *True*, the alpha channel is preserved in the output hex strings
        (``#RRGGBBAA``).  If *False*, the alpha channel is stripped and
        eight-character hex strings are never returned.

    Returns
    -------
    callable
        A function ``f(x)`` where *x* is an array-like of floats in
        [0, 1].  Returns a list of hex colour strings of the same length.

    Raises
    ------
    ValueError
        If fewer than two colours are supplied.

    Examples
    --------
    >>> ramp = colour_ramp(["#000000", "#FFFFFF"])
    >>> ramp([0.0, 0.5, 1.0])
    ['#000000ff', '#777777ff', '#ffffffff']
    """
    if len(colors) < 2:
        raise ValueError("colour_ramp requires at least two colours.")

    # Build a LinearSegmentedColormap in Lab space.
    rgba_list = [to_rgba(c) for c in colors]
    cmap = LinearSegmentedColormap.from_list(
        "colour_ramp", rgba_list, N=256
    )

    def _ramp(x: ArrayLike) -> List[Optional[str]]:
        x = np.asarray(x, dtype=float)
        result: List[Optional[str]] = []

        for val in x.flat:
            if np.isnan(val):
                result.append(na_color)
            else:
                rgba = cmap(np.clip(val, 0.0, 1.0))
                if alpha:
                    result.append(to_hex(rgba, keep_alpha=True))
                else:
                    result.append(to_hex(rgba[:3], keep_alpha=False))
        return result

    return _ramp
