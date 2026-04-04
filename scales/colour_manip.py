"""
Colour manipulation utilities for the scales package.

Python port of R/colour-manip.R from the R scales package
(https://github.com/r-lib/scales).  Provides helpers for adjusting
transparency, desaturation, HCL manipulation, mixing, and display.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb, to_rgba
from numpy.typing import ArrayLike

__all__ = [
    "alpha",
    "muted",
    "col2hcl",
    "show_col",
    "col_mix",
    "col_shift",
    "col_lighter",
    "col_darker",
    "col_saturate",
]


# ---------------------------------------------------------------------------
# Internal colour-space helpers
# ---------------------------------------------------------------------------

def _linearize(c: float) -> float:
    """sRGB channel [0, 1] -> linear RGB."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _delinearize(c: float) -> float:
    """Linear RGB -> sRGB channel [0, 1]."""
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _rgb_to_lab(r: float, g: float, b: float) -> tuple[float, float, float]:
    """
    Convert sRGB [0, 1] to CIELAB (D65 illuminant).

    Parameters
    ----------
    r, g, b : float
        sRGB colour channels in [0, 1].

    Returns
    -------
    tuple of float
        ``(L, a, b)`` in CIELAB space.
    """
    # sRGB -> linear
    rl = _linearize(r)
    gl = _linearize(g)
    bl = _linearize(b)

    # linear RGB -> XYZ (D65)
    x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl

    # D65 reference white
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def _f(t: float) -> float:
        delta = 6.0 / 29.0
        if t > delta ** 3:
            return t ** (1.0 / 3.0)
        return t / (3.0 * delta * delta) + 4.0 / 29.0

    fx = _f(x / xn)
    fy = _f(y / yn)
    fz = _f(z / zn)

    L = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)
    return L, a_star, b_star


def _lab_to_rgb(
    L: float, a_star: float, b_star: float
) -> tuple[float, float, float]:
    """
    Convert CIELAB (D65) to sRGB [0, 1], clamped.

    Parameters
    ----------
    L, a_star, b_star : float
        CIELAB coordinates.

    Returns
    -------
    tuple of float
        ``(r, g, b)`` clamped to [0, 1].
    """
    xn, yn, zn = 0.95047, 1.0, 1.08883
    delta = 6.0 / 29.0

    fy = (L + 16.0) / 116.0
    fx = fy + a_star / 500.0
    fz = fy - b_star / 200.0

    def _finv(t: float) -> float:
        if t > delta:
            return t ** 3
        return 3.0 * delta * delta * (t - 4.0 / 29.0)

    x = xn * _finv(fx)
    y = yn * _finv(fy)
    z = zn * _finv(fz)

    # XYZ -> linear RGB
    rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    r = np.clip(_delinearize(rl), 0.0, 1.0)
    g = np.clip(_delinearize(gl), 0.0, 1.0)
    b = np.clip(_delinearize(bl), 0.0, 1.0)
    return float(r), float(g), float(b)


def _lab_to_hcl(
    L: float, a: float, b: float
) -> tuple[float, float, float]:
    """
    Convert CIELAB to HCL (cylindrical Lab).

    Returns
    -------
    tuple of float
        ``(H, C, L)`` where H is in degrees [0, 360).
    """
    h = np.degrees(np.arctan2(b, a)) % 360
    c = np.sqrt(a ** 2 + b ** 2)
    return float(h), float(c), L


def _hcl_to_lab(
    h: float, c: float, l: float
) -> tuple[float, float, float]:
    """
    Convert HCL back to CIELAB.

    Parameters
    ----------
    h : float
        Hue in degrees.
    c : float
        Chroma.
    l : float
        Luminance (CIELAB *L*).

    Returns
    -------
    tuple of float
        ``(L, a, b)`` in CIELAB.
    """
    a = c * np.cos(np.radians(h))
    b = c * np.sin(np.radians(h))
    return l, float(a), float(b)


def _hex_to_hcl(colour: str) -> tuple[float, float, float, float]:
    """Return ``(H, C, L, alpha)`` for a colour string."""
    rgba = to_rgba(colour)
    L, a, b = _rgb_to_lab(rgba[0], rgba[1], rgba[2])
    h, c, l = _lab_to_hcl(L, a, b)
    return h, c, l, rgba[3]


def _hcl_to_hex(
    h: float, c: float, l: float, alpha_val: float = 1.0
) -> str:
    """Return hex string from HCL + alpha."""
    L, a, b = _hcl_to_lab(h, c, l)
    r, g, bb = _lab_to_rgb(L, a, b)
    return to_hex((r, g, bb, alpha_val), keep_alpha=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def alpha(
    colour: Union[str, Sequence[str]],
    alpha_value: Union[None, float, Sequence[Optional[float]]] = None,
) -> Union[str, List[str]]:
    """
    Modify the alpha transparency of colour(s).

    Parameters
    ----------
    colour : str or sequence of str
        One or more colour specifications (names, hex codes, etc.).
    alpha_value : float or sequence of float, optional
        New alpha value(s) in [0, 1].  If *None*, colours are returned
        unchanged.

    Returns
    -------
    str or list of str
        Colour(s) with the requested alpha channel, as ``#RRGGBBAA`` hex
        strings.

    Examples
    --------
    >>> alpha("red", 0.5)
    '#ff000080'
    """
    scalar_input = isinstance(colour, str)
    colours = [colour] if scalar_input else list(colour)

    if alpha_value is None:
        result = [to_hex(to_rgba(c), keep_alpha=True) for c in colours]
        return result[0] if scalar_input else result

    # Vectorise alpha_value
    if isinstance(alpha_value, (int, float)):
        alphas: List[Optional[float]] = [float(alpha_value)] * len(colours)
    else:
        alphas = list(alpha_value)

    if len(alphas) == 1:
        alphas = alphas * len(colours)
    elif len(colours) == 1:
        colours = colours * len(alphas)

    if len(colours) != len(alphas):
        raise ValueError(
            "colour and alpha_value must have compatible lengths."
        )

    result = []
    for c, a in zip(colours, alphas):
        rgba = to_rgba(c)
        new_alpha = rgba[3] if a is None else float(a)
        result.append(to_hex((*rgba[:3], new_alpha), keep_alpha=True))

    return result[0] if scalar_input and isinstance(alpha_value, (int, float)) else result


def muted(colour: str, l: float = 30, c: float = 70) -> str:
    """
    Desaturate a colour by reducing luminance and chroma in HCL space.

    Parameters
    ----------
    colour : str
        A colour specification.
    l : float, default 30
        Target luminance (CIELAB *L*).
    c : float, default 70
        Target chroma.

    Returns
    -------
    str
        Muted colour as a hex string.

    Examples
    --------
    >>> muted("red")  # doctest: +SKIP
    '#c66565ff'
    """
    h_val, _c, _l, a_val = _hex_to_hcl(colour)
    return _hcl_to_hex(h_val, c, l, a_val)


def col2hcl(
    colour: Union[str, Sequence[str]],
    h: Optional[float] = None,
    c: Optional[float] = None,
    l: Optional[float] = None,
    alpha_value: Optional[float] = None,
) -> Union[str, List[str]]:
    """
    Convert colour(s) to HCL, optionally overriding components, and return hex.

    Parameters
    ----------
    colour : str or sequence of str
        Input colour(s).
    h, c, l : float, optional
        Override hue, chroma, and/or luminance.
    alpha_value : float, optional
        Override alpha.

    Returns
    -------
    str or list of str
        Hex colour string(s).
    """
    scalar_input = isinstance(colour, str)
    colours = [colour] if scalar_input else list(colour)

    result = []
    for col in colours:
        hv, cv, lv, av = _hex_to_hcl(col)
        hv = h if h is not None else hv
        cv = c if c is not None else cv
        lv = l if l is not None else lv
        av = alpha_value if alpha_value is not None else av
        result.append(_hcl_to_hex(hv, cv, lv, av))

    return result[0] if scalar_input else result


def show_col(
    colours: Sequence[str],
    labels: bool = True,
    borders: Optional[str] = None,
    cex_label: float = 1.0,
    ncol: Optional[int] = None,
) -> None:
    """
    Display colours in a rectangular grid using matplotlib.

    Parameters
    ----------
    colours : sequence of str
        Colour specifications to display.
    labels : bool, default True
        Whether to print the hex code inside each swatch.
    borders : str, optional
        Border colour for each swatch.  *None* means no visible border.
    cex_label : float, default 1.0
        Relative text size for labels.
    ncol : int, optional
        Number of columns.  If *None*, a roughly square layout is chosen.
    """
    n = len(colours)
    if n == 0:
        return

    if ncol is None:
        ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(
        nrow, ncol, figsize=(ncol * 1.2, nrow * 1.2),
        squeeze=False,
    )

    for idx in range(nrow * ncol):
        row, col_idx = divmod(idx, ncol)
        ax = axes[row][col_idx]
        ax.set_xticks([])
        ax.set_yticks([])

        if idx < n:
            ax.set_facecolor(colours[idx])
            edge = borders if borders is not None else "none"
            for spine in ax.spines.values():
                spine.set_edgecolor(edge)
                spine.set_linewidth(1 if borders else 0)
            if labels:
                # Choose readable text colour
                r, g, b = to_rgb(colours[idx])
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                text_col = "white" if lum < 0.5 else "black"
                ax.text(
                    0.5, 0.5,
                    to_hex(colours[idx]),
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=8 * cex_label,
                    color=text_col,
                )
        else:
            ax.set_visible(False)

    fig.tight_layout(pad=0.5)
    plt.show()


def col_mix(
    a: str,
    b: str,
    amount: float = 0.5,
    space: str = "rgb",
) -> str:
    """
    Mix two colours.

    Parameters
    ----------
    a, b : str
        Colour specifications.
    amount : float, default 0.5
        Mixing fraction.  0 returns *a*, 1 returns *b*.
    space : str, default "rgb"
        Colour space for interpolation (``"rgb"`` or ``"lab"``).

    Returns
    -------
    str
        Mixed colour as a hex string.
    """
    rgba_a = to_rgba(a)
    rgba_b = to_rgba(b)

    if space == "lab":
        La, aa, ba = _rgb_to_lab(*rgba_a[:3])
        Lb, ab, bb = _rgb_to_lab(*rgba_b[:3])
        Lm = La + amount * (Lb - La)
        am = aa + amount * (ab - aa)
        bm = ba + amount * (bb - ba)
        r, g, bl = _lab_to_rgb(Lm, am, bm)
        alpha_m = rgba_a[3] + amount * (rgba_b[3] - rgba_a[3])
        return to_hex((r, g, bl, alpha_m), keep_alpha=True)
    else:
        mixed = tuple(
            rgba_a[i] + amount * (rgba_b[i] - rgba_a[i]) for i in range(4)
        )
        return to_hex(mixed, keep_alpha=True)


def col_shift(col: str, amount: float = 10) -> str:
    """
    Shift the hue of a colour.

    Parameters
    ----------
    col : str
        A colour specification.
    amount : float, default 10
        Degrees to shift hue.

    Returns
    -------
    str
        Hue-shifted colour as a hex string.
    """
    h, c, l, a = _hex_to_hcl(col)
    return _hcl_to_hex((h + amount) % 360, c, l, a)


def col_lighter(col: str, amount: float = 10) -> str:
    """
    Increase the luminance of a colour.

    Parameters
    ----------
    col : str
        A colour specification.
    amount : float, default 10
        Amount to add to luminance (*L* in CIELAB).

    Returns
    -------
    str
        Lighter colour as a hex string.
    """
    h, c, l, a = _hex_to_hcl(col)
    return _hcl_to_hex(h, c, max(0, min(100, l + amount)), a)


def col_darker(col: str, amount: float = 10) -> str:
    """
    Decrease the luminance of a colour.

    Equivalent to ``col_lighter(col, -amount)``.

    Parameters
    ----------
    col : str
        A colour specification.
    amount : float, default 10
        Amount to subtract from luminance.

    Returns
    -------
    str
        Darker colour as a hex string.
    """
    return col_lighter(col, -amount)


def col_saturate(col: str, amount: float = 10) -> str:
    """
    Increase the chroma (saturation) of a colour.

    Parameters
    ----------
    col : str
        A colour specification.
    amount : float, default 10
        Amount to add to chroma.

    Returns
    -------
    str
        More saturated colour as a hex string.
    """
    h, c, l, a = _hex_to_hcl(col)
    return _hcl_to_hex(h, max(0, c + amount), l, a)
